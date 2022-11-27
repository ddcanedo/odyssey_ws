# Author: Daniel Canedo
#
# Project: ODYSSEY
#
# Summary: Web service for the Machine Learning algorithms

import os
import sys
import csv
import numpy as np 
import cv2
import rasterio
from PIL import Image
import torch
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
import laspy
import pickle
from flask import request, Flask
import json
from jakteristics import las_utils, compute_features, FEATURE_NAMES
from shapely.geometry import Polygon
from shapely.wkt import loads
import random

csv.field_size_limit(sys.maxsize)
app = Flask(__name__)
Image.MAX_IMAGE_PIXELS = None

# Converts polygons to bounding boxes | Expected format is MULTIPOLYGON
def poly2bb(annotations, xMinImg, xMaxImg, yMaxImg, yMinImg, width, height):
	bbs = {}
	for key in annotations:
		bbs[key] = []
		raw = annotations[key].replace(", ", ",").split(",")

		raw[0] = raw[0].strip(raw[0][:raw[0].find("(")] + "(((")
		raw[len(raw)-1] = raw[len(raw)-1][:len(raw[len(raw)-1])-2]

		xPoints = []
		yPoints = []
		for i in range(0, len(raw)):
			if raw[i][len(raw[i])-1] != ")":
				if raw[i][0] != "(":
					xPoints.append(float(raw[i].split(" ")[0]))
				else:
					xPoints.append(float(raw[i].split(" ")[0].strip("((")))
				yPoints.append(float(raw[i].split(" ")[1]))
			else:
				xPoints.append(float(raw[i].split(" ")[0]))
				yPoints.append(float(raw[i].split(" ")[1].strip(")")))

				xMin = min(xPoints)
				xMax = max(xPoints)
				yMin = min(yPoints)
				yMax = max(yPoints)

				# The bounding box must be within the image limits
				if xMin >= xMinImg and xMax <= xMaxImg and yMin >= yMinImg and yMax <= yMaxImg:

					# Maps coordinates from GIS reference to image pixels
					xMinBb = int(map(xMin, xMinImg, xMaxImg, 0, width))
					xMaxBb = int(map(xMax, xMinImg, xMaxImg, 0, width))
					yMaxBb = int(map(yMin, yMinImg, yMaxImg, height, 0))
					yMinBb = int(map(yMax, yMinImg, yMaxImg, height, 0))
					xPoints = []
					yPoints = []

					bbs[key].append((xMinBb, xMaxBb, yMinBb, yMaxBb))
	return bbs


# Converts coordinates from GIS reference to image pixels
def map(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2

# Intersection over Union
def getIou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[2], boxB[2])
	xB = min(boxA[1], boxB[1])
	yB = min(boxA[3], boxB[3])
	# Computes the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# Computes the area of both rectangles
	boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
	boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)
	# Computes the IoU
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

# Convert the detection to real world coordinates
def convert2GIS(xyxy, cropExtent, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg):

	xMin = map(int(xyxy[0]), 0, resolution, cropExtent[0], cropExtent[1])
	xMax = map(int(xyxy[2]), 0, resolution, cropExtent[0], cropExtent[1])
	yMin = map(int(xyxy[3]), 0, resolution, cropExtent[2], cropExtent[3])
	yMax = map(int(xyxy[1]), 0, resolution, cropExtent[2], cropExtent[3])

	xMin = map(xMin, 0, width, xMinImg, xMaxImg)
	xMax = map(xMax, 0, width, xMinImg, xMaxImg)
	yMax = map(yMax, height, 0, yMinImg, yMaxImg)
	yMin = map(yMin, height, 0, yMinImg, yMaxImg)

	return (xMin, xMax, yMin, yMax), "((" + str(xMin) + " " + str(yMin) + "," + str(xMin) + " " + str(yMax) + "," + str(xMax) + " " + str(yMax) + "," + str(xMax) + " " + str(yMin) + "))"


# Validates a detection using the Point Clouds
def pointCloud(validationModel, pointClouds, cropExtent, className, bb):
	tmp = ""
	for cloud in os.listdir(pointClouds):
		tmp = pointClouds + "/" + cloud
		break

	# Creates empty .las file to later populate it with points
	with laspy.open(tmp) as f:
		w = laspy.open("tmp.las", mode="w", header = f.header)
		w.close()

	count = 0
	# Iterates over the point clouds
	with laspy.open("tmp.las", mode = "a") as w:
		for cloud in os.listdir(pointClouds):
			with laspy.open(pointClouds + "/" + cloud) as f:
				# Checks if there is an overlap with the cropped image and the point cloud
				if bb[0] <= f.header.x_max and bb[1] >= f.header.x_min and bb[2] <= f.header.y_max and bb[3] >= f.header.y_min:
					# Appends the points of the overlapping region to the previously created .las file
					las = f.read()          
					x, y = las.points.x.copy(), las.points.y.copy()
					mask = (x >= bb[0]) & (x <= bb[1]) & (y >= bb[2]) & (y <= bb[3])
					roi = las.points[mask]
					w.append_points(roi)
					count += 1
	
	# If temporary las was populated with points
	if count > 0:
		xyz = las_utils.read_las_xyz("tmp.las")

		# Compute 3D features
		features = compute_features(xyz, search_radius=3)
		
		if np.isnan(features).any() == False:

			stats = {}
			for i in FEATURE_NAMES:
				stats[i] = []
			
			for feature in features:
				for i in range(len(FEATURE_NAMES)):
					stats[FEATURE_NAMES[i]].append(feature[i])

			# Each point contributes to 14 features which is too heavy, therefore calculate
			# the mean and standard deviation of of every feature for each point
			X = []
			for i in FEATURE_NAMES:		
				mean = np.mean(stats[i])
				stdev = np.std(stats[i])
				X += [mean,stdev]

			# Removes temporary las
			os.remove("tmp.las")
			
			# 1 is validated, -1 is not validated
			if validationModel.predict([X]) == -1:
				return False
			else:
				return True

	# Return -1 if there are no Point Clouds in this region
	return -1


# Returns a list with 100% visible objects and their respective labels
def checkVisibility(image, crop, processedObjects, bbs):
	visibleObjects = []

	for key in bbs:
		for bb in bbs[key]:
			# The object is 100% inside the cropped image
			if bb[0] >= crop[0] and bb[1] <= crop[1] and bb[2] >= crop[2] and bb[3] <= crop[3]:
				# The object was already processed
				if bb in processedObjects:
					return []
				else:
					visibleObjects.append(bb)
			# The object is 100% outside the cropped image
			elif bb[1] < crop[0] or bb[0] > crop[1] or bb[3] < crop[2] or bb[2] > crop[3]:
				continue
			# The object is partially visible
			else:
				return []

	# Update list of processed objects
	processedObjects.extend(visibleObjects)
	return visibleObjects


# Checks if bounding box is intersecting cropped image
def intersection(bb, crop):
    return not (bb[1] < crop[0] or bb[0] > crop[1] or bb[3] < crop[2] or bb[2] > crop[3])


# Return the LBR polygons that intersect this region
def LBRroi(polygons, bb):
	p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	intersection = []
	for polygon in polygons:
		if polygon.intersects(p):
			intersection.append(polygon.intersection(p))
	return intersection

# Chekcs if the detection is intersecting a LBR polygon
def LBR(roi, bb):
	p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	for polygon in roi:
		if polygon.intersects(p):
			return True
	return False

# Get the extent resulting from the intersection between the user selected region and the loaded image
def getExtent(img, extent, coords, width, height):
	x1 = max(extent[0], coords[0])
	y1 = max(extent[1], coords[1])
	x2 = min(extent[2], coords[2])
	y2 = min(extent[3], coords[3])

	xMin = int(map(x1, extent[0], extent[2], 0, width))
	xMax = int(map(x2, extent[0], extent[2], 0, width))
	yMax = int(map(y1, extent[1], extent[3], height, 0))
	yMin = int(map(y2, extent[1], extent[3], height, 0))

	return (xMin, xMax, yMin, yMax)

# Creates a dataset folder in YOLO format
def createDatasetDir(datasetPath, imagesPath, labelsPath, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath):
	if not os.path.exists(datasetPath):
		os.makedirs(datasetPath)
		os.makedirs(imagesPath)
		os.makedirs(labelsPath)
		os.makedirs(imagesTrainPath)
		os.makedirs(imagesValPath)
		os.makedirs(labelsTrainPath)
		os.makedirs(labelsValPath)
	else:
		if not os.path.exists(imagesPath):
			os.makedirs(imagesPath)
			os.makedirs(imagesTrainPath)
			os.makedirs(imagesValPath)
		else:
			if not os.path.exists(imagesTrainPath):
				os.makedirs(imagesTrainPath)
			if not os.path.exists(imagesValPath):
				os.makedirs(imagesValPath)
			
		if not os.path.exists(labelsPath):
			os.makedirs(labelsPath)
			os.makedirs(labelsTrainPath)
			os.makedirs(labelsValPath)
		else:
			if not os.path.exists(labelsTrainPath):
				os.makedirs(labelsTrainPath)
			if not os.path.exists(labelsValPath):
				os.makedirs(labelsValPath)



@app.route("/", methods=["GET", "POST"])
def main():

	# Inference =============================================================================
	if request.form["purpose"] == "inference":

		# Load annotations, paths and coordinates
		annotations = json.loads(request.form["annotations"])
		images = json.loads(request.form["geotiff"])
		coordinates = json.loads(request.form["coords"])

		# Standard YOLO resolution
		resolution = 640

		imgsz=resolution  # inference size (pixels)
		conf_thres=0.25  # confidence threshold
		iou_thres=0.45  # NMS IOU threshold
		max_det=1000  # maximum detections per image
		classes=None  # filter by class: --class 0, or --class 0 2 3
		agnostic_nms=False  # class-agnostic NMS
		cudnn.benchmark = True  # set True to speed up constant image size inference
		device = "cpu"
		device = select_device(device)

		# YOLO model
		weights = "best.pt"

		# LBR model
		polygonsCsv = "LBR.csv"

		# Get the polygons from the LBR model
		polygons = []
		with open(polygonsCsv) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				p = loads(row["WKT"])
				polygons.append(p)

		# Validation model trained with Point Clouds
		validationModel = pickle.load(open("pointCloud.sav", "rb"))

		# Load YOLO model
		model = attempt_load(weights, map_location=device)
		stride = int(model.stride.max())  # model stride
		names = model.module.names if hasattr(model, "module") else model.names  # get class names

		# Dictionary to store detections based on the class
		aux = {}
		for name in names:
			aux[name] = []

		# Path to the Point Clouds folder for the validation process
		pointClouds = "../ODYSSEY/LAS"

		
		#validated = 0
		#detections = 0

		# Iterates over the images
		for image in images:
			# Load image
			img = Image.open(image).convert("RGB")
			geoRef = rasterio.open(image)
		
			# Parse corners of the image (GIS reference)
			xMinImg = geoRef.bounds[0]
			xMaxImg = geoRef.bounds[2]
			yMinImg = geoRef.bounds[1]
			yMaxImg = geoRef.bounds[3]

			width, height = img.size

			# Get the extent resulting from the intersection between the user selected region and the loaded image
			extent = getExtent(img, (xMinImg, yMinImg, xMaxImg, yMaxImg), coordinates, width, height)

			# Transforms the polygons into bounding boxes
			bbs = poly2bb(annotations, xMinImg, xMaxImg, yMaxImg, yMinImg, width, height)

			# Sliding window going through the extent of interest
			for i in range(extent[0], extent[1], resolution):
				for j in range(extent[2], extent[3], resolution):

					croppedOriginalImg = img.crop((i, j, resolution+i, resolution+j))
					cropExtent = [i, resolution+i, j, resolution+j]

					croppedImg = np.array(croppedOriginalImg)
					displayImg = croppedImg.copy()

					# Maps the cropped image extent from pixels to real world coordinates
					xMin = map(cropExtent[0], 0, width, xMinImg, xMaxImg)
					xMax = map(cropExtent[1], 0, width, xMinImg, xMaxImg)
					yMax = map(cropExtent[2], height, 0, yMinImg, yMaxImg)
					yMin = map(cropExtent[3], height, 0, yMinImg, yMaxImg)

					# Return the LBR polygons that intersect this region
					roiPolygons = LBRroi(polygons, (xMin,xMax,yMin,yMax))

					# If so, attempts to detect objects in that region
					if len(roiPolygons) > 0:
						roiBbs = []
						for key in bbs:
							for bb in bbs[key]:
								# Saves annotated objects in this region to a list
								if intersection(bb, cropExtent):
									xMin = int(map(bb[0], i, i+resolution, 0, resolution))
									xMax = int(map(bb[1], i, i+resolution, 0, resolution))
									yMin = int(map(bb[2], j, j+resolution, 0, resolution))
									yMax = int(map(bb[3], j, j+resolution, 0, resolution))
									roiBbs.append((xMin, xMax, yMin, yMax))
									#cv2.rectangle(displayImg, (xMin, yMin), (xMax,yMax), (255,0,0), 2)

						# Convert croppedImg to YOLO format for inference			
						croppedImg = letterbox(croppedImg, imgsz, stride=stride, auto=True)[0]
						croppedImg = croppedImg.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
						croppedImg = np.ascontiguousarray(croppedImg)
						croppedImg = torch.from_numpy(croppedImg).to(device)
						croppedImg = croppedImg.float()
						croppedImg /= 255  # 0 - 255 to 0.0 - 1.0
						if len(croppedImg.shape) == 3:
							croppedImg = croppedImg[None]  # expand for batch dim

						# Inference
						pred = model(croppedImg)[0]
						pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

						#boxes = 0

						# Iterates over the detections
						for x, det in enumerate(pred):
							if len(det):
								# Rescale boxes from img_size to im0 size
								det[:, :4] = scale_coords(croppedImg.shape[2:], det[:, :4], displayImg.shape).round()

								for *xyxy, conf, cls in reversed(det):
									#cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 1)
									# Convert detection to real world coordinates
									GISbb, strGISbb = convert2GIS(xyxy, cropExtent, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg)
									
									# Chekcs if the detection is intersecting a LBR polygon
									lbr = LBR(roiPolygons, GISbb)

									if lbr:
										c = int(cls)  # integer class
										className = names[c]

										# Validation using Point Clouds
										validation = pointCloud(validationModel, pointClouds, cropExtent, className, GISbb)
										
										# Checks if the detection is already annotated or not
										annotated = False
										for b in roiBbs:
											if getIou(b, (int(xyxy[0]), int(xyxy[2]), int(xyxy[1]), int(xyxy[3]))) > 0.2:
												annotated = True
												break

										#if validation == False:
											#if annotated == False:
												#detections += 1
												#boxes += 1
												#color = (0,0,255)	
												#cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
										
										# Saves detections validated by both Point Clouds and LBR
										if (validation == True or validation == -1) and annotated == False:
											if annotated == False:
												#detections += 1
												#boxes += 1
												#color = (0,255,0)
												#print("Detection validated with point clouds and LBR")
												#validated += 1
												#cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
												aux[className].append(strGISbb)
										#elif validation == -1:
											# there is no point cloud for this region
											#if annotated == False:
												#detections += 1
												#boxes += 1
												#color = (0,255,0)
												#print("Detection validated - No point cloud data here")
												#validated += 1
												#cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
												#aux[className].append(strGISbb)
								
									# debug
									#else:
									#	cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255,255), 2)

									
									

						
						#if boxes > 0:
						#	cv2.imshow("Cropped Image", displayImg)
						#	cv2.waitKey(0)

			img.close()


		#print("[===========================]")
		#print("Detections:", detections)
		#print("Validated detections:", validated)
		#print("[===========================]")

		# Populates dictionary with the validated detections to return 
		data = {}
		for key in aux.keys():
			if len(aux[key]) != 0:
				data[key] = "MULTIPOLYGON ("			
				for i in range(len(aux[key])):
				
					if i != len(aux[key])-1:
						data[key] += aux[key][i] + ", "
					else:
						data[key] += aux[key][i] + ")"
		
		# Deletes temporary las file used for the Point Cloud validation
		if os.path.isfile("tmp.las"):
			os.remove("tmp.las")

		return data


	# Training =============================================================================
	elif request.form["purpose"] == "training":
		# loads annotations and paths
		annotations = json.loads(request.form["annotations"])
		images = json.loads(request.form["geotiff"])

		# Checks if there are annotations
		if not annotations:
			return ("No annotations", 404)


		# Creates dataset folder in YOLOv5 format
		datasetPath = "dataset/"
		imagesPath = datasetPath + "images/"
		labelsPath = datasetPath + "labels/"
		imagesTrainPath = imagesPath + "train/"
		imagesValPath = imagesPath + "val/"
		labelsTrainPath = labelsPath + "train/"
		labelsValPath = labelsPath + "val/"

		createDatasetDir(datasetPath, imagesPath, labelsPath, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath)

		# Standard YOLO resolution
		resolution = 640
		# % for validation
		validationSize = 10

		# associates a YOLO label to a class 
		classes = {}
		label = 0
		classesFile = open(datasetPath + "classes.txt", "a+")
		for key in annotations:
			classes[key] = label
			classesFile.write(str(label) + " - " + key + "\n")
			label+=1
		classesFile.close()

		# Iterates over the images
		for image in images:
			# List to save all the objects that are processed to keep uniqueness
			processedObjects = []

			# Load image
			img = Image.open(image)

			geoRef = rasterio.open(image)
			width, height = img.size

			# Parse corners of the image (GIS reference)
			xMinImg = geoRef.bounds[0]
			xMaxImg = geoRef.bounds[2]
			yMinImg = geoRef.bounds[1]
			yMaxImg = geoRef.bounds[3]

			# Transforms the polygons into bounding boxes
			bbs = poly2bb(annotations, xMinImg, xMaxImg, yMaxImg, yMinImg, width, height)

			# Iterates over the bounding boxes
			for key in bbs:
				for bb in bbs[key]:
					# Check if bounding box is unique
					if bb not in processedObjects:
						# Randomizes a list of unique points covering a range around the object
						x = list(range(bb[1]-resolution//2, bb[0]+resolution//2))
						y = list(range(bb[3]-resolution//2, bb[2]+resolution//2))
						random.shuffle(x)
						random.shuffle(y)

						# List to save 100% visible objects
						visibleObjects = []
						crop = []

						# Iterates over the list of random points
						for i in x:
							if visibleObjects:
								break
							for j in y:
								# Gets a region of interest expanding the random point into a region of interest with a certain resolution
								crop = (i-resolution//2, i+resolution//2, j-resolution//2, j+resolution//2)
								# Checks if that region of interest only covers 100% visible objects
								visibleObjects = checkVisibility(image, crop, processedObjects, bbs)
								if visibleObjects:
									break

						# If we obtain a list of visible objects within a region of interest, we save it
						if visibleObjects:
							# Uses the coordinates as the name of the image and text files
							xMin = int(map(crop[0], 0, width, xMinImg, xMaxImg))
							xMax = int(map(crop[1], 0, width, xMinImg, xMaxImg))
							yMax = int(map(crop[2], height, 0, yMinImg, yMaxImg))
							yMin = int(map(crop[3], height, 0, yMinImg, yMaxImg))
							coords = "("+ str(xMin) + "_" + str(xMax) + "_" + str(yMin) + "_" + str(yMax) + ")"

							# Training/Validation split
							if np.random.uniform(0,1) > validationSize/100:
								labelPath = labelsTrainPath
								imagePath = imagesTrainPath				
							else:
								labelPath = labelsValPath
								imagePath = imagesValPath

							# Writes the image
							croppedImg = img.crop((crop[0], crop[2], crop[1], crop[3]))
							imgName = imagePath + image.split(".")[0].split("/")[-1] + coords + ".png"
							croppedImg.save(imgName)

							# Writes the label file
							txtFile = open(labelPath + image.split(".")[0].split("/")[-1] + coords + ".txt", "a+")
							for i in range(len(visibleObjects)):						
								# Maps the object in YOLO format
								centerX = (visibleObjects[i][0] + visibleObjects[i][1])/2.0
								centerX = map(centerX, crop[0], crop[1], 0, 1)
								centerY = (visibleObjects[i][2] + visibleObjects[i][3])/2.0
								centerY = map(centerY, crop[2], crop[3], 0, 1)
								w  = (centerX - map(visibleObjects[i][0], crop[0], crop[1], 0, 1)) * 2.0
								h = (centerY - map(visibleObjects[i][2], crop[2], crop[3], 0, 1)) * 2.0


								# Writes/Appends the annotations to a text file that has the same name of the respective image
								txtFile.write(str(classes[key]) + " " + str(centerX) + " " + str(centerY) + " " +str(w)+ " "+ str(h) + "\n")

							txtFile.close()

		img.close()

	return ("", 204)

if __name__ == "__main__":
	app.run()