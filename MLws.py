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
from PIL import Image, ImageDraw
import torch
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
import laspy
import pickle
from flask import request, Flask
import json
from jakteristics import las_utils, compute_features, FEATURE_NAMES
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
import random
import pyqtree


csv.field_size_limit(sys.maxsize)
app = Flask(__name__)
Image.MAX_IMAGE_PIXELS = None

# Converts polygons to bounding boxes | Expected format is MULTIPOLYGON
def poly2bb(annotations, xMinImg, xMaxImg, yMaxImg, yMinImg, width, height):
	bbs = {}
	for key in annotations:
		wkt = annotations[key].replace(", ", ",")

		polygonStrings = wkt[16:-3].split(")),((")
		
		# Iterate through the polygon strings
		for polygonStr in polygonStrings:
			# Split the polygon string into point strings
			pointStrings = polygonStr.split(",")
			polygonPoints = []
			xPoints = []
			yPoints = []
			# Iterate through the point strings
			for pointStr in pointStrings:
				# Split the point string into x and y values
				x, y = map(float, pointStr.split(" "))
				xPoints.append(x)
				yPoints.append(y)
				polygonPoints.append([x, y])

			xMin = min(xPoints)
			xMax = max(xPoints)
			yMin = min(yPoints)
			yMax = max(yPoints)

			# The bounding box must be within the image limits
			if xMin >= xMinImg and xMax <= xMaxImg and yMin >= yMinImg and yMax <= yMaxImg:

				# Maps coordinates from GIS reference to image pixels
				xMinBb = round(mapping(xMin, xMinImg, xMaxImg, 0, width))
				xMaxBb = round(mapping(xMax, xMinImg, xMaxImg, 0, width))
				yMaxBb = round(mapping(yMin, yMinImg, yMaxImg, height, 0))
				yMinBb = round(mapping(yMax, yMinImg, yMaxImg, height, 0))

				polygon = []
				for p in polygonPoints:
					xPoly = mapping(p[0], xMinImg, xMaxImg, 0, width)
					yPoly = mapping(p[1], yMinImg, yMaxImg, height, 0)
					polygon.append([xPoly, yPoly])

				bbs[(xMinBb, xMaxBb, yMinBb, yMaxBb)] = [key, polygon]

	return bbs


# Converts coordinates from GIS reference to image pixels
def mapping(value, min1, max1, min2, max2):
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

# Non maximum suppression to deal with overlapped detection due to the sliding window step
def nms(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	boxes = np.array(boxes)
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,2]
	x2 = boxes[:,1]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

# Convert the detection to real world coordinates
def convert2GIS(xyxy, cropExtent, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg):

	xMin = mapping(int(xyxy[0]), 0, resolution, cropExtent[0], cropExtent[1])
	xMax = mapping(int(xyxy[2]), 0, resolution, cropExtent[0], cropExtent[1])
	yMin = mapping(int(xyxy[3]), 0, resolution, cropExtent[2], cropExtent[3])
	yMax = mapping(int(xyxy[1]), 0, resolution, cropExtent[2], cropExtent[3])

	xMin = mapping(xMin, 0, width, xMinImg, xMaxImg)
	xMax = mapping(xMax, 0, width, xMinImg, xMaxImg)
	yMax = mapping(yMax, height, 0, yMinImg, yMaxImg)
	yMin = mapping(yMin, height, 0, yMinImg, yMaxImg)

	return (xMin, xMax, yMin, yMax)


# Saves the corresponding point cloud to the cropped region
def pointCloudCrop(spindex, LASPath, image, coords, crop, visibleObjects, imgName, pointClouds, xMinImg, xMaxImg, yMinImg, yMaxImg, width, height, resolution):
	# Gets one cloud to later use its header to write an empty .las file
	tmp = ''
	for cloud in os.listdir(pointClouds):
		tmp = pointClouds + '/' + cloud
		break

	cloudName = LASPath + image.split("/")[-1].split(".")[0] + coords + ".las"

	# Creates empty .las file to later populate it with points
	clouds = {}
	with laspy.open(tmp) as f:
		for i in range(len(visibleObjects)):
			c = cloudName.split('.')[0] + str(i) + '.las'
			clouds[c] = 0
			w = laspy.open(c, mode='w', header = f.header)
			w.close()

	for i in range(len(visibleObjects)):
		# Gets bounding box in GIS reference
		xMin = mapping(visibleObjects[i][0], 0, width, xMinImg, xMaxImg)
		xMax = mapping(visibleObjects[i][1], 0, width, xMinImg, xMaxImg)
		yMax = mapping(visibleObjects[i][2], height, 0, yMinImg, yMaxImg)
		yMin = mapping(visibleObjects[i][3], height, 0, yMinImg, yMaxImg)

		matches = spindex.intersect((xMin,yMin,xMax,yMax))

		for match in matches:
			with laspy.open(match) as f:
				# Checks if there is an overlap with the cropped image and the point cloud
				if xMin <= f.header.x_max and xMax >= f.header.x_min and yMin <= f.header.y_max and yMax >= f.header.y_min:
					las = f.read()
					# Appends the points of the overlapping region to the previously created .las file
					          
					x, y = las.points[las.classification == 2].x.copy(), las.points[las.classification == 2].y.copy()
					mask = (x >= xMin) & (x <= xMax) & (y >= yMin) & (y <= yMax)
					if True in mask:
						roi = las.points[las.classification == 2][mask]

						with laspy.open(cloudName.split('.')[0] + str(i) + '.las', mode = 'a') as w:
							w.append_points(roi)
							# Updates the previously created .las file header
							if clouds[cloudName.split('.')[0] + str(i) + '.las'] == 0:
								w.header.x_min = np.min(roi.x)
								w.header.y_min = np.min(roi.y)
								w.header.z_min = np.min(roi.z)
								w.header.x_max = np.max(roi.x)
								w.header.y_max = np.max(roi.y)
								w.header.z_max = np.max(roi.z)
							else:
								if w.header.x_min > np.min(roi.x):
									w.header.x_min = np.min(roi.x)
								if w.header.y_min > np.min(roi.y):
									w.header.y_min = np.min(roi.y)
								if w.header.z_min > np.min(roi.z):
									w.header.z_min = np.min(roi.z)
								if w.header.x_max < np.max(roi.x):
									w.header.x_max = np.max(roi.x)
								if w.header.y_max < np.max(roi.y):
									w.header.y_max = np.max(roi.y)
								if w.header.z_max < np.max(roi.z):
									w.header.z_max = np.max(roi.z)

						clouds[cloudName.split('.')[0] + str(i) + '.las'] += 1

	# If .las file was not populated with points, deletes it
	for c in clouds:
		with laspy.open(c) as f:
			las = f.read()
			if clouds[c] == 0:
				os.remove(c)				


# Validates a detection using the Point Clouds
def pointCloud(spindex, validationModel, pointClouds, cropExtent, className, bb):
	tmp = ''
	for cloud in os.listdir(pointClouds):
		tmp = pointClouds + '/' + cloud
		break


	# Creates empty .las file to later populate it with points
	with laspy.open(tmp) as f:
		w = laspy.open('tmp.las', mode='w', header = f.header)
		w.close()

	count = 0
	# Checks if there is an overlap with the cropped image and the point cloud
	matches = spindex.intersect((bb[0], bb[2], bb[1], bb[3]))

	# Iterates over the matched Point Clouds
	with laspy.open('tmp.las', mode = 'a') as w:
		for match in matches:
			with laspy.open(match) as f:
				las = f.read()          
				x, y = las.points[las.classification == 2].x.copy(), las.points[las.classification == 2].y.copy()
				mask = (x >= bb[0]) & (x <= bb[1]) & (y >= bb[2]) & (y <= bb[3])
				# Checks if there is an overlap with the cropped image and the point cloud
				if True in mask:
					roi = las.points[las.classification == 2][mask]
					# Appends the points of the overlapping region to the previously created .las file
					w.append_points(roi)
					count += 1

	# If temporary las was populated with points	
	if count > 0:
		xyz = las_utils.read_las_xyz('tmp.las')
		# Compute 3D features
		features = compute_features(xyz, search_radius=3)
		
		if np.isnan(features).any() == False:
			stats = {}
			for i in FEATURE_NAMES:
				stats[i] = []
			
			for feature in features:
				for i in range(len(FEATURE_NAMES)):
					stats[FEATURE_NAMES[i]].append(feature[i])

			X = []
			# Each point contributes to 14 features which is too heavy, therefore calculate
			# the median, standard deviation, variance, and covariance of every feature for each point
			for i in FEATURE_NAMES:		
				median = np.median(stats[i])
				stdev = np.std(stats[i])
				var = np.var(stats[i])
				cov = np.cov(stats[i])
				X += [median, stdev, var, cov]
			
			# Removes temporary las
			os.remove('tmp.las')
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

	for bb in bbs:
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
def intersectsBb(bb, crop):
    return not (bb[1] < crop[0] or bb[0] > crop[1] or bb[3] < crop[2] or bb[2] > crop[3])


# Return the LBR polygons that intersect this region
def LBRroi(polygons, bb):
	p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	intersection = []
	for polygon in polygons:
		if polygon.intersects(p):
			intersection.append(polygon.intersection(p))
	return intersection

# Checks if the detection is intersecting a LBR polygon
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

	xMin = round(mapping(x1, extent[0], extent[2], 0, width))
	xMax = round(mapping(x2, extent[0], extent[2], 0, width))
	yMax = round(mapping(y1, extent[1], extent[3], height, 0))
	yMin = round(mapping(y2, extent[1], extent[3], height, 0))

	return (xMin, xMax, yMin, yMax)

# Creates a dataset folder in YOLO format
def createDatasetDir(datasetPath, LASPath, imagesPath, labelsPath, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath, labelsPolyTrain, labelsPolyVal):
	if not os.path.exists(datasetPath):
		os.makedirs(datasetPath)
		os.makedirs(LASPath)
		os.makedirs(imagesPath)
		os.makedirs(labelsPath)
		os.makedirs(imagesTrainPath)
		os.makedirs(imagesValPath)
		os.makedirs(labelsTrainPath)
		os.makedirs(labelsValPath)
		os.makedirs(labelsPolyTrain)
		os.makedirs(labelsPolyVal)
	else:
		if not os.path.exists(LASPath):
			os.makedirs(LASPath)

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
			os.makedirs(labelsPolyTrain)
			os.makedirs(labelsPolyVal)
		else:
			if not os.path.exists(labelsTrainPath):
				os.makedirs(labelsTrainPath)
			if not os.path.exists(labelsValPath):
				os.makedirs(labelsValPath)
			if not os.path.exists(labelsPolyTrain):
				os.makedirs(labelsPolyTrain)
			if not os.path.exists(labelsPolyVal):
				os.makedirs(labelsPolyVal)



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

		imgsz=(resolution, resolution) # inference size (pixels)
		conf_thres=0.25  # confidence threshold
		iou_thres=0.45  # NMS IOU threshold
		max_det=1000  # maximum detections per image
		agnostic_nms=False
		classes=None
		bs = 1  # batch
		device = "0"  # 0 for gpu
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
		model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
		stride, names, pt = model.stride, model.names, model.pt
		imgsz = check_img_size(imgsz, s=stride)  # check image size
		model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
		seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

		# Dictionary to store detections based on the class
		aux = {}
		for name in names.values():
			aux[name] = []

		# Path to the Point Clouds folder for the validation process
		pointClouds = "LAS"
		# Index Point Clouds to a tree for faster search later on
		spindex = pyqtree.Index(bbox=(0, 0, 100, 100))
		for cloud in os.listdir(pointClouds):
			with laspy.open(pointClouds + '/' + cloud) as f:
				spindex.insert(pointClouds + '/' + cloud, (f.header.x_min, f.header.y_min, f.header.x_max, f.header.y_max))

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
			for i in range(extent[0], extent[1], resolution//2):
				for j in range(extent[2], extent[3], resolution//2):

					croppedOriginalImg = img.crop((i, j, resolution+i, resolution+j))
					cropExtent = [i, resolution+i, j, resolution+j]

					croppedImg = np.array(croppedOriginalImg)
					displayImg = croppedImg.copy()

					# Maps the cropped image extent from pixels to real world coordinates
					xMin = mapping(cropExtent[0], 0, width, xMinImg, xMaxImg)
					xMax = mapping(cropExtent[1], 0, width, xMinImg, xMaxImg)
					yMax = mapping(cropExtent[2], height, 0, yMinImg, yMaxImg)
					yMin = mapping(cropExtent[3], height, 0, yMinImg, yMaxImg)

					# Return the LBR polygons that intersect this region
					roiPolygons = LBRroi(polygons, (xMin,xMax,yMin,yMax))

					# If so, attempts to detect objects in that region
					if len(roiPolygons) > 0:
						annotatedBbs = []
						for bb in bbs:
							# Saves annotated objects in this region to a list
							if intersectsBb(bb, cropExtent):
								xMin = round(mapping(bb[0], i, i+resolution, 0, resolution))
								xMax = round(mapping(bb[1], i, i+resolution, 0, resolution))
								yMin = round(mapping(bb[2], j, j+resolution, 0, resolution))
								yMax = round(mapping(bb[3], j, j+resolution, 0, resolution))
								annotatedBbs.append((xMin, xMax, yMin, yMax))

						# Convert croppedImg to YOLO format for inference			
						with dt[0]:
							im = letterbox(croppedImg, imgsz, stride=stride, auto=True)[0]
							# Convert
							im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW
							im = np.ascontiguousarray(im)
							im = torch.from_numpy(im).to(model.device)
							im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
							im /= 255  # 0 - 255 to 0.0 - 1.0
							if len(im.shape) == 3:
								im = im[None]  # expand for batch dim

						with dt[1]:
							pred = model(im, augment=False, visualize=False)

						# NMS
						with dt[2]:
							pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

						# Iterates over the detections
						for x, det in enumerate(pred):
							if len(det):
								# Rescale boxes from img_size to im0 size
								det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], displayImg.shape).round()

								for *xyxy, conf, cls in reversed(det):
									# Convert detection to real world coordinates
									GISbb = convert2GIS(xyxy, cropExtent, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg)
									
									# Checks if the detection is intersecting a LBR polygon
									lbr = LBR(roiPolygons, GISbb)

									if lbr:
										c = int(cls)  # integer class
										className = names[c]

										# [!] Warning about the Local Outlier Factor algorithm used for the point cloud validation
										#
										# When novelty is set to True be aware that you must only use predict, decision_function and score_samples 
										# on new unseen data and not on the training samples as this would lead to wrong results. I.e., the result 
										# of predict will not be the same as fit_predict
										#
										# Source: https://scikit-learn.org/stable/modules/outlier_detection.html#outlier-detection
										annotated = False
										for b in annotatedBbs:
											if getIou(b, (int(xyxy[0]), int(xyxy[2]), int(xyxy[1]), int(xyxy[3]))) > 0.2:
												annotated = True
												break

										# Validation using Point Clouds
										validation = False
										if annotated == False:
											validation = pointCloud(spindex, validationModel, pointClouds, cropExtent, className, GISbb)
										
										if validation == True or annotated == True:	
											aux[className].append(GISbb)

			img.close()

		# Populates dictionary with the validated detections to return 
		data = {}
		for key in aux.keys():
			if len(aux[key]) != 0:
				data[key] = "MULTIPOLYGON ("
			finalValidated = nms(aux[key], iou_thres)
			for i in range(len(finalValidated)):
				xMin, xMax, yMin, yMax = finalValidated[i]
				strGISbb = '((' + str(xMin) + ' ' + str(yMin) + ', ' + str(xMin) + ' ' + str(yMax) + ', ' + str(xMax) + ' ' + str(yMax) + ', ' + str(xMax) + ' ' + str(yMin) + '))'
				if i != len(finalValidated)-1:
					data[key] += strGISbb + ", "
				else:
					data[key] += strGISbb + ")"
				
		
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
		LASPath = datasetPath + "LAS/"
		imagesPath = datasetPath + "images/"
		labelsPath = datasetPath + "labels/"
		imagesTrainPath = imagesPath + "train/"
		imagesValPath = imagesPath + "val/"
		labelsTrainPath = labelsPath + "train/"
		labelsValPath = labelsPath + "val/"
		labelsPolyTrain = labelsPath + "trainPoly/"
		labelsPolyVal = labelsPath + "valPoly/"

		createDatasetDir(datasetPath, LASPath, imagesPath, labelsPath, imagesTrainPath, imagesValPath, labelsTrainPath, labelsValPath, labelsPolyTrain, labelsPolyVal)


		# Path to the Point Clouds folder to store archaeological site points to train a Local Outlier Factor later
		pointClouds = "LAS"
		# Index Point Clouds to a tree for faster search later on
		spindex = pyqtree.Index(bbox=(0, 0, 100, 100))
		for cloud in os.listdir(pointClouds):
			with laspy.open(pointClouds + '/' + cloud) as f:
				spindex.insert(pointClouds + '/' + cloud, (f.header.x_min, f.header.y_min, f.header.x_max, f.header.y_max))


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

		pathFile = open(datasetPath + "paths.txt", "a+")

		# Iterates over the images
		for image in images:
			pathFile.write(image + "\n")
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
			for bb in bbs:
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
					while len(x) > 0 and len(y) > 0:
						i = random.choice(x)
						j = random.choice(y)
						x.remove(i)
						y.remove(j)
						# Gets a region of interest expanding the random point into a region of interest with a certain resolution
						crop = (i-resolution//2, i+resolution//2, j-resolution//2, j+resolution//2)
						# Checks if that region of interest only covers 100% visible objects
						visibleObjects = checkVisibility(image, crop, processedObjects, bbs)
						if visibleObjects:
							break

					# If we obtain a list of visible objects within a region of interest, we save it
					if visibleObjects:
						# Uses the coordinates as the name of the image and text files
						coords = '('+ str(crop[0]) + '_' + str(crop[1]) + '_' + str(crop[2]) + '_' + str(crop[3]) + ')'

						
						# Training/Validation split
						if np.random.uniform(0,1) > validationSize/100:
							labelPath = labelsTrainPath
							polyPath = labelsPolyTrain
							imagePath = imagesTrainPath				
						else:
							labelPath = labelsValPath
							polyPath = labelsPolyVal
							imagePath = imagesValPath

						# Writes the image
						croppedImg = img.crop((crop[0], crop[2], crop[1], crop[3]))
						imgName = imagePath + image.split("/")[-1].split(".")[0] + coords + ".png"
						croppedImg.save(imgName)

						# Writes the label file in YOLO format
						txtFile = open(labelPath + image.split("/")[-1].split(".")[0] + coords + ".txt", "a+")
						for obj in visibleObjects:						
							# Maps the object in YOLO format
							centerX = (obj[0] + obj[1])/2.0
							centerX = mapping(centerX, crop[0], crop[1], 0, 1)
							centerY = (obj[2] + obj[3])/2.0
							centerY = mapping(centerY, crop[2], crop[3], 0, 1)
							w = (centerX - mapping(obj[0], crop[0], crop[1], 0, 1)) * 2.0
							h = (centerY - mapping(obj[2], crop[2], crop[3], 0, 1)) * 2.0


							# Writes/Appends the annotations to a text file that has the same name of the respective image
							txtFile.write(str(classes[bbs[obj][0]]) + " " + str(centerX) + " " + str(centerY) + " " +str(w)+ " "+ str(h) + "\n")

						txtFile.close()

						# Writes the label file for the polygons to use on the Data Augmentation later
						txtFile = open(polyPath + image.split("/")[-1].split(".")[0] + coords + ".txt", "a+")
						for obj in visibleObjects:						
							
							poly = bbs[obj][1]

							polygon = ""
							for p in poly:
								polygon += " "
								pX = round(mapping(p[0], crop[0], crop[1], 0, resolution))
								pY = round(mapping(p[1], crop[2], crop[3], 0, resolution))
								polygon += str(pX) + " " + str(pY)	

							# Writes/Appends the annotations to a text file that has the same name of the respective image
							txtFile.write(str(classes[bbs[obj][0]]) + polygon + "\n")

						txtFile.close()


						pointCloudCrop(spindex, LASPath, image, coords, crop, visibleObjects, imgName, pointClouds, xMinImg, xMaxImg, yMinImg, yMaxImg, width, height, resolution)


		img.close()
		pathFile.close()

	return ("", 204)

if __name__ == "__main__":
	app.run()