import requests
from PIL import Image
import base64
from io import BytesIO
import csv
import json
Image.MAX_IMAGE_PIXELS = None

def polys(row):
	raw = row.split(",")
	raw[0] = raw[0].strip(raw[0][:raw[0].find('(')] + '(((')
	raw[len(raw)-1] = raw[len(raw)-1].strip(')))')
	xPoints = []
	yPoints = []
	for i in range(0, len(raw)):
	    xPoints.append(float(raw[i].split(" ")[0]))
	    yPoints.append(float(raw[i].split(" ")[1]))

	return (xPoints, yPoints)

#URL = 'http://192.168.1.65:8082'
URL = 'http://127.0.0.1:5000'
annotations = ['../ODYSSEY/Images/LRM/PNPG.csv', '../ODYSSEY/Images/LRM/Arcos.csv']
#DTM = 'Arcos.tif'
path = ['../ODYSSEY/Images/LRM/PNPG.tif', '../ODYSSEY/Images/LRM/Arcos.tif']#, '../ODYSSEY/Images/LRM/Viana.tif', '../ODYSSEY/Images/LRM/Coura.tif']
# bbox_converted = [xmin, ymin, xmax, ymax]
#coords = (-61809,236336,-2247, 245758)
coords = (0,258000,6000, 266000)

#image1 =  base64.b64encode(open(LRM,'rb').read()).decode('ascii')
#image2 =  base64.b64encode(open(DTM,'rb').read()).decode('ascii')
#images = {}
#images['LRM'] = image1
#images['DTM'] = image2

data = {}
classes = ['mamoa', 'outro']

x = 0
for annotation in annotations:
	with open(annotation) as csvfile:
		reader = csv.DictReader(csvfile)
		# This considers that polygons are under the column name "WKT" and labels are under the column name "Id"
		polygons = "MULTIPOLYGON ((("
		count = 0
		for row in reader:
			xPoints, yPoints = polys(row['WKT'])

			if count != 0:
				polygons += ', (('

			for i in range(len(xPoints)):
				if i != len(xPoints)-1:
					polygons += str(xPoints[i]) + " " + str(yPoints[i]) + ','
				else:
					polygons += str(xPoints[i]) + " " + str(yPoints[i]) + '))'

			count += 1
		polygons += ')'

	data[classes[x]] = polygons
	x+=1
	print(count)

purpose = 'training' # training/inference



types = ['LRM']
multipleFiles = [('annotations', json.dumps(data)), ('geotiff', json.dumps(path)), ('coords', json.dumps(coords)), ('purpose', purpose)]

received = requests.post(URL, data=multipleFiles)

print(received.text)