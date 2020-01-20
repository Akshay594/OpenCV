#link https://drive.google.com/open?id=1OBWIFn0LzBHNHRYAmbW6NKeKOh2h-PwI

import cv2
from imutils import contours
import imutils
import numpy as np


ocr = cv2.imread('ocr.png')
ocr = cv2.cvtColor(ocr, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(ocr, 12, 255, cv2.THRESH_BINARY_INV)[1]

# cv2.imshow('thresh', thresh)
# cv2.imshow('ocr', ocr)

# cv2.waitKey(0)

digits = {}

ocrcnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ocrcnts = imutils.grab_contours(ocrcnts)
ocrcnts = contours.sort_contours(ocrcnts, method="left-to-right")[0]

for i, c in enumerate(ocrcnts):
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh[y:y+h, x:x+w]
	roi = cv2.resize(roi, (50, 80))

	digits[i] = roi

#################### Using Gradient for card reading ################

rectKernel = np.ones((3, 9), dtype="uint8")
sqKernel = np.ones((5, 5), dtype="uint8")

card = cv2.imread('card.png')
card = imutils.resize(card, width=300)
gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, sqKernel)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

# cv2.imshow('tophat', gradX)
# cv2.waitKey(0)

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
locs = []

for (i, c) in enumerate(cnts):
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
 
	if ar > 2.5 and ar < 4.0:
		if (w > 40 and w < 55) and (h > 10 and h < 20):
			locs.append((x, y, w, h))

# for (x, y, w, h) in locs:
# 	cv2.rectangle(card, (x, y), (x+w, y+h), (0, 0, 255), 2)

# cv2.imshow('image', card)
# cv2.waitKey(0)

for (x, y, w, h) in locs:
	cv2.rectangle(card, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('image', card)
cv2.waitKey(0)
