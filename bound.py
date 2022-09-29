# IMPROVEMENTS
# My script looks for the max of the blue objects to draw a bounding box
# so that small areas of blue are not highlighted

# REFERENCES
# - Bounding box: https://stackoverflow.com/questions/23398926/drawing-bounding-box-around-given-size-area-contour
# - Masking color: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
	_, frame = cap.read()

    # Convert BGR to HSV
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
	lower_blue = np.array([110,50,50])
	upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
	mask = cv.inRange(hsv, lower_blue, upper_blue)

	# countour blue images
	contours,hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

	max = 0

	showRect = None
	x = 0
	y = 0
	w = 0
	h = 0


	for c in contours:
		if cv.contourArea(c) > max:
			max = cv.contourArea(c)
			rect = cv.boundingRect(c)
			x,y,w,h = cv.boundingRect(c)
			showRect= True

	if showRect:
		cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # Bitwise-AND mask and original image
	res = cv.bitwise_and(frame,frame, mask= mask)
	cv.imshow('frame',frame)

	k = cv.waitKey(5) & 0xFF
	if k == 27:
		break
	
cv.destroyAllWindows()