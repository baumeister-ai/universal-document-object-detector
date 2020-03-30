import cv2
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


def extract_target_image(img, min_max_hue=(0, 20)):
	
	# filter out anything not red
	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	grey_img = 255 * cv2.inRange(img_hsv,
	                             (min_max_hue[0], 25, 0),
	                             (min_max_hue[1], 255, 255))
	
	# run a Hough transform to find lines of rectangle
	threshold = 30
	min_line_length = int(0.1 * min(img.shape[0:2]))
	max_line_gap = int(0.0001 * min(img.shape[0:2]))
	lines = cv2.HoughLinesP(grey_img, 1, np.pi/180,
	                        threshold=threshold,
	                        minLineLength=min_line_length,
	                        maxLineGap=max_line_gap)
	
	# if we didn't find any lines, return None
	if type(lines) is not np.ndarray:
		return None, None
	
	# get the max and min x and y values of the bounding box
	min_x = int(np.min(np.hstack((lines[:, 0, 0], lines[:, 0, 2]))))
	max_x = int(np.max(np.hstack((lines[:, 0, 0], lines[:, 0, 2]))))
	min_y = int(np.min(np.hstack((lines[:, 0, 1], lines[:, 0, 3]))))
	max_y = int(np.max(np.hstack((lines[:, 0, 1], lines[:, 0, 3]))))
	
	# if we didn't find an area, return None
	if min_x == max_x or min_y == max_y:
		return None, None
	
	# crop the image
	img_crop = img[min_y : max_y, min_x : max_x, :]
	
	# get the size of the crop
	x_scale = (max_x - min_x) / img.shape[1]
	y_scale = (max_y - min_y) / img.shape[0]
	
	"""
	# set up the figure
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(img, origin="upper")
	ax[1].imshow(img_crop, origin="upper")
	plt.show()
	"""
	
	return img_crop, (x_scale, y_scale)



if __name__== "__main__":
	
	img = cv2.imread("data/target/geometry-083-target.png")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	print(extract_target_image(img))


