import cv2
import numpy as np

'''
@author: rohangupta

References:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective/20355545#20355545
'''


import numpy as np
import cv2
import timeit
import os
from matplotlib import pyplot as plt
import math

UBIT = "rgupta24"

import alignment
import warp
import blend
import seam_carving

def contraharmonic_mean(img, size, Q):
    num = np.power(img, Q + 1)
    denom = np.power(img, Q)
    kernel = np.full(size, 1.0)
    result = cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)
    return result

def computeMapping(leftImage, rightImage):
		leftGrey = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
		rightGrey = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)
		#orb = cv2.ORB_create()#xfeatures2d.SIFT_create()#
		if use_algorithm == "SIFT":
			orb = cv2.xfeatures2d.SIFT_create()
		elif use_algorithm == "SURF":
			orb = cv2.xfeatures2d.SURF_create()
		elif use_algorithm == "ORB":
			orb = cv2.ORB_create(nfeatures=1500)


		leftKeypoints, leftDescriptors = orb.detectAndCompute(leftGrey, None)
		rightKeypoints, rightDescriptors = orb.detectAndCompute(rightGrey, None)

		#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		#matches = bf.match(leftDescriptors, rightDescriptors)
		#matches = sorted(matches, key=lambda x: x.distance)

		# Brute-Force matching with SIFT descriptors
		bf = cv2.BFMatcher()

		# Matching the keypoints with k-nearest neighbor (with k=2)
		matches = bf.knnMatch(leftDescriptors, rightDescriptors, k=2)

		#nMatches = int(
		#	float(20) * len(matches) / 100
		#)

		#if nMatches < 4:
		#	return None

		goodMatch = []
		# Performing ratio test to find good matches
		for m, n in matches:
			if m.distance < 0.75 * n.distance:
				goodMatch.append(m)

		#matches = matches[:nMatches]
		motionModel = eTranslate
		nRANSAC = cv2.RANSAC
		RANSACThreshold = float(5.0)

		return alignment.alignPair(
			leftKeypoints, rightKeypoints, goodMatch, motionModel, nRANSAC,
			RANSACThreshold
		)



name_of_path1 = "data/1/img"

#SIFT SURF ORB
def pano(set_n, use_algorithm):
	name_of_set = "ep" + str(set_n)
	name_of_path0 = "data/" + name_of_set#+ "SIFT" + "_set_Lunch/"

	if use_algorithm == "SIFT":
		scale_percent = 30
	elif use_algorithm == "SURF":
		scale_percent = 20
	elif use_algorithm == "ORB":
		scale_percent = 10

	scale_percent = 20
	w_p = 200
	name_of_final = use_algorithm + "_" + name_of_set


	focalLength = 2500
	k1 = -0.0484573
	k2 = 0.0100024
	f = focalLength

	start_time = timeit.default_timer()

	name_of_path = name_of_path0
	iter = 1

	processedImages = None

	point_time_1 = timeit.default_timer()

	files = os.listdir(name_of_path0)
	images = [cv2.imread(os.path.join(name_of_path0, i)) for i in files]

	images_crop = []

	for i in images:
		# calculate the 50 percent of original dimensions
		width = int(i.shape[1] * scale_percent / 100)
		height = int(i.shape[0] * scale_percent / 100)

		# dsize
		dsize = (width, height)

		# resize image
		images_crop.append(cv2.resize(i, dsize))
		#images_crop.append(i)

	point_time_2 = timeit.default_timer()

	processedImages = [warp.warpSpherical(i, f, k1, k2) for i in images_crop]
	#processedImages = images_crop

	t = np.eye(3)
	ipv = []
	for i in range(0, len(processedImages) - 1):
		ipv.append(blend.ImageInfo('', processedImages[i], np.linalg.inv(t)))
		t = computeMapping(processedImages[i], processedImages[i+1]).dot(t)

	ipv.append(blend.ImageInfo('', processedImages[len(processedImages)-1], np.linalg.inv(t)))
	t = computeMapping(processedImages[len(processedImages)-1], processedImages[0]).dot(t)

	result = blend.blendImages(ipv, int(w_p), False)
	#cv2.imwrite("result_" + name_of_final + ".jpg", result)
	stop_time = timeit.default_timer()
	height = result.shape[0]
	width = result.shape[1]
	filtr_res = contraharmonic_mean(result, (height,width), 0.5)
#cv2.imwrite("result_" + name_of_final + "_f.jpg", result)

# Create our shapening kernel, it must equal to one eventually
	kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
# applying the sharpening kernel to the input image & displaying it.
	sharpened = cv2.filter2D(result, -1, kernel_sharpening)
#cv2.imwrite("result_" + name_of_final + "_s.jpg", sharpened)

	median = result#cv2.medianBlur(result, 3)
#median = cv2.medianBlur(median, 3)
#cv2.imwrite("result_" + name_of_final + "_m.jpg", median)

	sharpenedmedian = cv2.filter2D(median, -1, kernel_sharpening)
	cv2.imwrite("result_" + name_of_final + "_SM.jpg", sharpenedmedian)

#seam_carving.main_seam("result_" + name_of_final + "_SM.jpg", "result_" + name_of_final + "_SM_SEAM.jpg")

	resave = cv2.imread("result_" + name_of_final + "_SM.jpg")
	cv2.imwrite("it3/result_" + name_of_final + ".jpg", resave)

	print('Execution time ' + name_of_set + ' ' + use_algorithm + ': ', stop_time - start_time)


print("start")
eTranslate = 0
eHomography = 1
use_algorithm = "SURF"
iterator_set = 1
while iterator_set < 8:
	pano(iterator_set, use_algorithm)
	iterator_set = iterator_set + 1

use_algorithm = "SIFT"
iterator_set = 1
while iterator_set < 8:
	pano(iterator_set, use_algorithm)
	iterator_set = iterator_set + 1

use_algorithm = "ORB"
iterator_set = 1
while iterator_set < 8:
	pano(iterator_set, use_algorithm)
	iterator_set = iterator_set + 1

print("end")
cv2.destroyAllWindows()