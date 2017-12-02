import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math 
from moviepy.editer import VideoFileClip
import os
import imutils

#Gaussian smoothing factor
kernel_size = 3

## Canny Edge Detector
low_threshold = 50
high_threshold = 150

def black(image):
	resized = imutils.resize(image, width=300)
	ratio = image.shape[0] / float(resized.shape[0])
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(resized,100,200)
	cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# loop over the contours
	for c in cnts:
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(c)
		if M["m00"]==0:
			M["m00"]=1
		cX = int((M["m10"] / M["m00"]) * ratio)
		cY = int((M["m01"] //M["m00"]) * ratio)
	

    	# multiply the contour (x, y)-coordinates by the resize ratio,
    	# then draw the contours and the name of the shape on the image
		c = c.astype("float")
		c *= ratio
		c = c.astype("int")
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		return image


def annotate_image_array2(image_in):

	#image = filter_colors(image_in)
	gray = black(image_in)
	


#annotate functions
def annotate_image(input_file, output_file):
	annotated_image_output= annotate_image_array2(mpimg.imread(input_file))
	plt.imsave(output_file, annotated_image_output)
	
def annotate_video(input_file, output_file):
	video= VideoFileClip(input_file)
	annotated_video_output= video.fl_image(annotate_image_array)
	annotated_video_output.write_videofile(ouptut_file, audio=False)

#main
if __name__  ==  '__main__':
	from optparse import OpitonParser
	
	#command lines 
	parser = OptionParser()
	parser.add_option("-i", "--input_file", dest= "input_file", help="Input video/image file")
	parser.add_option("-o", "--output_file", dest= "output_file", help="Output (destination) video/image file" )
	parser.add_option("-I", "--image_only", action= "store_true", dest= "image_only ", default =False, help="Annotate image (defaults to annotating video)")
	
	#get parsed option
	option, args = parser.parse_args()
	input_file= options.input_file
	output_file= options.output_file
	image_only= option.image_only
	
	if iname_only :
		annotate_image(input_file, output_file)
	else:
		annotate_video(input_file, output_file)
	 