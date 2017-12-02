import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math 
from moviepy.editer import VideoFileClip
import os
import imutils


#annotate functions
def annotate_image(input_file, output_file):
	annotated_image_output= annotate_image_array2(mpimg.imread(input_file))
	plt.imsave(output_file, annotated_image_output)
	
def annotate_video(input_file, output_file)


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
	 