import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import math
import os
import imutils

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Region-of-interest vertices
trap_bottom_width = 1.5 
trap_top_width = 1 
trap_height = 0.4 

# Hough Transform
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15	
min_line_length = 10
max_line_gap = 20	


def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
def canny(img, low_threshold, high_threshold):
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_lines(img, lines, color=[0, 144, 255], thickness=5):
	
	# In case of error, don't draw the line(s)
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True
	
	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
		# Calculate slope
		if x2 - x1 == 0.: 
			slope = 999. 
		else:
			slope = (y2 - y1) / (x2 - x1)
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)
	#devide left lines and right lines	
	lines = new_lines
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2 
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)
			
	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []
	
	for line in right_lines:
		x1, y1, x2, y2 = line[0]
		
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		
		right_lines_y.append(y1)
		right_lines_y.append(y2)
		
	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
	else:
		right_m, right_b = 1, 1
		draw_right = False
		
	# Left lane lines
	left_lines_x = []
	left_lines_y = []
	
	for line in left_lines:
		x1, y1, x2, y2 = line[0]
		
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		
		left_lines_y.append(y1)
		left_lines_y.append(y2)
		
	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
	else:
		left_m, left_b = 1, 1
		draw_left = False
	
	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - trap_height)
	
	right_x1 = (y1 - right_b) / right_m
	right_x2 = (y2 - right_b) / right_m
	
	left_x1 = (y1 - left_b) / left_m
	left_x2 = (y2 - left_b) / left_m
	
	# Convert calculated end points from float to int
	y1 = int(y1)
	y2 = int(y2)
	right_x1 = int(right_x1)
	right_x2 = int(right_x2)
	left_x1 = int(left_x1)
	left_x2 = int(left_x2)
	
	# Draw the right and left lines on image
	if draw_right and draw_left:
		m_lower=(right_x1 +left_x1)//2
		r=(y2-y1)//6
		z=y1+r
		for i in range(5):
			if i<2:
				z+=r
				y1+=r
				left_x1+= int((m_lower - left_x1)/3) 
				right_x1-= int((right_x1 - m_lower)/3)
				continue
			cv2.line(img, (left_x1, y1), (m_lower, z), color, thickness)
			cv2.line(img, (right_x1, y1), (m_lower, z), color, thickness)
			z+=r+8
			y1+=r
			left_x1+= int((m_lower - left_x1)/3) 
			right_x1-= int((right_x1 - m_lower)/3) 
			#faded part
			
	    
	
	if draw_right and not draw_left:
		r=(y2-y1)//6
		z=y1+r
		left_x1=right_x2 -(right_x1-right_x2)
		m_lower=(right_x1 +left_x1)//2
		for i in  range(5):
			if i<2:
				z+=r
				y1+=r
				left_x1+= int((m_lower - left_x1)/3) 
				right_x1-= int((right_x1 - m_lower)/3)
				continue
			cv2.line(img, (right_x1, y1), (right_x2, z), color, thickness)
			cv2.line(img, (left_x1, y1), (right_x2, z), color, thickness)
			z+=r
			y1+=r
			left_x1+= int((m_lower - left_x1)/3) 
			right_x1-= int((right_x1 - m_lower)/3) 
			
	
	if not draw_right and draw_left:
		r=(y2-y1)//6
		z=y1+r
		right_x1=left_x2 +(left_x2-left_x1)
		m_lower=(right_x1 +left_x1)//2
		for i in  range(5):
			if i<2:
				z+=r
				y1+=r
				left_x1+= int((m_lower - left_x1)/3) 
				right_x1-= int((right_x1 - m_lower)/3)
				continue
			cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
			cv2.line(img, (right_x1, y1), (left_x2, y2), color, thickness)
			z+=r
			y1+=r
			left_x1+= int((m_lower - left_x1)/3) 
			right_x1-= int((right_x1 - m_lower)/3) 
			
		
	
	
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
	draw_lines(line_img, lines)
	return line_img

def weighted_img(img, initial_img, al=0.8, bt=1., ga=0.):
	return cv2.addWeighted(initial_img, al, img, bt, ga)

def filter_colors(image):
	white_threshold = 200 
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv2.inRange(image, lower_white, upper_white)
	white_image = cv2.bitwise_and(image, image, mask=white_mask)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([90,100,100])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

	return image2

def annotate_image_array(image_in):
	image = filter_colors(image_in)
	gray = grayscale(image)
	blur_gray = gaussian_blur(gray, kernel_size)
	edges = canny(blur_gray, low_threshold, high_threshold)
	imshape = image.shape
	vertices = np.array([[\
		((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
		((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
		, dtype=np.int32)
	masked_edges = region_of_interest(edges, vertices)
	line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
	initial_image = image_in.astype('uint8')
	annotated_image = weighted_img(line_image, initial_image)
	
	return annotated_image

def black(image):
	resized = imutils.resize(image, width=300)
	ratio = image.shape[0] / float(resized.shape[0])
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(resized,100,200)
	cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# loop over the contours
	for c in cnts:
		M = cv2.moments(c)
		if M["m00"]==0:
			M["m00"]=1
		cX = int((M["m10"] / M["m00"]) * ratio)
		cY = int((M["m01"] //M["m00"]) * ratio)
	
		c = c.astype("float")
		c *= ratio
		c = c.astype("int")
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		return image

def annotate_image_array2(image_in):
	image = filter_colors(image_in)
	gray = black(image_in)
	blur_gray = gaussian_blur(gray, kernel_size)
	edges = canny(blur_gray, low_threshold, high_threshold)

	imshape = image.shape
	vertices = np.array([[\
		((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
		((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
		, dtype=np.int32)
	masked_edges = region_of_interest(edges, vertices)

	line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
	
	initial_image = image_in.astype('uint8')
	annotated_image = weighted_img(line_image, initial_image)
	
	return annotated_image

def annotate_image(input_file, output_file):
	annotated_image = annotate_image_array2(mpimg.imread(input_file))
	plt.imsave(output_file, annotated_image)

def annotate_video(input_file, output_file):
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image_array)
	annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
	from optparse import OptionParser

	# Configure command line options
	parser = OptionParser()
	parser.add_option("-i", "--input_file", dest="input_file",
					help="Input video/image file")
	parser.add_option("-o", "--output_file", dest="output_file",
					help="Output (destination) video/image file")
	parser.add_option("-I", "--image_only",
					action="store_true", dest="image_only", default=False,
					help="Annotate image (defaults to annotating video)")

	# Get and parse command line options
	options, args = parser.parse_args()

	input_file = options.input_file
	output_file = options.output_file
	image_only = options.image_only

	if image_only:
		annotate_image(input_file, output_file)
	else:
		annotate_video(input_file, output_file)