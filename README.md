# Vision
Rajasthan Hackathon 3.0
[![Vision](https://github.com/kunalrajora/vision/blob/master/resource/Vis_logo.jpg?raw=true)](https://github.com/kunalrajora/vision/blob/master/resource/My%20Movie%203.mp4?raw=true)

This is a project by team Vision. The goal is to detect lane lines on video using computer vision methods and show path on them in real time(OpenCV library).

Directories:
- All commented code can be found at vision.py python file.
- Images folder contains augmentation on images.
- Video folder contains augmentation on video.

# Overview

Whole pipeline consist of following steps:

1. Removing camera distortion. We need camera calibration for that.
2. Filtering lane lines with morphology operations
3. Determining lane lines.
7. Trapezoidal view
8. Drawing lane lines and reprojecting from trapezoidal view back to front camera view.

# Camera calibration

Camera calibration is very importaint step. For successful camera calibration it's good to use bigger count of calibration images with wider variety of callibration pattern positioning. I did calibration with CV2 library to implement gaussian blur, Canny edge detection, hough lines and much more detailing. More about camera calibration may be found [here](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html).

# Morphology filter

To be able to threshold lane lines more correclty I decided to do prefiltering with "opening" morphology operation. Opening with horizontal kernel is used for separating all lane lines width or less width elements from the image. Also opening may be used for removing lane lines from picture if it is needed. More about morphology operations may be found [here](http://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html). As input for morphology filtering I use linear combination of grayscale and S channel from HLS color space representation.

```python
    """returning the annotated image as a Numpy array """
	# Only keep white and yellow pixels in the image, all other pixels become black
	image = filter_colors(image_in)
	#convert image
	gray = black(image_in)
	
    # Gaussian smoothing
    kernel_size = 3

	# Apply Gaussian smoothing
	blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
	
```

![morphology filtering example] (https://github.com/kunalrajora/vision/blob/master/resource/Gray&Gauss.png?raw=true)

# Identifying lane lines

After applying morphology filtering I find gradients on the image in different ways. We used hough lines and canny functions to idenitify lanes. To find lane lines vision use combination of mentioned thresholds:

```python
    
    # Canny Edge Detector
    low_threshold = 10
    high_threshold = 200
    
    Apply Canny Edge Detector
	
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    # Hough Transform
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15	 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10
    max_line_gap = 20
	
    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
	    draw_lines(line_img, lines)
	    return line_img
```
    

Here is an example of each their combination.

 (https://github.com/kunalrajora/vision/blob/master/resource/Canny.png?raw=true)

# Trapezoidal view

Considering road is near flat surface we can reproject it as plane from front camera view into trapezoidal view using perspective projection. I use OpenCV cv2.polygon for that. Needed projection matrix can be found with 4 points which is known to be on the road or region of interest.This provides only the road region everything else is blacked out. Here is example of input and reprojected image.

```python
# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 1.2  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.07  # ditto for top edge of trapezoid
trap_height = 0.4  # height of the trapezoid expressed as percentage of image height

# Helper functions

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
```

![trapezoidal view perspective reprojection] (https://github.com/kunalrajora/vision/blob/master/resource/r_o_i.png?raw=true)

# Drawing direction arrows on lane lines

After reprojecting to trapezoidal view lines may be fitted with 2 degree polynomial or quadratic equation y=A*x**2 + B*x + C. 
I do it in three steps.

1. Seperating left or right line using there slope.
2. Calculating distance between these two lines.
3. Drawing arrows at the center of the road for better understanding. 
![Arrow drawing projection] (https://github.com/kunalrajora/vision/blob/master/resource/draw.png?raw=true)

1.Spliting lines
```
python

def draw_lines(img, lines, color=[0, 144, 255], thickness=5):
	# In case of error, don't draw the line(s)
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True
	
	# Find slopes of all lines
	# But only care about lines where abs(slope) > slope_threshold
	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
		
		# Calculate slope
		if x2 - x1 == 0.:  # corner case, avoiding division by 0
			slope = 999.  # practically infinite slope
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		# Filter lines based on slope
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)
		
	lines = new_lines
	
	# Split lines into right_lines and left_lines
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)
			
	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []
```
2. Calculating coordinates and lines distance
```python
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
	
	# Find 2 end points for right and left lines, used for drawing the line as by formula
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
```
3. Drawing lines
```python
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
```


# Example processed video frame image

Here is an example of resulted video frame image with drawn lane lines, curvature and offset information
(https://github.com/kunalrajora/vision/blob/master/resource/Vis_app.png?raw=true)

# Conclusion

Reprojecting into trapezoidal view and fitting lane lines by polynomial is promising method which may find not only straight lane lines but also curved ones. But it is not robust enough to deal with complex environment with tree shadows, road defects, brightness/contrast issues. It will be effective on environments where lane lines are bright, contrast, not occluded or overlapped.

In many situations human drivers consider lane lines not as direct driving rule but as hint of which position of the road it is better to take. Also human drivers may predict lanes on roads without lane lines marks. I think this OpenCV approach may be used for generating dataset for more complex model. Such as neural network which may be able to predict lane lines marks. Using well detected parts of video we may erase lane line marks using morphology operations and use such images without lane lines marks (and also another augmentation technics) along with detected curves as dataset for training. I think such method may be used to get lane lines on roads that even have no lane lines.We can also use this road signs detection, distance from vehicle detection and street lights detection.


