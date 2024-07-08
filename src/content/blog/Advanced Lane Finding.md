---
title: Advanced Lane Finding 
description: Advanced Lane Finding 
pubDate: Nov 20 2018
heroImage: /blog-placeholder-3.jpg
---

# Project 2

# Advanced Lane Finding 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Advanced Lane Detection Project which includes advanced image processing to detect lanes irrespective of the road texture, brightness, contrast, curves etc. Used Image warping and sliding window approach to find  lane lines.  The curvature of the lane and vehicle position with respect to center is detected.

# The Steps involved are:

##### 1. Computing the camera calibration matrix and distortion coefficients for 9*6 chessboard images. 
##### 2. Apply a distortion correction to raw images.
##### 3. Use color transforms, gradients, etc., to create a thresholded binary image.
##### 4. Apply a perspective transform to rectify binary image ("birds-eye view") to get a warped image.
##### 5. Detect lane pixels and fit to find the lane boundary.
##### 6. Determine the real curvature of the lane and vehicle position with respect to center.
##### 7. Warp the detected lane boundaries back onto the original image.
##### 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Rubric Points

## Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

 Open CV functions like findChessboardCorners(), drawChessboardCorners() and calibrateCamera() is used to find corners. Corners are Img_points to plot on the image. Img_points and Obj_points are used to calculate mtx and dist coefficients to undistort  the distorted images.
##### Calibrated Image with points drawn: 

![output](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/output_result_0.jpg?raw=true)

##### Distortion Corrected Calibrated Image:

![input](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/distortion_not_corrected.jpg?raw=true)
![input](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/distortion_corrected.jpg?raw=true)





## Pipeline

#### 1. Provide an example of a distortion-corrected image.

#### Some examples of Distortion Corrected Images are given below.

Undistort test_images folder, using cv2.undistort() output_images are saved to output_images folder

#### These images are Distortion Corrected :

![output_1](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/output/test1.jpg?raw=true)
![output_2](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/output/test2.jpg?raw=true)
![output_3](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/output/test3.jpg?raw=true)
![output_4](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/output/test4.jpg?raw=true)
![output_5](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/output/test5.jpg?raw=true)
![output_6](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/output/test6.jpg?raw=true)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

Function undist_color_thresh_binary() will detect gradient threshold and  color thresholding from  sobel operation and HLS color space, will output a common of both  as binary image.



### Input Image
![output_1](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/test1.jpg?raw=true)

### Output Image
![output_2](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/color_gradient_thresholded.jpg?raw=true)



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Function perspective_transf() will do the following Perspective Transform is the Bird's eye view for Lane images. It will change the perspective to top view. 

Using src, dst point values the Perspective Transform is done. 	


![output_1](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/test_image_transformed_image.jpg?raw=true)



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After getting Perspective Transform binary warped image, function fit_polynomial_with_plot() using sliding window it will mark lane lines.

### Finding the lines - Sliding Window and fitting a polynomial

![output_1](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/find_lane_pixels.jpg?raw=true)


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Function curvature_position() will calculate the radius of curvature and the position of the vehicle with respect to center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Function detect_lane_orginal() will transform back to the orginal image and plot the road. Along with radius of curvature and center position. 

##### Lane Area Drawn without Information:

![warp_detected_lane](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/warp_detected_lane.jpg?raw=true)

##### Lane Area Drawn with Radius of Curvature and Central Offset:

![lane_boundary_curvature_position](https://github.com/srikar8/srikar8.github.io/blob/master/images/Udacity/Advanced-Lane-Finding-P2/lane_boundary_curvature_position.jpg?raw=true)


# Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

##### Here is the Link to the Project Video 

<iframe width="560" height="315" src="https://www.youtube.com/embed/6sYKu5J3KLI"></iframe>




# Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The Pipeline I followed will draw the lane area perfectly over the road with radius of curvature and center positon. Video was executed on project, challenge, hard_challenge videos.

This pipeline will fails in situations like when road curve is alot. In video hard_challenge you can see the road area is detected wrongly.
