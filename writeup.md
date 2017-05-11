# **Advanced Lane Finding Project**

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Link to my [project code](https://github.com/cpierceworld/CarND-Advanced-Lane-Lines/blob/wip/Advanced_Lane_Lines.ipynb)


[//]: # (Image References)

[image1]: ./output_images/camera_calibration.png "Calibration"
[image2]: ./output_images/perspective_transform.png "Perspective"
[image3]: ./output_images/01_undistorted_img.png "Undistorted"
[image4]: ./output_images/02_binary_threshold.png "Binary Trheshold"
[image5]: ./output_images/03_warped_binary_threshold.png "Warped Threshold"
[image6]: ./output_images/03_histogram.png "Histogram"
[image7]: ./output_images/03_windows.png "Search Using Windows"
[image8]: ./output_images/03_around_lines.png "Search Around Lines"
[image9]: ./output_images/03_warped_img.png "Image Warped"
[image10]: ./output_images/04_detected_lane_warped.png "Filled Lane Warped"
[image11]: ./output_images/05_detected_lane_unwarp.png "Filled Lane Unwarped"
[video1]: ./project_video_out.mp4 "Video"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

A copy of this writeup can be found [here](https://github.com/cpierceworld/CarND-Advanced-Lane-Lines/blob/wip/writeup.md).  The code this writeup references can be found [here](https://github.com/cpierceworld/CarND-Advanced-Lane-Lines/blob/wip/Advanced_Lane_Lines.ipynb)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./Advanced_Lane_Lines.ipynb".

I  created a class called `Transform` to hold the both the "camera calibration" maps and the "perspective transform" matrix (and inverse).  The method that does the camera calibration is called `calibrate_with_chessboards()`.

This method takes in a "glob" pattern to load up chessboard images.  It also takes an nx and ny value which are the number of horizontal and vertical chessboard corders respectivly (all chessboard images matched by the glob pattern should have the same nx, ny values).

I prepare "object points" corresponding to (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

(If the parameter `save_corners` was passed as True then, for every chessboard image where corners were detected, the image file name, and corers are stored in the `chessboard_imgs_and_corners` property.   This is to allow client code to be able to call `cv.drawChessboardCorners` with the found corners if it wants to)

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  These in turn are used to generate the "undistortion maps" using the `cv2.initUndistortRectifyMap()` fuction.  These maps are reused over and over to undistort each image in the video (in the `remove_distortion()`)

The code that actually calls the calibration is in the third cell of the notebook.  Here is an example of an undistorted chessboard:
![alt text][image1]


### Pipeline (single images)

The pipeline is defined in the `process_image()` method in the 13th cell of the notebook.

#### 1. Provide an example of a distortion-corrected image.

I used the `Transform.remove_distortion()` (see cell 1 for defention of `Transform` class) to remove camera distortion from each image of the video.  Here is an example frame of the video undistorted:

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Thresholding is done in the 5th cell of the note book in the `to_binary_mask()` method, which in turn uses the utilities in the 4th cell: `sobel_thresh()` and `color_chanel_thresh()`.  Here's an example of my output for this step.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform is handled by the `Transform` class defined in code cell 2.

The `Transform` has to be initialized with source and destination points via the `init_perspective_transform()` method.  The actual call to initialize the `Transform` is in code cell 3, and I used the following points:

| Source       | Destination   | 
|:------------:|:-------------:| 
| 140, 720     | 290, 720      | 
| 550, 435     | 290, 0        |
| 680, 435     | 990, 0        |
| 1130, 720    | 990, 720      |

Once initialzed, the `Transform.warp_perspective()` is used to peform the perspective transform, and `Transform.unwarp_perspective()` is used to tranform back to the original perspectve.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use helper classes to keep track of lines/lanes accross multple frames of video:
1. `Line` class (code cell 7).  This holds information about a line (quadratic polynomial coeeficients for the "fit") for the current frame's line, and previous 5 detected lines.  It has methods to validate the detected line and to keep a "best fit" which is an average of previously detected lines.
2. `Lane` class (code cell 8).  This holds information about a lane, which consists of a left and right `Line`.  This class has methods to validate that the detected lines make a valid lane, and keep track of the number of consecutive non-detected lanes.

The lane-line finding is in the `find_lane()` method in code cell 11.  This method delegates to either the `find_lane_pixels_initial()` or the `find_lane_pixels_using_previous()` depending on whether or not a lane has been detected in previous frames of video.

##### 4.1 Initial search for Lane-Lines

The code to do the initial search for lane-lines is in code cell 9 in the method `find_lane_pixels_initial()`.

This code uses a sliding window technique to find lines, searching from the bottom of the window up in 9 horizontal layers.  To know where to start, it takes the perspective warped binary image and takes a histogram of the bottom 3rd  of the image.  The left/right peaks are used as the starting x postion of the sliding search windows.   Below is an example of a warped, binary image and its histogram:

![alt text][image5]
![alt text][image6]

Once we have the starting x positions, we draw boxes around those positions and any pixels turned ON within those boxes are used to fit a line to.  We take the average x position of the ON pixels, and that becomes the starting x postion for the next layer up to which the sliding window moves. Below is an example of the sliding windws for our example image:

![alt text][image7]


##### 4.2 Search for Lane-Lines given previously detected lines

If we already have a "best fit" set of lane lines from previous frames of video, then we use the `find_lane_pixels_using_previous()` method found in code cell 10.

This method takes the previously detected lines, and plots them from the bottom to the top of the image.  It then takes a window around each line and any ON pixels within the windows are used to fit the new lines.  Below is an example of the previous found lines, the windows around those lines, and the found pixes colored blue and red:

![alt text][image8]

##### 4.3 Fiting the lines and validation

Back in the `find_lane()` method in code cell 11; regardless of whether the detected pixels are found via `find_lane_pixels_initial()` or `find_lane_pixels_using_previous()`, they are passed into the `Lane.set_new_lane_pixels()` method (which in turn passes the left and right pixes to the left and right `Line.set_new_line_piexels()` method)

The `Line.set_new_line_piexels()` will fit a quadratic to the points represented by the pixels.  It then does some validation on the line such as "does the curvature make sense?" and "is the left/right lane on the left/right side of the frame?".  If it fails validation, then the line is "not detected".

The `Lane.set_new_lane_pixels()` checks to see if the left/right line were "detected" (passed validation).  If they were then the `set_new_lane_pixels()` does its own validation such as "does the width of the lane make sense?" and "are the lane lines parallel?".  If the lane fails validation, then the lane is "not detected".

The `find_lanes()` method will reset the lane after 5 failed lane detections to force it to look for lines from scratch again.

Below is an example of the found lane filled in:

![alt text][image10]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

There is a utility method for calculating radius of curvature in code cell 6 - `calc_radius_of_curvature()`.

Each `Line` calculates and keeps track of its own radius of curvature at the time it is initialized via `Line.set_new_line_pixels()`.

Each `Lane` has a method to calculate its radius of curvature as the average of its two line's radius curvature (in `Lane.get_radius_of_curvature()`).

The `Lane` also has a method to calculate vechicle offset (in `Lane.get_offset_from_center()`), which findes the mid point between its left and right line, and sees how off from the center of the frame that midpoint is.

The pipeline method `process_image()` method in code cell 13 get the radius of curvature and the offset and embeds them in the image frame.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The the last step of the pipeline method `process_image()` method in code cell 13 is to unwarp the highlighted lane (via `Transform.unwarp_perspective()`) and superimpose it on the original image:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some issues I encountered working on this project:
1. Tuning the binarizing thresholds (gradient and color thresholds) was time consuming and subjective.  I found that the values that work well in one video don't necessariy work well in another.
2. Shadows, tar lines, HOV markings, etc. all throw off line detection since they also show up with strong gradients.  All of these make the code, as it is, fail on the challenge videos.  I tried adding histogram normalization to the images, but that seemed to make things worse.
3. On the challenge videos, I found that it rather quickly got stuck on lanes that didn't even resemble lanes, like 1 pixel wide lanes on the shoulder, criss crossing left/right lane lines, wierd left/right facing parabolas, etc.   I put in some sanity checks to throw out obviously bad lanes, but could have more.

Some possible improvements:
1. For the initial x positions from the histogram, try to use peaks that are closest to being a 'lane distance" apart. 
2. Switch to using the "convolution" method of finding window centers.  I found some scenarios where using the pixel x-position average causes the window to shift out of place due to outlier pixels affecting the average.   The convolution solution should be less susceptible to that problem.
3. Maybe a convolution kernel yielding a high output for "dark-light-dark" regions, where the "light" width is the expected width of a lane line marking (e.g. [-1, -1, -1, -1, 1, 1, 1, ... 1, -1, -1, -1, -1]).
4. The lane width logic is not really correct (as it subtracts every x position of a left line from the corresponding right line).   For tight turns where the one line goes off the image, will end up with widths much large than a normal lane width.  Better solution would be a "check parallel".  One possibility would be to use the inverse slope between successive pixels in one line to find corresponding pixels in another line.


