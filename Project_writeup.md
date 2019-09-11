**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_image.jpg "Undistorted"
[image2]: ./output_images/undistorted_roadimage.jpg "Road Transformed"
[image3]: ./output_images/binary_combined.jpg "Binary Example"
[image4]: ./output_images/straight_lines1_BinImgWLines.jpg "Warp Example"
[image5]: ./output_images/straight_lines1_BinImgBirdsEye.jpg "Warp Example birdeye"
[image6]: ./output_images/Window_Search.jpg "Window search"
[image7]: ./output_images/example_output.jpg "Output"
[video1]: ./output_images/project_video_V10.mp4 "Video"
[video2]: ./output_images/challenge_video_V10.mp4 "Video"
[video3]: ./output_images/challenge_video_search_V9.mp4 "Video"



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Code overview
The code is built into a main file: `Advanced-Lane-Finding.ipynb`

A Line class: `Line.py`

and helper functions: `MyTools.py`

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./Advanced-Lane-Finding.ipynb" in chapter 1.1.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
This part is implemented in `GetImageAndObjectPoints(imgPaths, doPlots=False)`

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function, implemented in `cal_undistort(img, objpoints, imgpoints)` .  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
Here the calibration and distortion coefficients found before is used to transform a road image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HLS color, grayscale and Canny thresholds to generate a binary image (thresholding steps in `CannyDetect(img, low_threshold=80, high_threshold=130)` and `ColorThreshold(img)` in  `MyTools.py`). The grayscale image was used to threshold the canny image, to get dark lines in the lane filtered away, which is a problem using canny detection. This was done in combination with an erosion function (`cv2.erode()`), to widthen the dark lines before used for thresholding of the canny image.   

Here's an example of my combined binary image output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In *Chapter 1.3* of the notebook, I apply a perspective transform to the binary image from before. The perspective transform was found using a straigt lane picture and selecting pixel points from the lane-lines into *src*-points. Here it was chosen not to go too far into the horizon, as it had proved difficult to get good lane estimates there. Then selecting a square of image points as the *dst*-points. The *dst*-points was selected so it included the entire lane together with some offset in both sides of the image.

```python
# src is Calculated by function
dst = np.float32([(400, dimY),(400, 0),(1000, 0),(1000, dimY)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 720      | 400, 720      | 
| 540, 490      | 400, 0        |
| 751, 490      | 1000, 0       |
| 1120, 720     | 1000, 720     |

The function `M = cv2.getPerspectiveTransform(src, dst)` was used to calculate the transformation matrix and the function `cv2.warpPerspective(img_binary_col, M, (dimX, dimY))` was used to warp the image perspective.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane lines is calculated in *Chapter 1.4*. The code is implemented in the `Lane.py` class. This class is initiated as a left or right lane. Then lane-pixels are found using either window search from the bottom of the image and upwards, initiated from a histogram, or a polynomial search around a already found polynomial. A new search is iniatiated with a window search, but if a good polynomial exists, all search is done by polynomial search until a number of missing lane detection in consecutive frames is seen, before a new window search is triggered. A function is made to identify if 
the found polynomial is accepted and calculate best fit polynomial and lane x-offset, to reduce noise in detection. 

Window search is done by: `find_lane_pixels_from_windows(self, img_binary)`

Polynomial search is done by: `find_lane_pixels_around_poly(self, img_binary)`

Polynomial is calculated by: `fit_polynomial_from_idx(self)`

Check found polynomial and calculate new best fit parameters: `CheckPolynomialAndSetBestFit(self, polyFit)`  

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `MyTools.py`. The function `CalculateLaneOffset(leftLine, RightLine)` calculates the lane offset, from the left and right lane class objects, using the best x positions.

The `Line`-class function `CalculateLineCurvature(self, polyFit)` calculates a radius based on a polynomial fit and using X and y-axis pixel to meters scaling. 



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in *Chapter 1.4* using the `Line`-objects together with the function `MyTools.BirdsEyeLaneFromLeftAndRightPolynomials()`. Then I warped the image back to driver view and plotted it using and `MyTools.MergeAndPlotBGRImages()`. Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline for running the Lane-finding algorithm is found in *Chapter 1.5*. It can be run in debug mode by seeting the *debug* flag to 1, which enables display of the pixel search/tracking view in the output video.  


Here's a [link to my video result][video1]

Link to challenge video result [link to challenge video result][video2]

Link to challenge video searching [link to challenge video search][video3]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had many difficulties on the 'challange_video.mp4' to enable a robustness towards the dark lines on the left and right lane when using Canny edge detection. This was solved in the end like 
explanied earlier. Also it was difficult to get a robust polynomial fit, when only few pixels where available. This was solved by avereging over 30 frames, and by having many quality checks
enabled for a new polynomial to be accepted. My pipeline will probably fail when facing severe curvature on the road, as well as bad lighting conditions, or lane markings.

An improvement could be to do a brightness correction of the image before thresholding etc, maybe using histogram equalization. Also it might improve the algorithm to do a perspective transform before
doing Canny and color thresholding. This would make distant lane lines much wider, and thus with more pixels in them.  
Also adaptive thresholding techniques could help the canny and color thresholding to improve.   
