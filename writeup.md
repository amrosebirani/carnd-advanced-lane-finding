## Advanced Lane Detection

---

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

[image1]: ./test_images/test1.jpg "Original"
[image2]: ./output_images/test_undistorted_1.jpg "Undistorted"
[image3]: ./output_images/test_thresholded_1.jpg "Binary Thresholded Image"
[image4]: ./output_images/test_masked_1.jpg "Masked Image"
[image5]: ./output_images/test_warped_1.jpg "Warped"
[image6]: ./output_images/test_sliding_window_1.jpg "Sliding Window"
[image7]: ./output_images/test_previous_frame_1.jpg "Previous Frame"
[image8]: ./output_images/test_lane_drawn_1.jpg "Lane drawn"
[video1]: ./output_images/Video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2-11 code cells of the IPython notebook located in "./Advanced_Lane_Lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original
![alt text][image1]

Undistorted
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Original
![alt text][image1]

Undistorted
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at code cells 12-13-14).  Here's an example of my output for this step.

Undistorted
![alt text][image2]

Thresholded
![alt text][image3]

#### 3. After this I masked the noise from the image using aregion of interest mask. 

Defined in code cells 15-16-17-18. I used a trapezoid mask as in the first project for this.

Thresholded
![alt text][image3]

Masked

![alt text][image4]

#### 4. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in code cells 19-20-21. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([
    [250,720],
    [620,450],
    [720,450],
    [1200, 720]])
dst = np.float32([
    [300,720],
    [300,0],
    [1100,0],
    [1100, 720]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Masked
![alt text][image4]

Warped

![alt text][image5]

#### 5. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this is in the code cells 24-25-26-27.

I used the sliding window method described in the project for the identification of lane lines.

By using histogram to identify the 2 peaks, followed by collecting all the pixels in those peaks. Then fitting a polynomial to the 2 sets of pixels to identify the lane lines.


Warped
![alt text][image5]

Windowed

![alt text][image6]

Using previous frame

![alt text][image7]

#### 6. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the code cells 22 and 23

```
ym_per_pix = 30.0/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def calculate_curvature(lefty, righty, leftx, rightx):
    # Fit new polynomials to x,y in world space
    y_eval = np.max(lefty)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad

def check_vehicle_position(left_fit, right_fit, width, height):
    left_x_bottom = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    right_x_bottom = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    midpoint = (left_x_bottom + right_x_bottom) / 2.0
    image_mid = width / 2.0
    return image_mid - midpoint
```

#### 7. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in the code cell 28

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


Here's a [link to my video result](./output_images/Video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have took a basic pipeline approach. In the prooject video the pieline is working properly, however on the challenge video it's not working accurately. I have used the sanity check based on curvature to check on bad detections. That improved some performance on the challenge video, but I still need to apply smoothing to make it even better.
 
