# **Finding Lane Lines on the Road**

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image1]: ./examples/gray.png "Grayscale"
[image2]: ./examples/region-of-interest.png "Region"
[image3]: ./examples/hough_transform.png "hough"
[image4]: ./examples/pipeline_complete.png "Complete"

---

### Reflection

### 1. Description of the Pipeline.

##### My pipeline consisted of 6 steps.
1. Converted the images to grayscale.
2. Create the region of interest so the pipeline is only processing a region of the image.
3. Hough Transform in the region created by the previous step. A hough transform identifies lines in the image. This outputted an array of points of the lines.
4. Created 2 arrays left_lines and right_lines which took the previous step's output was sorted into.
5. Took the average of the both line array's slope and y-intercepts.
6. Calculated starting points and end points and plotted the result to the image.

#### Grayscale
![Gray][image1]
#### Region of Interest (and black everywhere else)
![Region of Interest][image2]
#### Hough Transform
![Hough Transform][image3]
#### Complete
![Complete][image4]

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the road is shaded or under a bridge. Also in rainy conditions this would not work. Another shortcoming could potentially be on a sharp turn where only 1 line can be seen.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to utilize different color spaces. A smoothing mechanism would help make the lines display less jittery on the output video.
