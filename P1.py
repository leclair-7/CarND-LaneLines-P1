





import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#reading in an image
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def draw_lane_lines(lines):
    '''
    Takes output from cv2.HoughLinesP and outputs two polyfitted lines
    '''
    lines_augmented = [[i]+[(i[0][3] - i[0][1]) / (i[0][2] - i[0][0])]for i in lines]
    
    rightlines = [i[0] for i in lines_augmented if i[1] < 0 ]
    leftlines  = [i[0] for i in lines_augmented if i[1] > 0 ]
    #print(rightlines, len(rightlines))
    #cv2.line(img_hough_lines,(y_1L,x_1L), (y_2L,x_2L),(255,0,0),5)
    x_right = []
    y_right = []
    x_left = []
    y_left = []    
    for i in rightlines:
        x_right.append(i[0][0])
        x_right.append(i[0][2])
        y_right.append(i[0][1])
        y_right.append(i[0][3])
    for i in leftlines:
        x_left.append(i[0][0])
        x_left.append(i[0][2])
        y_left.append(i[0][1])
        y_left.append(i[0][3])
    #make the left line
    left_line = np.polyfit(x_right, y_right,1)
    #make the right line
    right_line = np.polyfit(x_left, y_left,1)
    
    return [left_line, right_line]
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    image_copy = image.copy()    
    '''
    GRAY -> Gaussian blur -> Canny Edge Detection
    (In that order)    
    '''
    gray = grayscale(image_copy)    
    blurred_gray = gaussian_blur(gray,7)    
    canny_low_threshold = 50
    canny_high_threshold = 150
    edges = canny(blurred_gray,canny_low_threshold, canny_high_threshold)
    
    '''
    MASK:
    Next we create the region of interest    
    '''
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    imshape = image_copy.shape
    
    xl,yl = (475, 315)
    xr,yr = (490, 315)
    vertices = np.array([[(0+50,imshape[0]),(xl,yl), (xr,yr), (imshape[1]-50,imshape[0])]], dtype=np.int32)
    lanes_of_image = region_of_interest(edges, vertices)
    #plt.imshow(lanes_of_image, cmap='gray')
    
    '''
    Hough Transform - lines addition    
    '''
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_len = 35
    max_line_gap = 18
    #black and red lines from the Houngh Transform 
    img_hough_lines = hough_lines(lanes_of_image, rho, theta, threshold, min_line_len, max_line_gap)    
    #plt.imshow(img_hough_lines)
    lines = cv2.HoughLinesP(lanes_of_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)    
    left_line, right_line = draw_lane_lines(lines)   
    
    x_bottom_left = 150
    xl -= 30
    left_bottom_point = (x_bottom_left,int(x_bottom_left*left_line[0] + left_line[1]))
    left_top_point = (xl,int(xl*left_line[0] + left_line[1]))
    x_bottom_right = 900
    xr += 30
    right_bottom_point = (xr,int(xr*right_line[0] + right_line[1]))
    right_top_point = (x_bottom_right,int(x_bottom_right*right_line[0] + right_line[1]))
    
    image_lanes = np.zeros((image_copy.shape[0], image_copy.shape[1], 3), dtype=np.uint8)
    cv2.line(image_lanes,left_bottom_point, left_top_point,[255,0,0],7 )    
    cv2.line(image_lanes,right_bottom_point, right_top_point,[255,0,0],7 )
    
    weighted_image = weighted_img(image_lanes, image, α=0.8, β=1., γ=0.)
    
    #print(weighted_image.shape)
    
    return weighted_image


white_output = 'white.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("solidWhiteRight.mp4").subclip(0,8)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


print("It finished")