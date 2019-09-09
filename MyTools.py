# various tools used in exercises
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def GetFileNameFromFilePath(fileName):
    fileName = os.path.basename(fileName)
    return fileName[:-4]

    
def CannyDetect(img, low_threshold=80, high_threshold=240):
    # Use blurring and canny transform to obtain a trhesholding image
    # Gausian filter image for removing noise edges
    img_canny = cv2.GaussianBlur(img, (9, 9), 0)
    # Canny edge detection
    img_canny = cv2.Canny(img_canny, low_threshold, high_threshold)    
    return img_canny


def ColorThreshold(img):
    # Use Color transform to obtain a threshold image
    l_Thres = 200#200
    s_Thres = 90#100    
    Hue_Thres = 15#15 # Yellow lines
    Hue_Width = 5
    gray_Thres = 200
    
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img_hls[:,:,0]
    l = img_hls[:,:,1]
    s = img_hls[:,:,2]
    h_channel = np.zeros_like(h)
    l_channel = np.zeros_like(h)
    s_channel = np.zeros_like(h)
    lumin_channel = np.zeros_like(img_gray)
    
    h_channel[(h > Hue_Thres) & (h < Hue_Thres+Hue_Width)] = 0 # deactivated
    l_channel[l > l_Thres] = 255
    s_channel[s > s_Thres] = 255
    lumin_channel[img_gray > gray_Thres] = 255
    
    combined = h_channel + l_channel + s_channel + lumin_channel
    
    return combined, h_channel, l_channel, s_channel, lumin_channel


def CombineBinaryImages(img1, img2):
    # Combine images
    img_binary = np.zeros_like(img1)
    img_binary[img1.nonzero()] = 255
    img_binary[img2.nonzero()] = 255
    return img_binary


def MyImageWrite(fileName, savePath, img):
    cv2.imwrite(os.path.join(savePath, fileName),img)
    
def ProcessVideo(vidfile, out_folder, ProcessFunc):
    # Process video file using ProcessFunc(image) and save to out_folder
    fileOut = os.path.join(out_folder, GetFileNameFromFilePath(vidfile)) + vidfile[-4:]
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip(vidfile)
    white_clip = clip1.fl_image(ProcessFunc) #NOTE: this function expects color images!!
    white_clip.write_videofile(fileOut, audio=False)
    
    
def BirdsEyeLaneFromLeftAndRightPolynomials(img, leftPoly, rightPoly):    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = leftPoly[0]*ploty**2 + leftPoly[1]*ploty + leftPoly[2]
    right_fitx = rightPoly[0]*ploty**2 + rightPoly[1]*ploty + rightPoly[2]
    
    left_line  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    poly_line_pts = np.hstack((left_line, right_line))
    
    if len(img.shape)>2:
        window_img = np.zeros_like(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        window_img = np.zeros_like(img)

    cv2.fillPoly(window_img, np.int_([poly_line_pts]), (0,255, 0))

    return window_img


def MergeAndPlotBGRImages(img1, img2, text):
    img_out = cv2.addWeighted(img1, 1, img2, 0.3, 0)    
    
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title(text)
    
    
    
def CalculateLaneOffset(leftLine, RightLine):
    carCenter = leftLine.ImgWidth//2
    roadCenter = (RightLine.bestx - leftLine.bestx)//2 + leftLine.bestx
    
    xOffset = (roadCenter - carCenter)*RightLine.m_per_pix_x
    
    return xOffset
   


def CalculatePerspectiveTransform(imgShape, y_upperValue):
    # From Image points ([(210, dimY),(590, 455),(695, 455),(1120, dimY)])
    #Calculate source and destination points using linear equation
    dimY, dimX = imgShape
    slopeL = (590-210)/(455-dimY)
    offsetL = 210-slopeL*dimY
    x_upper_L = y_upperValue*slopeL+offsetL

    slopeR = (695-1120)/(455-dimY)
    offsetR = 1120-slopeR*dimY
    x_upper_R = y_upperValue*slopeR+offsetR
    src = np.float32([(210, dimY),(x_upper_L, y_upperValue),(x_upper_R, y_upperValue),(1120, dimY)])
    # destination points
    dst = np.float32([(400, dimY),(400, 0),(1000, 0),(1000, dimY)])

    return src, dst