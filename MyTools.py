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
   


    