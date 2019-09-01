# various tools used in exercises
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def GetFileNameFromFilePath(fileName):
    fileName = os.path.basename(fileName)
    return fileName[:-4]
    

def MyImageWrite(fileName, savePath, img):
    cv2.imwrite(os.path.join(savePath, fileName),img)
    
    
def FindLeftAndRightLaneXbaseFromHist(img,minXidx=0,maxXidx=1200):
    # return x values in image from where most pixels are found in histogram    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[minXidx:midpoint]) + minXidx
    rightx_base = np.argmax(histogram[midpoint:maxXidx]) + midpoint    
    
    return leftx_base, rightx_base

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
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title(text)
    
    
    
def CalculateLaneOffset(leftLine, RightLine):
    carCenter = leftLine.ImgWidth//2
    roadCenter = (RightLine.bestx - leftLine.bestx)//2 + leftLine.bestx
    
    xOffset = (roadCenter - carCenter)*RightLine.m_per_pix_x
    
    return xOffset
   
def CalculateLineCurvature(line):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''   
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = line.ImgHeight 

    curverad = np.power(1 + (2*line.best_fit[0]*y_eval + line.best_fit[1])**2, 1.5)/(2*np.abs(line.best_fit[0]))
    return curverad 

    