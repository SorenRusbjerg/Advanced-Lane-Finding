import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, laneSide):
        # was the line detected in the last iterations?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = []  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        
        # HYPERPARAMETERS
        self.m_per_pix_y = 40/720 # meters per pixel in y dimension 
        self.m_per_pix_x = 3.7/600 # meters per pixel in x dimension

        # Threshold for finding new polynomial fit from scratch
        self.BadFramesThreshold = 10
        # Nr of bad frames detected in a row
        self.NrOfBadFrames = 0 
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin when using window search 
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 1000
        # Set minimum number of pixels found to include new pixels in polynomial
        self.minpixPoly = 10000
        # Set the width of the windows +/- margin when using poly search 
        self.polymargin = 50
        # Max allowed change to x-position
        self.x_threshold = 70        
        # Number of frames to average polynomial fit
        self.N_Average = 6
        
        self.ImgHeight = 700
        self.ImgWidth  = 1200
        
        if laneSide == 'left':
            self.laneSide = 'left'
        else:
            self.laneSide = 'right'

    
    def find_lane_pixels_from_windows(self, binary_warped):
        # find lane pixel positions from start x-index and using window-boxing along the y-axis 
        self.ImgHeight, self.ImgWidth = binary_warped.shape[:2]
        
        # Calculate bestx from histogram
        self.FindLaneXbaseFromHist(binary_warped)
        
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        x_current = self.bestx

        # Create empty lists to receive lane pixel indices
        lane_inds = []

        Missing_detect = 0
        out_img = np.zeros_like(binary_warped)
        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - self.margin
            win_x_high = x_current + self.margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,255,0), 2) 
  
            # Identify the nonzero pixels in x and y within the window #
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > self.minpix:
                # Append these indices to the lists
                print('Pixels found in {} window: {} pixels: {} '.format(self.laneSide , window, len(good_inds)))
                lane_inds.append(good_inds)
                x_current = np.int(np.mean(nonzerox[good_inds]))
                if window == 0: # If first window update x-value and best x value
                    self.recent_xfitted.append(x_current)
                    self.bestx =  np.mean(self.recent_xfitted[-self.N_Average:])
            else:
                Missing_detect += 1.0    
        
        # If window is not detected in lots of windows, set missing detect
        self.detected = Missing_detect < self.nwindows * 0.6
 
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds]   
        
        # Calculate new polynomial coeefficients
        self.fit_polynomial_from_idx()

        return out_img


    def fit_polynomial_from_idx(self):
        # Fit a second order polynomial to each using `np.polyfit`
        if len(self.allx) > 2:
            newFit = np.polyfit(self.ally, self.allx, 2)  
        else:
            newFit = np.array([0,0,self.bestx])
            
        self.CheckPolynomialAndSetBestFit(newFit)
                
        
    def CheckPolynomialAndSetBestFit(self, polyFit):
        # Check for correct Curvature and x-base value and set best_fit polynomial 
        minCurve = 150
        
        xPosFromPoly = self.CalculateXposFromPoly(polyFit)
        x_diff = np.abs(self.bestx - xPosFromPoly) # calculate xpos-difference from best fit
        # Check for absolute position on the road is sensible
        if self.laneSide == 'right':
            badPos = self.ImgWidth//2 > xPosFromPoly
        else:
            badPos = self.ImgWidth//2 < xPosFromPoly
        
        
        lineCurvature = self.CalculateLineCurvature(polyFit)        
        if ((lineCurvature < minCurve) or (x_diff > self.x_threshold) or (badPos==True)):            
            print('lineCurvature: {:.0f}, x-poly: {}, x_diff: {:.1f}, badPos: {}'.format(lineCurvature,xPosFromPoly,x_diff,badPos))
            # Make sure that there always exists a best fit poly
            if self.best_fit == []:
               self.best_fit = [0, 0, self.bestx] # Straight line 
            
            self.NrOfBadFrames += 1 
            if self.NrOfBadFrames > self.BadFramesThreshold:
                self.detected = False
                self.current_fit = []  
        else:
            self.radius_of_curvature = lineCurvature
            self.current_fit.append(polyFit)
            # Calculate best fit
            self.best_fit = np.mean(self.current_fit[-self.N_Average:], axis=0)
            self.NrOfBadFrames = 0
        
        
    
    def PlotPolynomialFit(self, img, plot=1):
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        try:
            fitx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        except TypeError:
            # Avoids an error if fit are still none or incorrect
            print('The function failed to fit a line!')

        if len(img.shape)>2:
            img_out = img.copy()
        else:
            img_out = np.dstack((img,img,img))
        ## Visualization ##
        # Colors in the lane regions
        img_out[self.ally, self.allx] = [255, 0, 0]
        
        if plot==1:
            plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
            plt.title('Poly fit ')

            # Plots the left and right polynomials on the lane lines
            plt.plot(fitx, ploty, color='yellow')
        
        return img_out
    
    
    def FindLaneXbaseFromHist(self, img):
        # return x values in image from where most pixels are found in histogram    
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        if self.laneSide == 'left':
            maxXidx = self.ImgWidth//2
            minXidx = 0
        else:
            maxXidx = self.ImgWidth
            minXidx = self.ImgWidth//2
        
        self.bestx = np.argmax(histogram[minXidx:maxXidx]) + minXidx
        

    def find_lane_pixels_around_poly(self, binary_warped):
        # Search after lane pixel positions along the best fit polynomial
        
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        lane_inds = (
            ((nonzeroy**2 * self.best_fit[0] + nonzeroy * self.best_fit[1] + self.best_fit[2] - self.polymargin) < nonzerox) & 
            ((nonzeroy**2 * self.best_fit[0] + nonzeroy * self.best_fit[1] + self.best_fit[2] + self.polymargin) > nonzerox))
        
        if (np.sum(lane_inds) > self.minpixPoly):
            print('Pixels found in {} poly pixels: {} '.format(self.laneSide, np.sum(lane_inds)))
            # Again, extract line pixel positions
            self.allx = nonzerox[lane_inds]
            self.ally = nonzeroy[lane_inds]  
            # Calculate new polynomial coeefficients
            self.fit_polynomial_from_idx()
        else:
            self.NrOfBadFrames += 1 
            if self.NrOfBadFrames > self.BadFramesThreshold:
                self.detected = False
                self.current_fit = []  
        
        
    def find_lane_pixels(self, img):
        # Find lane pixels using one of two methods based on if line is already detected
        if self.detected:
            # Use poly search
            self.find_lane_pixels_around_poly(img)
        else:
            self.find_lane_pixels_from_windows(img)
        


    def CalculateLineCurvature(self, polyFit):
        # Use best curve fit in pixel as input and find curvature at bottom of road in meters
        '''
        Calculates the curvature of polynomial functions in pixels.
        '''   
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = self.ImgHeight*self.m_per_pix_y 
        
        # Rcurve = (1+(2*A*y+B)²)³/² / abs(2*A)   
        A = polyFit[0]*self.m_per_pix_x/(self.m_per_pix_y**2) # scaled A from pix to meters
        B = polyFit[1]*self.m_per_pix_x/(self.m_per_pix_y)    # scaled B from pix to meters
        radius_of_curvature = np.power(1 + (2*A*y_eval + B**2), 1.5)/(2*np.abs(A))
        return radius_of_curvature
    
    def CalculateXposFromPoly(self, polyFit):
        Ypos = self.ImgHeight
        Xpos = Ypos**2 * polyFit[0] + Ypos * polyFit[1] + polyFit[2]
        return Xpos
                                            