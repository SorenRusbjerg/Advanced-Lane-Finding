import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
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

        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin when using window search 
        self.margin = 90
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        # Set the width of the windows +/- margin when using poly search 
        self.polymargin = 50
        
        # Number of frames to average polynomial fit
        self.N_Average = 6
        
        self.ImgHeight = 700
        self.ImgWidth  = 1200

    
    def find_lane_pixels(self, binary_warped, x_base):
        self.ImgHeight, self.ImgWidth = binary_warped.shape[:2]
        
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        x_current = x_base

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

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > self.minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))
                if window == 0: # If first window update x-value and best x value
                    self.recent_xfitted.append(x_current)
                    self.bestx =  np.mean(self.recent_xfitted[-self.N_Average:])
            else:
                Missing_detect += 1.0    
        
        # If window is not detected lot of windows, set missing detect
        self.detected = Missing_detect > self.nwindows * 0.6
 
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds]        

        return out_img


    def fit_polynomial_from_idx(self):
        # Fit a second order polynomial to each using `np.polyfit`
        newFit = np.polyfit(self.ally, self.allx, 2)        
        if len(self.current_fit) > 1:
            self.diffs = np.abs(self.current_fit[-1] - newFit)
            if (self.diffs[0] > 0.00 or self.diffs[1] > 0.3 or self.diffs[2] > 30):
                self.detected = False
            else:
                self.detected = True
                self.current_fit.append(newFit)
        else:
            self.current_fit.append(newFit)
            self.detected = True
        
        # Calculate best fit
        self.best_fit = np.mean(self.current_fit[-self.N_Average:], axis=0)
    
    
    def PlotPolynomialFit(self, img):
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
        
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.title('Poly fit ')

        # Plots the left and right polynomials on the lane lines
        plt.plot(fitx, ploty, color='yellow')
        
        return img_out
    
    
    def search_around_poly(self, binary_warped):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        lane_inds = (((nonzeroy**2 * self.best_fit[0] + nonzeroy * self.best_fit[1] + self.best_fit[2] - self.polymargin) < nonzerox) & 
        ((nonzeroy**2*self.best_fit[0] + nonzeroy*self.best_fit[1] + self.best_fit[2] + self.polymargin) > nonzerox))

        # Again, extract line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds]  
        
        # Calculate new polynomial coeefficients
        fit_polynomial_from_idx()

