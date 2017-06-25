import numpy as np
import cv2


class Line:
    ''' keep track of detected lines '''
    average_count = 5
    def __init__(self):
        self.poly_fit = []
        self.lane_ind = []
        self.lane_start = []
        self.lane_x = []
        self.lane_y = []
        self.curve = []

    def add_poly_fit(self, poly_fit):
        self.poly_fit.append(poly_fit)
        if len(self.poly_fit) > Line.average_count:
            del self.poly_fit[0]

    def get_last_poly_fit(self):
        return self.poly_fit[-1]

    def get_average_poly_fit(self):
        return np.average(self.poly_fit, axis=0)

    def add_lane_ind(self, lane_ind):
        self.lane_ind.append(lane_ind)
        if len(self.lane_ind) > Line.average_count:
            del self.lane_ind[0]

    def get_last_lane_ind(self):
        return self.lane_ind[-1]

    def add_lane_points(self, x, y):
        if x is not None and y is not None:
            self.lane_x.append(x)
            self.lane_y.append(y)
        else:
            self.lane_x.append([])
            self.lane_y.append([])

    def get_lane_points_poly_fit(self, ym_per_pix=1, xm_per_pix=1):
        if Line.average_count == 1 or len(self.lane_x) < 3 or len(self.lane_y) < 3:
            return np.polyfit(self.lane_y[-1] * ym_per_pix, self.lane_x[-1] * xm_per_pix, 2)    
        else:
            lane_y = np.concatenate((np.repeat(self.lane_y[-1], 4),
                                     np.repeat(self.lane_y[-2], 2),
                                     self.lane_y[-3]))
            lane_x = np.concatenate((np.repeat(self.lane_x[-1], 4),
                                     np.repeat(self.lane_x[-2], 2),
                                     self.lane_x[-3]))
            return np.polyfit(lane_y * ym_per_pix, lane_x * xm_per_pix, 2)    

    def add_lane_start(self, lane_start):
        self.lane_start.append(lane_start)
        if len(self.lane_start) > Line.average_count:
            del self.lane_start[0]

    def get_last_lane_start(self):
        return self.lane_start[-1]

    def get_lane_start(self):
        return self.lane_start[-1]
        # if len(self.lane_start) == 1:
        #    return self.lane_start[0]
        # return (self.lane_start[-1] + self.lane_start[-2]) / 2.0

    def get_average_lane_start(self):
        return np.average(self.lane_start, axis=0)

    def add_curve(self, cr):
        self.curve.append(cr)
        if (len(self.curve)) > Line.average_count:
            del self.curve[0]
    
    def get_average_curve(self):
        return np.average(self.curve, axis=0)

    def get_last_curve(self):
        return self.curve[-1]

class LaneFinder:
    ''' Finds lanes from perspective image '''
    right_line = Line()
    left_line = Line()

    def __init__(self, image, cache=False):
        if not cache:
            Line.average_count = 1
            LaneFinder.right_line.average_count = 1
            LaneFinder.left_line.average_count = 1
        self.image = image
        if len(self.image.shape) != 2:
            raise Exception("Invalid image channels, expected 1 but {} provided.".\
                            format(len(self.image.shape)))
        self.histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        image_midpoint = self.histogram.shape[0]//2
        self.left_lane_start = np.argmax(self.histogram[:image_midpoint])
        self.right_lane_start = np.argmax(self.histogram[image_midpoint:]) + image_midpoint
        self.left_fit = None
        self.right_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.left_curverad = None
        self.right_curverad = None
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.distance_from_center = ((self.left_lane_start + self.right_lane_start) / 2 -
                                     self.image.shape[1] / 2)
        self.left_windows = []
        self.right_windows = []

    def slide_window(self, n_windows=9, window_width=100, min_pixel=50):
        ''' Slides a window on both lanes to find a polynomial fit '''
        window_height = np.int(self.image.shape[0]/n_windows)

        nonzero = self.image.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        LaneFinder.left_line.add_lane_start(self.left_lane_start)
        LaneFinder.right_line.add_lane_start(self.right_lane_start)

        left_current = LaneFinder.left_line.get_lane_start()
        right_current = LaneFinder.right_line.get_lane_start()

        left_lane_inds = []
        right_lane_inds = []
        for window in range(n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.image.shape[0] - (window + 1) * window_height
            win_y_high = self.image.shape[0] - window * window_height
            win_xleft_low = left_current - window_width
            win_xleft_high = left_current + window_width
            win_xright_low = right_current - window_width
            win_xright_high = right_current + window_width
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                              (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)
                             ).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                               (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)
                              ).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > min_pixel:
                left_current_ = np.mean(nonzero_x[good_left_inds])
                left_current = np.int(np.average((left_current, left_current_), weights=(0.4, 0.6)))
            if len(good_right_inds) > min_pixel:
                right_current_ = np.mean(nonzero_x[good_right_inds])
                right_current = np.int(np.average((right_current, right_current_), weights=(0.4, 0.6)))
            self.left_windows.append([(win_xleft_low, win_y_low), (win_xleft_high, win_y_high), len(good_left_inds)])
            self.right_windows.append([(win_xright_low, win_y_low), (win_xright_high, win_y_high), len(good_right_inds)])

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(left_lane_inds)
        self.right_lane_inds = np.concatenate(right_lane_inds)

        LaneFinder.left_line.add_lane_ind(self.left_lane_inds)
        LaneFinder.right_line.add_lane_ind(self.right_lane_inds)

        # Extract left and right line pixel positions
        left_x = nonzero_x[self.left_lane_inds]
        left_y = nonzero_y[self.left_lane_inds]
        right_x = nonzero_x[self.right_lane_inds]
        right_y = nonzero_y[self.right_lane_inds]

        LaneFinder.left_line.add_lane_points(left_x, left_y)
        LaneFinder.right_line.add_lane_points(right_x, right_y)

        # Fit a second order polynomial to each lane
        if len(left_x) > 0 and len(left_y) > 0:
            self.left_fit = LaneFinder.left_line.get_lane_points_poly_fit()
        else:
            self.left_fit = LaneFinder.left_line.get_last_poly_fit()
        if len(right_x) > 0 and len(right_y) > 0:
            self.right_fit = LaneFinder.right_line.get_lane_points_poly_fit()
        else:
            self.right_fit = LaneFinder.right_line.get_last_poly_fit()

        LaneFinder.left_line.add_poly_fit(self.left_fit)
        LaneFinder.right_line.add_poly_fit(self.right_fit)

        y_eval = self.image.shape[0] * self.ym_per_pix
        if len(left_x) > 0 and len(left_y) > 0:
            # Fit new polynomials to x,y in world space
            left_fit_cr = LaneFinder.left_line.get_lane_points_poly_fit(self.ym_per_pix, self.xm_per_pix)
            #np.polyfit(left_y * self.ym_per_pix, left_x * self.xm_per_pix, 2)
            left_1st_derivative = 2 * left_fit_cr[0] * y_eval + left_fit_cr[1]
            left_2nd_derivative = 2 * left_fit_cr[0]
            self.left_curverad = (((1 + (left_1st_derivative) ** 2) ** 1.5) /
                                  np.absolute(left_2nd_derivative))
        else:
            self.left_curverad = 0
        LaneFinder.left_line.add_curve(self.left_curverad)

        if len(right_x) > 0 and len(right_y) > 0:
            right_fit_cr = LaneFinder.right_line.get_lane_points_poly_fit(self.ym_per_pix, self.xm_per_pix)
            right_1st_derivative = 2 * right_fit_cr[0] * y_eval + right_fit_cr[1]
            right_2nd_derivative = 2 * right_fit_cr[0]
            self.right_curverad = (((1 + (right_1st_derivative) ** 2) ** 1.5) /
                                   np.absolute(right_2nd_derivative))
        else:
            self.right_curverad = 0
        LaneFinder.right_line.add_curve(self.right_curverad)

    def visualize(self, draw_on_image=True, draw_lane_pixels=True, draw_lane=True, draw_windows=True):
        ''' visualize founded lanes and windows on image '''
        #if self.left_fit is None or self.right_fit is None:
        #    return None
        vis_img = np.dstack((self.image, self.image, self.image))
        if not draw_on_image:
            vis_img = np.zeros_like(vis_img)
        ploty = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])

        left_fit = LaneFinder.left_line.get_average_poly_fit()
        right_fit = LaneFinder.right_line.get_average_poly_fit()

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        nonzero = self.image.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        if draw_lane:
            cv2.fillPoly(vis_img, np.int_([pts]), (0, 255, 0))

        if draw_lane_pixels:
            left_lane_inds = LaneFinder.left_line.get_last_lane_ind()
            right_lane_inds = LaneFinder.right_line.get_last_lane_ind()
            vis_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
            vis_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        if draw_windows:
            for win in self.left_windows:
                cv2.rectangle(vis_img, win[0], win[1], (255, 0, 0), 2)
                cv2.putText(vis_img, str(win[2]), (win[0][0] + 10, win[1][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            for win in self.right_windows:
                cv2.rectangle(vis_img, win[0], win[1], (255, 0, 0), 2)
                cv2.putText(vis_img, str(win[2]), (win[0][0] + 10, win[1][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        return vis_img

    def draw_info(self, image):
        ''' draw curve information on input image'''
        text = "Left Curve: {:6.2f}m".format(LaneFinder.left_line.get_average_curve())
        cv2.putText(image, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        text = "Right Curve: {:6.2f}m".format(LaneFinder.right_line.get_average_curve())
        cv2.putText(image, text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        text = "Distance from center {:4.2f}m to the {}"
        if self.distance_from_center < 0:
            text = text.format(-self.distance_from_center * self.xm_per_pix, 'left')
        elif self.distance_from_center == 0:
            text = "Distance from center 0m"
        else:
            text = text.format(self.distance_from_center * self.xm_per_pix, 'right')
        cv2.putText(image, text, (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image
