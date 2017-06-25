import os
import numpy as np
import cv2
from skimage.feature import hog

class VehicleDetectorOptions:
    color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 8 # HOG orientations
    pix_per_cell = 9 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    def __repr__(self):
        return ','.join([self.color_space, str(self.orient), str(self.pix_per_cell),
                         str(self.cell_per_block), str(self.hog_channel),
                         str(self.spatial_size), str(self.hist_bins),
                         str(self.spatial_feat), str(self.hist_feat),
                         str(self.hog_feat)])

def find_image_files(input_dir):
    ''' Finds all jpg and png images in input_dir '''
    filenames = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                filenames.append(os.path.join(root, filename))
    return filenames

def convert_color(img, conv='YCrCb'):
    ''' converts color space '''
    if conv == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    ''' Define a function to return HOG features and visualization '''
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=True, block_norm='L2-Hys',
               visualise=vis, feature_vector=feature_vec)

def bin_spatial(img, size=(32, 32)):
    ''' Define a function to compute binned color features '''
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    ''' Define a function to compute color histogram features '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features_image(img, options=None):
    ''' Define a function to extract features from a single image window '''
    if options is None or not isinstance(options, VehicleDetectorOptions):
        options = VehicleDetectorOptions()
    # Define an empty list to receive features
    img_features = []
    # Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, options.color_space)
    # Compute spatial features if flag is set
    if options.spatial_feat:
        spatial_features = bin_spatial(feature_image, size=options.spatial_size)
        # Append features to list
        img_features.append(spatial_features)
    # Compute histogram features if flag is set
    if options.hist_feat:
        hist_features = color_hist(feature_image, nbins=options.hist_bins)
        # Append features to list
        img_features.append(hist_features)
    # Compute HOG features if flag is set
    if options.hog_feat:
        if options.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     options.orient, options.pix_per_cell,
                                                     options.cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, options.hog_channel],
                                            options.orient,
                                            options.pix_per_cell, options.cell_per_block,
                                            vis=False, feature_vec=True)
        # Append features to list
        img_features.append(hog_features)

    # Return concatenated array of features
    return np.concatenate(img_features)

def extract_features_image_list(imgs, options):
    ''' Define a function to extract features from a list of images '''
    if options is None or not isinstance(options, VehicleDetectorOptions):
        options = VehicleDetectorOptions()
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        # Extract features and append to list of features
        features.append(extract_features_image(image, options))
    # Return list of feature vectors
    return features


def find_cars(img, scale, ystart, ystop, normalizer, svc, options):
    ''' Define a single function that can extract features
        using hog sub-sampling and make predictions '''
    if options is None or not isinstance(options, VehicleDetectorOptions):
        options = VehicleDetectorOptions()
    draw_img = np.copy(img)
    #img = img.astype(np.float32) / 255

    if ystart is None:
        ystart = 0
    if ystop is None:
        ystop = img.shape[0]
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv=options.color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    hog_features = []
    if options.hog_feat:
        if options.hog_channel == 'ALL':
            for channel in range(ctrans_tosearch.shape[2]):
                ch = ctrans_tosearch[:, :, channel]
                hog_features.append(get_hog_features(ch, options.orient, options.pix_per_cell,
                                                     options.cell_per_block, feature_vec=False))
        else:
            ch = ctrans_tosearch[:, :, options.hog_channel]
            hog_features.append(get_hog_features(ch, options.orient, options.pix_per_cell,
                                                 options.cell_per_block, feature_vec=False))

    nxblocks = (ctrans_tosearch.shape[1] // options.pix_per_cell) - options.cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // options.pix_per_cell) - options.cell_per_block + 1

    window = 64
    nblocks_per_window = (window // options.pix_per_cell) - options.cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            patch_features = []

            xleft = xpos* options.pix_per_cell
            ytop = ypos* options.pix_per_cell

            # Extract the image patch
            if options.spatial_feat or options.hist_feat:
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))

                # Get color features
                if options.spatial_feat:
                    spatial_features = bin_spatial(subimg, size=options.spatial_size)
                    patch_features.append(spatial_features)
                if options.hist_feat:
                    hist_features = color_hist(subimg, nbins=options.hist_bins)
                    patch_features.append(hist_features)

            # Extract HOG for this patch
            if options.hog_feat:
                patch_hog_features = []
                for hog_feat in hog_features:
                    patch_hog_features.append(hog_feat[ypos:ypos+nblocks_per_window,
                                                       xpos:xpos+nblocks_per_window].ravel())
                patch_features.append(np.hstack(patch_hog_features))

            # Scale features and make a prediction
            test_features = normalizer.transform(np.concatenate(patch_features).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw + ystart),
                               (xbox_left + win_draw, ytop_draw + win_draw+ystart)))
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw+win_draw + ystart), (0, 0, 255), 6)

    return bboxes, draw_img

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    ''' Define a function to draw bounding boxes '''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

