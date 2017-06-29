import numpy as np
import cv2
import time
import random
import pickle
from skimage.feature import hog
from skimage.measure import label
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils.vehicle_functions import *


class VehicleDetector:
    def __init__(self, options=None):
        self.normalizer = None
        self.model = None
        self.test_acc = None
        if options is not None and isinstance(options, VehicleDetectorOptions):
            self.options = options
        else:
            self.options = VehicleDetectorOptions()

    def train(self, vehile_dir, non_vehicle_dir):
        ''' Train a model on vehicle and non-vehicle images '''
        # 1. Find all train images in vehicle and non-vehicle directories
        start_time = time.time()
        vehile_image_list = find_image_files(vehile_dir)
        non_vehile_image_list = find_image_files(non_vehicle_dir)
        end_time = time.time()
        print(round(end_time-start_time, 2), 'seconds to find images.')

        # 2. Extract features for the train images
        start_time = time.time()
        vehicle_features = extract_features_image_list(vehile_image_list, self.options)
        non_vehicle_features = extract_features_image_list(non_vehile_image_list, self.options)

        # 3. Stack vehicle and non-vehicle features in one numpy array
        features = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)

        # 4. Normalize the features using `StandardScaler`
        self.normalizer = StandardScaler().fit(features)
        # Apply the scaler to X
        norm_features = self.normalizer.transform(features)
        end_time = time.time()
        print(round(end_time-start_time, 2), 'seconds to extract.')

        # 5. Build label array with ones and zeros for vehicle and non-vehicle features respectively
        labels = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

        # 6. Shuffle features and labels and then split them into train and test features and labels
        norm_features, labels = shuffle(norm_features, labels)
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        print(norm_features.shape, labels.shape)
        X_train, X_test, y_train, y_test = train_test_split(norm_features, labels,
                                                            test_size=0.2, random_state=rand_state)

        print('Using:', self.options.orient, 'orientations', self.options.pix_per_cell,
              'pixels per cell and', self.options.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # 7. Build an instance of `LinearSVC` model and `fit` it on the train features. I used default model parameters
        self.model = LinearSVC()
        # Check the training time for the SVC
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        print(round(end_time-start_time, 2), 'Seconds to train SVC...')

        # 8. Test the model on test features
        self.test_acc = self.model.score(X_test, y_test)
        print('Test Accuracy of SVC = ', round(self.test_acc, 4))

    def save(self, filename):
        ''' saves trained model to file '''
        obj = {'model': self.model, 'normalizer': self.normalizer, 'options': self.options, 'acc': self.test_acc}
        with open(filename, 'wb') as out_file:
            pickle.dump(obj, out_file)

    def load(self, filename):
        ''' loads trained model from file '''
        with open(filename, 'rb') as in_file:
            obj = pickle.load(in_file)
            self.normalizer = obj['normalizer']
            self.options = obj['options']
            self.model = obj['model']
            self.test_acc = obj['acc']

    def find(self, image):
        ''' finds all cars in an image and returns a list of bounding boxes
            if output_dir isn't None, saves all intermediate images to output_dir '''
        # Find cars in multi-scale image
        scale_config = [(0.95, 400, 512, 1),
                        (1.0, 360, 512, 1),
                        (1.5, 380, 580, 1),
                        (2.0, 400, 580, 1)
                       ]
        bboxes = []
        for cfg in scale_config:
            bb, _ = find_cars(image, cfg[0], cfg[1], cfg[2], cfg[3],
                              self.normalizer, self.model, self.options)
            bboxes.extend(bb)
        # Draw bounding boxes
        bbox_img = image
        bbox_img = draw_boxes(bbox_img, bboxes, color=(0, 0, 255), thick=2)

        # Build heatmap
        heatmap = np.zeros_like(image)
        for bbox in bboxes:
            heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
        # Threshold heatmap
        threshold = 3
        heatmap[heatmap < threshold] = 0
        heat_scale = 5
        heatmap_cmap = cv2.applyColorMap(heatmap*heat_scale, cv2.COLORMAP_HOT)

        # Find connected components using skimage.measure.label
        heatmap[heatmap > 0] = 1
        labelmap, label_num = label(heatmap, background=0, return_num=True)
        if label_num > 0:
            labelmap_cmap = labelmap*255//label_num
        else:
            labelmap_cmap = labelmap

        # Find final bounding box
        final_bboxes = []
        for lbl in range(1, label_num+1):
            label_indices = (labelmap == lbl).nonzero()
            bbox = ((np.min(label_indices[1]), np.min(label_indices[0])),
                    (np.max(label_indices[1]), np.max(label_indices[0])))
            final_bboxes.append(bbox)
        final_image = draw_boxes(image, final_bboxes, color=(0, 0, 255), thick=2)

        return final_bboxes, final_image, labelmap_cmap, heatmap_cmap, bbox_img
