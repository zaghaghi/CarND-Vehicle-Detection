import numpy as np
import cv2
import time
import random
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils.vehicle_functions import *


class VehicleDetector:
    def __init__(self, options=None):
        self.normalizer = None
        self.model = None
        if options is not None and isinstance(options, VehicleDetectorOptions):
            self.options = options
        else:
            self.options = VehicleDetectorOptions()

    def train(self, vehile_dir, non_vehicle_dir):
        ''' Train a model on vehicle and non-vehicle images '''
        start_time = time.time()
        vehile_image_list = find_image_files(vehile_dir)
        non_vehile_image_list = find_image_files(non_vehicle_dir)
        end_time = time.time()
        print(round(end_time-start_time, 2), 'seconds to find images.')

        start_time = time.time()
        vehicle_features = extract_features_image_list(vehile_image_list, self.options)
        non_vehicle_features = extract_features_image_list(non_vehile_image_list, self.options)

        features = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
        # Fit a per-column scaler
        self.normalizer = RobustScaler().fit(features)
        # Apply the scaler to X
        norm_features = self.normalizer.transform(features)
        end_time = time.time()
        print(round(end_time-start_time, 2), 'seconds to extract.')

        # Define the labels vector
        labels = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

        norm_features, labels = shuffle(norm_features, labels)
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        print(norm_features.shape, labels.shape)
        X_train, X_test, y_train, y_test = train_test_split(norm_features, labels,
                                                            test_size=0.2, random_state=rand_state)

        print('Using:', self.options.orient, 'orientations', self.options.pix_per_cell,
              'pixels per cell and', self.options.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        self.model = LinearSVC()
        # Check the training time for the SVC
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        print(round(end_time-start_time, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.model.score(X_test, y_test), 4))


    def save(self, filename):
        ''' saves trained model to file '''
        obj = {'model': self.model, 'normalizer': self.normalizer, 'options': self.options}
        with open(filename, 'wb') as out_file:
            pickle.dump(obj, out_file)

    def load(self, filename):
        ''' loads trained model from file '''
        with open(filename, 'rb') as in_file:
            obj = pickle.load(in_file)
            self.normalizer = obj['normalizer']
            self.options = obj['options']
            self.model = obj['model']

    def find(self, image_filename, output_dir=None):
        ''' finds all cars in an image and returns a list of bounding boxes
            if output_dir isn't None, saves all intermediate images to output_dir '''
        image = cv2.imread(image_filename)
        bboxes1, _ = find_cars(image, 1.9, image.shape[0] // 2, image.shape[0],
                               self.normalizer, self.model, self.options)
        bboxes2, _ = find_cars(image, 1.6, image.shape[0] // 2, image.shape[0] * 4 // 5,
                               self.normalizer, self.model, self.options)
        bboxes3, _ = find_cars(image, 1.3, image.shape[0] // 2, image.shape[0] * 4 // 5,
                               self.normalizer, self.model, self.options)
        result_img = draw_boxes(image, bboxes1, color=(0, 0, 255), thick=2)
        result_img = draw_boxes(result_img, bboxes2, color=(255, 0, 0), thick=2)
        result_img = draw_boxes(result_img, bboxes3, color=(0, 255, 0), thick=2)
        _, filename = os.path.split(image_filename)
        cv2.imwrite(os.path.join(output_dir, filename), result_img)
