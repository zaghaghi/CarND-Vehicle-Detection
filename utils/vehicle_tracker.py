import numpy as np
import cv2
from skimage.measure import label

class VehicleTracker:
    ''' Keep track of founded vehicles '''
    aggrigate_count = 12
    aggrigate_min_thres = 2
    def __init__(self):
        self.bboxes_list = []
        self.label_img_list = []

    def add_bboxes(self, bboxes):
        ''' add current frame bounding box to cache '''
        self.bboxes_list.append(bboxes)
        if len(self.bboxes_list) > self.aggrigate_count:
            del self.bboxes_list[0]

    def get_stable_bboxes(self, image_shape):
        ''' compute stable bounding boxes based on previous frames' bounding boxes '''
        if len(self.bboxes_list) == 1:
            return self.bboxes_list[0]
        agg_label_img = np.zeros(image_shape, np.uint8)
        for bboxes in self.bboxes_list:
            for bbox in bboxes:
                agg_label_img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

        agg_label_img[agg_label_img <= self.aggrigate_min_thres] = 0
        stable_label_img, label_num = label(agg_label_img > 0, return_num=True)

        stable_bboxes = []
        for lbl in range(1, label_num+1):
            max_num_in_agg_label = np.max(agg_label_img[stable_label_img == lbl])
            if max_num_in_agg_label < 1 + len(self.bboxes_list) // 2:
                continue
            label_indices = (stable_label_img == lbl).nonzero()
            bbox = ((np.min(label_indices[1]), np.min(label_indices[0])),
                    (np.max(label_indices[1]), np.max(label_indices[0])))
            stable_bboxes.append(bbox)
        return stable_bboxes

