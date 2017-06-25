import cv2
import numpy as np

class PerspectiveTransform:
    ''' transforms the image to a bird-eyes view '''
    def __init__(self, image):
        self.image = image
        self.img_size = (image.shape[1], image.shape[0])
        self.src = np.float32(
            [[(self.img_size[0] / 2) - 65, self.img_size[1] / 2 + 100],
             [((self.img_size[0] / 6) - 10), self.img_size[1]],
             [(self.img_size[0] * 5 / 6) + 60, self.img_size[1]],
             [(self.img_size[0] / 2 + 65), self.img_size[1] / 2 + 100]])
        self.dst = np.float32(
            [[(self.img_size[0] / 5), -20],
             [(self.img_size[0] / 5), self.img_size[1]],
             [(self.img_size[0] * 4 / 5), self.img_size[1]],
             [(self.img_size[0] * 4 / 5), -20]])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def get(self):
        ''' returns perspective transformed image using transformation matrix '''
        return cv2.warpPerspective(self.image, self.M, self.img_size, flags=cv2.INTER_LINEAR)

    def get_inverse(self):
        ''' returns perspective transformed image using inverse transformation matrix '''
        return cv2.warpPerspective(self.image, self.Minv, self.img_size, flags=cv2.INTER_LINEAR)

    def visualize(self):
        ''' visualize perspective transformation effect '''
        image = self.image.copy()
        cv2.line(image, tuple(self.src[0]), tuple(self.src[1]), (255, 0, 0), 3)
        cv2.line(image, tuple(self.src[1]), tuple(self.src[2]), (255, 0, 0), 3)
        cv2.line(image, tuple(self.src[2]), tuple(self.src[3]), (255, 0, 0), 3)
        cv2.line(image, tuple(self.src[3]), tuple(self.src[0]), (255, 0, 0), 3)
        image_perspective = self.get()
        cv2.line(image_perspective, tuple(self.dst[0]), tuple(self.dst[1]), (255, 0, 0), 3)
        cv2.line(image_perspective, tuple(self.dst[1]), tuple(self.dst[2]), (255, 0, 0), 3)
        cv2.line(image_perspective, tuple(self.dst[2]), tuple(self.dst[3]), (255, 0, 0), 3)
        cv2.line(image_perspective, tuple(self.dst[3]), tuple(self.dst[0]), (255, 0, 0), 3)
        image = cv2.copyMakeBorder(image, 10, 10, 10, 10,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
        image_perspective = cv2.copyMakeBorder(image_perspective, 10, 10, 10, 10,
                                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return np.hstack((image, image_perspective))
