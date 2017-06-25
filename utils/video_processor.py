import os
import cv2
from utils.camera_cal import CameraCalibration
from utils.binary_image import BinaryImage
from utils.perspective_transform import PerspectiveTransform
from utils.lane_finder import LaneFinder


class VideoProcessor:
    ''' Process a video for finding lanes '''
    cam_cal = None
    debug_dir = None
    debug_frame_number = 0
    debug_frame_bypass = 10

    @staticmethod
    def init(input_video, output_video, camera_cal_directory, debug_dir=None):
        ''' initialize static class members '''
        VideoProcessor.input_video = input_video
        VideoProcessor.output_video = output_video
        VideoProcessor.cam_cal = VideoProcessor.calibrate_camera(camera_cal_directory)
        VideoProcessor.debug_dir = debug_dir
        VideoProcessor.debug_frame_number = 0
        if debug_dir is not None:
            os.makedirs(os.path.join(debug_dir, 'original'), exist_ok=True)
            os.makedirs(os.path.join(debug_dir, 'undist'), exist_ok=True)
            os.makedirs(os.path.join(debug_dir, 'binary'), exist_ok=True)
            os.makedirs(os.path.join(debug_dir, 'perspective'), exist_ok=True)
            os.makedirs(os.path.join(debug_dir, 'lanes'), exist_ok=True)
            os.makedirs(os.path.join(debug_dir, 'final'), exist_ok=True)

    @staticmethod
    def write_debug_images(image, undist, binary, perspective, lanes, final):
        ''' Writes intermediate images to debug_dir '''
        filename = str(VideoProcessor.debug_frame_number) + '.png'
        cv2.imwrite(os.path.join(VideoProcessor.debug_dir, 'original', filename), image)
        cv2.imwrite(os.path.join(VideoProcessor.debug_dir, 'undist', filename), undist)
        cv2.imwrite(os.path.join(VideoProcessor.debug_dir, 'binary', filename), binary)
        cv2.imwrite(os.path.join(VideoProcessor.debug_dir, 'perspective', filename), perspective)
        cv2.imwrite(os.path.join(VideoProcessor.debug_dir, 'lanes', filename), lanes)
        cv2.imwrite(os.path.join(VideoProcessor.debug_dir, 'final', filename), final)

    @staticmethod
    def process_image(image):
        ''' process each frame image '''
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_undist = VideoProcessor.cam_cal.undistort(image)
        bin_img = BinaryImage(image_undist, kernel=5, grad_thresh=(20, 100),
                              sat_thresh=(120, 255), light_thresh=(45, 255),
                              mag_thresh=(30, 100), dir_thresh=(0.7, 1.3))
        image_binary = bin_img.get()
        pers_img = PerspectiveTransform(image_binary)
        image_perspective = pers_img.get()
        #image_perspective = cv2.cvtColor(image_perspective, cv2.COLOR_BGR2GRAY)
        finder = LaneFinder(image_perspective, cache=True)
        finder.slide_window(n_windows=9)
        image_perspective_overlay = finder.visualize(draw_lane_pixels=False, draw_on_image=False,
                                                     draw_windows=False)
        pers_img = PerspectiveTransform(image_perspective_overlay)
        image_overlay = pers_img.get_inverse()
        undist_overlay = cv2.addWeighted(image_undist, 1, image_overlay, 0.3, 0)
        undist_overlay = finder.draw_info(undist_overlay)
        if VideoProcessor.debug_dir is not None:
            if VideoProcessor.debug_frame_number % VideoProcessor.debug_frame_bypass == 0:
                image_lanes = finder.visualize()
                VideoProcessor.write_debug_images(image, image_undist, image_binary, image_perspective,
                                                  image_lanes, undist_overlay)
            VideoProcessor.debug_frame_number += 1
        return cv2.cvtColor(undist_overlay, cv2.COLOR_BGR2RGB)

    @staticmethod
    def calibrate_camera(camera_cal_directory):
        ''' calibrating camera '''
        filenames = os.listdir(camera_cal_directory)
        images = []
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = cv2.imread(os.path.join(camera_cal_directory, filename))
                images.append(img)
        cam_cal = CameraCalibration()
        cam_cal.compute(images)
        return cam_cal

    @staticmethod
    def process():
        ''' process input video '''
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(VideoProcessor.input_video)
        new_clip = clip.fl_image(VideoProcessor.process_image)
        new_clip.write_videofile(VideoProcessor.output_video, audio=False)

