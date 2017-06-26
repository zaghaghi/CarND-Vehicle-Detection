import os
import click
import random
import cv2
import numpy as np
from utils import VehicleDetector
from utils.vehicle_functions import *

general_cli = click.Group()

@general_cli.command('test-features')
@click.option('--vehicle-dir', help='Input directory of vehicle images.')
@click.option('--non-vehicle-dir', help='Input directory of non-vehicle images.')
@click.option('--output-dir', help='Output directory of feature images.')
def features(vehicle_dir, non_vehicle_dir, output_dir):
    # Extract a random samples from car and not-car images
    vehicle_images = find_image_files(vehicle_dir)
    non_vehicle_images = find_image_files(non_vehicle_dir)
    sample_vehicle_image = cv2.imread(random.sample(vehicle_images, 1)[0])
    sample_non_vehicle_image = cv2.imread(random.sample(non_vehicle_images, 1)[0])
    sample_vehicle_image = make_frame(sample_vehicle_image, "Vehicle sample")
    sample_non_vehicle_image = make_frame(sample_non_vehicle_image, "Non vehivle sample")
    sample_image = np.hstack((sample_vehicle_image, sample_non_vehicle_image))
    cv2.imwrite(os.path.join(output_dir, 'car_not_car.png'), sample_image)

    # Extract random samples and compute its HOG image
    samples = 4
    options = VehicleDetectorOptions()
    sample_vehicle_images = random.sample(vehicle_images, samples)
    sample_non_vehicle_images = random.sample(non_vehicle_images, samples)
    rows = []
    for idx in range(samples):
        v_image = cv2.imread(sample_vehicle_images[idx])
        v_image_cvt = convert_color(v_image, options.color_space)
        _, v_hog_0 = get_hog_features(v_image_cvt[:, :, 0], options.orient, options.pix_per_cell,
                                      options.cell_per_block, vis=True, feature_vec=False)
        _, v_hog_1 = get_hog_features(v_image_cvt[:, :, 1], options.orient, options.pix_per_cell,
                                      options.cell_per_block, vis=True, feature_vec=False)
        _, v_hog_2 = get_hog_features(v_image_cvt[:, :, 2], options.orient, options.pix_per_cell,
                                      options.cell_per_block, vis=True, feature_vec=False)
        nv_image = cv2.imread(sample_non_vehicle_images[idx])
        nv_image_cvt = convert_color(nv_image, options.color_space)
        _, nv_hog_0 = get_hog_features(nv_image_cvt[:, :, 0], options.orient, options.pix_per_cell,
                                       options.cell_per_block, vis=True, feature_vec=False)
        _, nv_hog_1 = get_hog_features(nv_image_cvt[:, :, 1], options.orient, options.pix_per_cell,
                                       options.cell_per_block, vis=True, feature_vec=False)
        _, nv_hog_2 = get_hog_features(nv_image_cvt[:, :, 2], options.orient, options.pix_per_cell,
                                       options.cell_per_block, vis=True, feature_vec=False)
        rows.append(np.hstack((make_frame(v_image, "Vehicle image"),
                               make_frame(v_hog_0*255, "HOG ch 1"),
                               make_frame(v_hog_1*255, "HOG ch 2"),
                               make_frame(v_hog_2*255, "HOG ch 3"),
                               make_frame(nv_image, "Non-Vehicle image"),
                               make_frame(nv_hog_0*255, "HOG ch 1"),
                               make_frame(nv_hog_1*255, "HOG ch 2"),
                               make_frame(nv_hog_2*255, "HOG ch 3"),
                              )))
    cv2.imwrite(os.path.join(output_dir, 'hog_example.png'), np.vstack(rows))

@general_cli.command('train')
@click.option('--vehicle-dir', help='Input directory of vehicle images.')
@click.option('--non-vehicle-dir', help='Input directory of non-vehicle images.')
@click.option('--model', default='model.p', help='Output model filename to save.')
def train(vehicle_dir, non_vehicle_dir, model):
    detector = VehicleDetector()
    detector.train(vehicle_dir, non_vehicle_dir)
    detector.save(model)

@general_cli.command('hyper-train')
@click.option('--vehicle-dir', help='Input directory of vehicle images.')
@click.option('--non-vehicle-dir', help='Input directory of non-vehicle images.')
def hyper_train(vehicle_dir, non_vehicle_dir):
    for color_space in ['RGB', 'YCrCb', 'LUV', 'HSV', 'YUV', 'HLS']:
        for hog_channel in [0, 1, 2, 'ALL']:
            for orient in [8, 9, 10, 11]:
                model_filename = 'model_{}_{}_{}.p'.format(color_space, hog_channel, orient)
                if os.path.exists(model_filename):
                    continue
                options = VehicleDetectorOptions()
                options.color_space = color_space
                options.orient = orient
                options.pix_per_cell = 8
                options.cell_per_block = 2
                options.hog_channel = hog_channel
                options.spatial_size = (16, 16)
                options.hist_bins = 32
                options.spatial_feat = False
                options.hist_feat = False
                options.hog_feat = True
                detector = VehicleDetector(options)
                detector.train(vehicle_dir, non_vehicle_dir)
                detector.save(model_filename)

@general_cli.command('print-info')
@click.option('--model', help='Input model filename.')
def print_info(model):
    detector = VehicleDetector()
    detector.load(model)
    print(detector.options)
    print("TestAccuracy(acc={})".format(round(detector.test_acc, 4)))

@general_cli.command('test')
@click.option('--model', default='model.p', help='Input model filename.')
@click.option('--input-dir', help='Input directory contains test images.')
@click.option('--output-dir', help='Output directory of pipeline images.')
def test(model, input_dir, output_dir):
    detector = VehicleDetector()
    detector.load(model)
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            detector.find(os.path.join(input_dir, filename), output_dir)

    # stack output images in a single image suitable for writeup
    for out_type in ['bbox', 'heat', 'label', 'final']:
        merge_filename = '{}_thumb_map.png'.format(out_type)
        thumb_map = make_image_thumb_map(os.path.join(output_dir, out_type), 3.0/10.0)
        cv2.imwrite(os.path.join(output_dir, merge_filename), thumb_map)

@general_cli.command('process-video')
@click.option('--model', default='model.p', help='Input model filename.')
@click.option('--input-file', help='Input video file.',
              prompt='Input video')
@click.option('--output-file', help='Output video file.',
              prompt='Output video')
@click.option('--lane-detection', default=False)
@click.option('--debug', default=False)
def process_video(model, input_file, output_file, lane_detection, debug):
    pass

if __name__ == '__main__':
    general_cli()

