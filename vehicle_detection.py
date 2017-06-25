import os
import click
from utils import VehicleDetector

general_cli = click.Group()

@general_cli.command('train')
@click.option('--vehicle-dir', help='Input directory of vehicle images.')
@click.option('--non-vehicle-dir', help='Input directory of non-vehicle images.')
@click.option('--model', default='model.p', help='Output model filename to save.')
def train(vehicle_dir, non_vehicle_dir, model):
    detector = VehicleDetector()
    detector.train(vehicle_dir, non_vehicle_dir)
    detector.save(model)

@general_cli.command('test')
@click.option('--model', default='model.p', help='Input model filename.')
@click.option('--input-dir', help='Input directory contains test images.')
@click.option('--output-dir', help='Output directory of pipeline images.')
def test(model, input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            detector = VehicleDetector()
            detector.load(model)
            detector.find(os.path.join(input_dir, filename), output_dir)


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

