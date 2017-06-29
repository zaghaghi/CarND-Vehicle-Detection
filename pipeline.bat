rem uncomment bellow line to do hyper train
rem python vehicle_detection.py hyper-train --vehicle-dir dataset\vehicles --non-vehicle-dir dataset\non-vehicles

rem generate images based on selected parameters and features
rem python vehicle_detection.py test-features --vehicle-dir dataset\vehicles --non-vehicle-dir dataset\non-vehicles --output-dir examples

rem train model with parameters in VehicleDetectorOptions in vehicle_function.py
rem python vehicle_detection.py train --vehicle-dir dataset\vehicles --non-vehicle-dir dataset\non-vehicles --model model.p

rem run model on test images
python vehicle_detection.py test --model model.p --input-dir test_images --output-dir output_images

rem run model on video sample images
python vehicle_detection.py test --model model.p --input-dir test_images\video_images --output-dir output_images\video_output_images

rem run model on test_video
python vehicle_detection.py process-video --model model.p --input-file test_video.mp4 --output-file test_output.mp4 --camera-cal camera.p

rem run model on project_video
python vehicle_detection.py process-video --model model.p --input-file project_video.mp4 --output-file project_output.mp4 --camera-cal camera.p --lane-detection
