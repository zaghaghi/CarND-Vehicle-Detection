rem uncomment bellow line to do hyper train
rem python vehicle_detection.py hyper-train --vehicle-dir dataset\vehicles --non-vehicle-dir dataset\non-vehicles
python vehicle_detection.py test-features --vehicle-dir dataset\vehicles --non-vehicle-dir dataset\non-vehicles --output-dir examples
python vehicle_detection.py train --vehicle-dir dataset\vehicles --non-vehicle-dir dataset\non-vehicles --model model.p
python vehicle_detection.py test --model model.p --input-dir test_images --output-dir output_images