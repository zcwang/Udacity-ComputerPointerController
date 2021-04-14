python3 src/main.py --face_model models/intel/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001 --head_model models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001 --facial_landmark_model models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009 --gaze_model models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002 -i bin/demo.mp4 -d $1 -o results/ --mouse_speed fast --mouse_precision medium --markers True
