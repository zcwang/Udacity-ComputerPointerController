import cv2
import numpy as np

from model_facilitator import ModelFacilitator

'''
# Inputs: 
1) blob in the format [BxCxHxW] with the name left_eye_image and the shape [1x3x60x60] where:
B - batch size
C - number of channels
H - image height
W - image width
2) blob in the format [BxCxHxW] with the name right_eye_image and the shape [1x3x60x60] where:
B - batch size
C - number of channels
H - image height
W - image width
3) Blob in the format [BxC] with the name head_pose_angles and the shape [1x3] where:
B - batch size
C - number of channels

# Outputs: the net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector. Please note that the output vector is not normalizes and has non-unit length.
1) Output layer name in Inference Engine format:
gaze_vector
2) Output layer name in Caffe2 format:
gaze_vector
'''


class Gaze_Estimation_Model(ModelFacilitator):
    def __init__(self):
        ModelFacilitator.__init__(self)

    def preprocess_output(self, outputs, outputs_names, frame, angles=None):
        gaze_vector = outputs[outputs_names[0]][0]
        angles = angles[0]

        roll = angles[1] * np.pi/180.0
        pitch = angles[0] * np.pi/180.0
        yaw = angles[2] * np.pi/180.0

        mouse_x = gaze_vector[0] * np.cos(roll) + gaze_vector[1] * np.sin(roll)
        mouse_y = -gaze_vector[0] * \
            np.sin(roll) + gaze_vector[1] * np.cos(roll)

        return [gaze_vector, (mouse_x, mouse_y)], frame
