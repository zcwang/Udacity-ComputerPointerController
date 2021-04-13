import cv2
import numpy as np

from model_facilitator import ModelFacilitator

'''
Process images based on the openvino's head-pose-estimation-adas-0001 model, 
# Input: "data" , shape: [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR.
# Outputs: layer names in Inference Engine format:
name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
# BTW Output layer names in Caffe* format:
name: "fc_y", shape: [1, 1] - Estimated yaw (in degrees).
name: "fc_p", shape: [1, 1] - Estimated pitch (in degrees).
name: "fc_r", shape: [1, 1] - Estimated roll (in degrees).
# Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitch or roll).
'''


def RotateMat(gamma, beta, alpha):
    '''
    Calculate a rotation matrix among roll, pitch and yaw angles
    # inputs:
        gamma --> angle that represents the roll angle
        beta  --> angle that represents the pitch angle
        alpha --> angle that represents the yaw angle
    # output
        R --> Calculated rotation matrix
    '''
    # Converting the rotation angles to radians
    gamma = gamma * np.pi/180.0
    beta = beta * np.pi/180.0
    alpha = alpha * np.pi/180.0

    # Creating the matrices that will contains the Rotation matrices components
    Rz = np.zeros(shape=(3, 3), dtype=float)
    Ry = np.zeros(shape=(3, 3), dtype=float)
    Rx = np.zeros(shape=(3, 3), dtype=float)

    # Calculating the raotion matrix parts
    Rz[0][0] = np.cos(alpha)
    Rz[0][1] = -np.sin(alpha)
    Rz[1][0] = np.sin(alpha)
    Rz[1][1] = np.cos(alpha)
    Rz[2][2] = 1
    Ry[0][0] = np.cos(beta)
    Ry[0][2] = np.sin(beta)
    Ry[1][1] = 1
    Ry[2][0] = -np.sin(beta)
    Ry[2][2] = np.cos(beta)
    Rx[0][0] = 1
    Rx[1][1] = np.cos(gamma)
    Rx[1][2] = -np.sin(gamma)
    Rx[2][1] = np.sin(gamma)
    Rx[2][2] = np.cos(gamma)

    return np.matmul(np.matmul(Rz, Ry), Rx)


class HeadPose_Estimation_Model(ModelFacilitator):
    def __init__(self):
        ModelFacilitator.__init__(self)

    def draw_outputs(self, angles, frame):
        vector_length = 50
        Origin = [frame.shape[1]/2, frame.shape[0]/2]
        R = RotateMat(angles[0][2], angles[0][0], angles[0][1])

        Vectors = [np.array([vector_length, 0, 0]),
                   np.array([0, -vector_length, 0]),
                   np.array([0, 0, -vector_length])]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        counter = 0
        for Vector, color in zip(Vectors, colors):
            V = np.matmul(R, Vector.transpose())
            if counter == 2:
                large_axisx = -int(V[0])+int(Origin[0])
                large_axisy = -int(V[1])+int(Origin[1])
            else:
                large_axisx = int(Origin[0])
                large_axisy = int(Origin[1])
            cv2.line(frame, (large_axisx, large_axisy),
                     (int(V[0])+int(Origin[0]), int(V[1])+int(Origin[1])), color, 3)
            counter += 1

    def preprocess_output(self, outputs, outputs_names, frame, angles=None):
        results = [outputs[feat][0][0] for feat in outputs_names]

        return np.resize(np.array(results), (1, 3)), frame[0]
