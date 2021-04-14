import numpy as np
import pandas as pd
import logging as log
import time
import os
import sys
import argparse
import cv2

from argparse import ArgumentParser
from openvino.inference_engine import IECore

from input_feeder import InputFeeder
from mouse_controller import MouseController

from face_detection import Face_Detection_Model
from head_pose_estimation import HeadPose_Estimation_Model
from facial_landmark_detection import FacialLandMark_Detection_Model
from gaze_estimation import Gaze_Estimation_Model


def build_argparser():
    parser = ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Defining required arguments
    required.add_argument("-fm", "--face_model", required=True, type=str,
                          help="Path to the model that will be used to detect faces on specific targets.")
    required.add_argument("-hm", "--head_model", required=True, type=str,
                          help="Path to the model that will be used to detect head positions on specific targets.")
    required.add_argument("-fldm", "--facial_landmark_model", required=True, type=str,
                          help="Path to the model that will be used to detect eyes landmarks on specific targets.")
    required.add_argument("-gzm", "--gaze_model", required=True, type=str,
                          help="Path to the model that will be used to detect faces on specific targets.")
    required.add_argument("-i", "--input", required=True, type=str,
                          help="Path to image or video file input whewre targets of interest are present.")
    required.add_argument("-o", "--output", required=True, type=str,
                          help="Path to save the results obtained.")
    # Defining optional arguments
    optional.add_argument("-spd", "--mouse_speed", required=False, type=str, default='medium',
                          help="Speed to which the mouse cursor move per instruction, "
                          "valid options are: slow, medium, fast.")
    optional.add_argument("-prcs", "--mouse_precision", required=False, type=str, default='medium',
                          help="Amount of precision for the mouse movements, "
                          "valid options are: low, medium, high.")
    optional.add_argument("-l", "--cpu_extension", required=False, type=str,
                          default=None,
                          help="Path to layer extension in case it is incompatible with the device architecture.")
    optional.add_argument("-d", "--device", type=str, required=False, default="CPU_CPU_CPU_CPU",
                          help="Specify the target device to infer on: "
                          "CPU, GPU, FPGA or MYRIAD is acceptable. "
                          "In this project, since four models are needed, the devices used shold be "
                          "defines such as: Device1_Device2_Device3_Device4.")
    optional.add_argument("-pf", "--thresh", type=float, required=False, default=0.6,
                          help="Probability threshold for detections filtering"
                          "(0.6 by default)")
    optional.add_argument("-mks", "--markers", type=bool, default=False,
                          help='Flag used to display markers that identify the models used outputs'
                          'such as faces, eyes landmarks and head position angles. '
                          "valid flag values are False and True.")
    args = parser.parse_args()
    return parser


def capture_stream(args):
    # input formats
    valid_exts = {'video': ['.mp4'], 'imgs': ['.jpg', '.bmp']}

    # to check source among image, video, or webcam as input
    if args.input == 'cam':
        input_type = 'cam'
    elif args.input[-4:] in valid_exts['video']:
        input_type = 'video'
    elif args.input[-4:] in valid_exts['imgs']:
        input_type = 'image'

    # only to use valid input file as selection
    if (args.input[-4:] not in valid_exts['video']) and (args.input[-4:] not in valid_exts['imgs']) and (args.input != 'cam'):
        log_object.error(
            "The selected input is no valid to be used on this program, try a different one.")
        sys.exit()

    return input_type


def doInferecneForInuptstream(args):
    # user-defined parameters for prediction
    devices = args.device.split('_')
    models = [args.face_model, args.head_model,
              args.facial_landmark_model, args.gaze_model]
    marker = args.markers
    log.basicConfig(level=log.DEBUG)
    log_object = log.getLogger()

    # Initialization for models needed for the inference
    face_network = Face_Detection_Model()
    head_network = HeadPose_Estimation_Model()
    landmarks_network = FacialLandMark_Detection_Model()
    gaze_network = Gaze_Estimation_Model()

    # use MC as controller of mouse (by pyautogui)
    MC = MouseController(args.mouse_precision, args.mouse_speed)

    # Loading the models and save their loading time for statistics
    loading_times = []
    load_time = time.time()
    face_network.load_model(
        models[0], devices[0], args.cpu_extension, marker, args.thresh)
    loading_times.append(time.time() - load_time)
    load_time = time.time()
    head_network.load_model(models[1], devices[1], args.cpu_extension, marker)
    loading_times.append(time.time() - load_time)
    load_time = time.time()
    landmarks_network.load_model(
        models[2], devices[2], args.cpu_extension, marker)
    loading_times.append(time.time() - load_time)
    load_time = time.time()
    gaze_network.load_model(models[3], devices[3])
    loading_times.append(time.time() - load_time)

    # input stream
    input_type = capture_stream(args)

    # new wrapper with input stream for pre-defined InputFeeder class
    feed = InputFeeder(input_type=input_type, input_file=args.input)

    # Load the video input and defining variables needed for the project statistics
    # and the flow of predictions per frame.
    feed.load_data()
    frame_count = 0
    counter = 0
    start_input_output_time = time.time()

    # Reading frame from the used defined input
    for flag, frame in feed.next_batch():
        if not flag:
            log_object.info("No detected frames as input source.")
            break

        # Block that is repeated in the next part of the code ----------------------
        # Block needed to measure the inferece times for the project statistics
        if counter == 0:
            inference_times = []
            inference_time = time.time()
            boxes, face_frame = face_network.predict([frame])
            inference_times.append(time.time() - inference_time)

            # Monitoring empty inferences to avoid errors
            if len(boxes) == 0:
                log_object.warning("No face detected.")
                continue
            else:
                # OpenVino selected models inferences and their times
                inference_time = time.time()
                head_pose_angles, _ = head_network.predict([face_frame])
                inference_times.append(time.time() - inference_time)
                inference_time = time.time()
                eyes_boxes, frames_eyes = landmarks_network.predict([
                                                                    face_frame])
                inference_times.append(time.time() - inference_time)
                inference_time = time.time()
                mouse_ctrl, _ = gaze_network.predict(
                    frames_eyes, head_pose_angles, 1)
                inference_times.append(time.time() - inference_time)
        # ---------------------------------------------------------------------------

        # OpenCV waiting
        key_pressed = cv2.waitKey(60)

        # Running inference on the face_detector model
        boxes, face_frame = face_network.predict([frame])

        # Monitoring empty inferences to avoid errors
        if len(boxes) == 0:
            log_object.warning("No face detected.")
            continue
        else:
            frame_count += 1

        # Openvino selected model inferences
        head_pose_angles, _ = head_network.predict([face_frame])
        eyes_boxes, frames_eyes = landmarks_network.predict([face_frame])
        mouse_ctrl, _ = gaze_network.predict(frames_eyes, head_pose_angles, 1)
        gaze_vector, mouse_pos = mouse_ctrl

        # Displaying intermediate results, if required by the user
        if marker == True:
            face_network.draw_outputs(boxes, frame)
            head_network.draw_outputs(head_pose_angles, face_frame)
            landmarks_network.draw_outputs(eyes_boxes, frame, boxes)

        # Move the mouse without any delay -- in review
        #MC.move(mouse_pos[0], mouse_pos[1])

        # Controlling the mouse with the obtained inference
        if frame_count % 5 == 0:
            MC.move(mouse_pos[0], mouse_pos[1])

        # User exit
        if key_pressed == 27:
            break

        # Estimating the input-output time for the project statistics and
        # ending the gathering of statistics
        if counter == 0:
            end_input_output_times = time.time() - start_input_output_time
            counter = 1

       # Displaying the gaze tracking inference
        image = np.uint8(frame)
        cv2.imshow('Visualization', cv2.resize(image, (960, 540)))

    # Releasing the capture
    feed.close()

    # Destroying any OpenCV windows created
    cv2.destroyAllWindows

    # Builing the project statistics
    Precisions = [model.split('/')[3] for model in models]

    dict_details = {'Model': ['Face_Detection', 'Head_Pose', 'Facial_Landmark', 'Gaze_Estimation'],
                    'Precision': Precisions, 'Device': devices, 'Loading Time': loading_times,
                    'Inference Time': inference_times}

    dict_summary = {'Performance Metric': ['Loading Time', 'Inference Time', 'Input-Output Time'],
                    'Seconds': [sum(loading_times), sum(inference_times), end_input_output_times/counter]}

    # Saving the project statistics results as csv files
    pd.DataFrame(dict_summary).to_csv(os.path.join(os.getcwd(), 'results',
                                                   args.output.split('/')[1] + '_summary.csv'))
    pd.DataFrame(dict_details).to_csv(os.path.join(os.getcwd(), 'results',
                                                   args.output.split('/')[1] + '_details.csv'))


def main():
    args = build_argparser().parse_args()

    doInferecneForInuptstream(args)


if __name__ == '__main__':
    main()
