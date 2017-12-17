#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2017 Vikram Voleti
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Eg.: python3 ex_cnn_head_pose_estimation_images_list a.txt -npy "poses"

import csv
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import time
import tqdm

from deepgaze.head_pose_estimation import CnnHeadPoseEstimator


def print_time_till_now(start_time):
    print("")
    ret = os.system("date")
    till_now = time.time() - start_time
    h = till_now//3600
    m = (till_now - h*3600)//60
    s = (till_now - h*3600 - m*60)//1
    print(h, "hr", m, "min", s, "sec")
    print("")


def load_poses_from_csv(csv_file_name):
    with open(csv_file_name, 'r') as f:  #opens PW file
        reader = csv.reader(f)
        poses = list(list([float(i) for i in rec]) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        return poses


def save_poses_in_csv(poses, csv_file_name="poses"):
    with open(csv_file_name + ".csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(poses)


def cnn_head_pose_estimation_images_list(inputFile, verbose=False, save_npy=True, npy_file_name="poses", save_csv=False, csv_file_name="poses"):

    return_val = 0

    # Read jpg file names
    if '.txt' in inputFile:
        try:
            with open(inputFile) as f:
                image_file_names = f.read().splitlines()
        except FileNotFoundError:
            print("\nERROR: File not found! Tried reading ", sys.argv[1], "\n")
            return 1
    # Image
    elif '.jpg' in inputFile or '.png' in inputFile:
        image_file_names = [inputFile]
    # Else
    else:
        raise ValueError("\n\n[ERROR] File type not understood! " + inputFile + "\n\n")

    # Tensorflow session
    sess = tf.Session() #Launch the graph in a session.
    my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object

    # Load the weights from the configuration folders
    DEEPGAZE_EXAMPLES_DIR = os.path.dirname(os.path.realpath(__file__))
    my_head_pose_estimator.load_roll_variables(os.path.realpath(os.path.join(DEEPGAZE_EXAMPLES_DIR, "../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf")))
    my_head_pose_estimator.load_pitch_variables(os.path.realpath(os.path.join(DEEPGAZE_EXAMPLES_DIR, "../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf")))
    my_head_pose_estimator.load_yaw_variables(os.path.realpath(os.path.join(DEEPGAZE_EXAMPLES_DIR, "../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k")))

    start_time = time.time()

    if save_npy:
        poses = np.empty((0, 3))
    elif save_csv:
        poses = []

    try:
        for i, image_file in tqdm.tqdm(enumerate(image_file_names), total=(len(image_file_names))):
            if verbose:
                tqdm.tqdm.write("Processing image " + image_file)
            elif save_csv:
                poses.append([])
            #Read the image with OpenCV
            image = cv2.imread(image_file)
            # Get the angles for roll, pitch and yaw
            roll = my_head_pose_estimator.return_roll(image)  # Evaluate the roll angle using a CNN
            pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
            yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
            if verbose:
                tqdm.tqdm.write("Estimated [roll, pitch, yaw] : [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
            elif save_npy:
                poses = np.vstack((poses, [(roll[0,0,0])/25, pitch[0,0,0]/45, yaw[0,0,0]/100]))
            elif save_csv:
                poses[-1].append(roll[0,0,0]/25)
                poses[-1].append(pitch[0,0,0]/45)
                poses[-1].append(yaw[0,0,0]/100)

    except KeyboardInterrupt:
        print("\n\nCtrl+C was pressed!\n\n")
        return_val = 1


    if save_npy:
        print("Saving in npy file", npy_file_name, "....")
        np.save(npy_file_name, poses)
        print("Saved.")
    elif save_csv:
        print("Saving in csv file", csv_file_name, "....")
        save_poses_in_csv(poses, csv_file_name=csv_file_name)
        print("Saved.")

    print_time_till_now(start_time)

    return return_val


# MAIN
if __name__ == "__main__":

    # INIT

    verbose = True
    save_npy = False
    npy_file_name = "poses"
    save_csv = False
    csv_file_name = "poses"

    # READ ARGUMENTS

    if len(sys.argv) < 2:
        print("[ERROR] Please mention jpg/png image, or .txt file with list of images as ARGUMENT.")
        sys.exit()

    inputFile = sys.argv[1]

    if len(sys.argv) > 2:
        if sys.argv[2] == '--save_npy' or sys.argv[2] == '-npy':
            save_npy = True
            verbose = False

        elif sys.argv[2] == '--save_csv' or sys.argv[2] == '-csv':
            save_csv = True
            verbose = False
            csv_file_name = "poses"

    if len(sys.argv) > 3:
        if save_npy:
            npy_file_name = sys.argv[3]
        elif save_csv:
            csv_file_name = sys.argv[3]

    # RUN

    try:
        return_val = cnn_head_pose_estimation_images_list(inputFile, verbose, save_npy, npy_file_name, save_csv, csv_file_name)

    except ValueError as err:
        print(err)

