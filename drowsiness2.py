import cv2 as cv
import mediapipe as mp
import time

import pandas as pd

import utils, math
import numpy as np
import pygame
from pygame import mixer
# from statistics.NormalDist import zscore
import statistics as stats

import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm



# variables
frame_counter = 0
CEF_COUNTER = 0
CMF_COUNTER = 0
MAR_THRESHOLD = 70

# constants
CLOSED_EYES_FRAME = 30
FONTS = cv.FONT_HERSHEY_COMPLEX

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

map_face_mesh = mp.solutions.face_mesh

# camera object
camera = cv.VideoCapture(0)
mixer.init()


# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]

    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

def mouth_aspect_ratio(img, landmarks, up_indices, low_indices):
    A = euclaideanDistance(landmarks[up_indices[6]], landmarks[low_indices[6]])
    B = euclaideanDistance(landmarks[up_indices[11]], landmarks[low_indices[11]])
    C = euclaideanDistance(landmarks[up_indices[13]], landmarks[low_indices[13]])

    MAR = (A + B + C) / 3.0
    return MAR

ear_arr = []
mar_arr = []

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # starting time here
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter += 1  # frame counter
        ret, frame = camera.read()  # getting frame from camera
        if not ret:
            break  # no more frames break

        #  resizing frame
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            MAR = mouth_aspect_ratio(frame, mesh_coords, UPPER_LIPS, LOWER_LIPS)
            if (frame_counter < 300):
                ear_arr.append(ratio)
                mar_arr.append(MAR)


            else:
                
                mar_mu = stats.mean(mar_arr)
                mar_sigma = stats.stdev(mar_arr)

                x_min = min(mar_arr)
                x_max = max(mar_arr)
                x = np.linspace(x_min, x_max, 100)
                plt.plot(x, norm.pdf(x, mar_mu, mar_sigma))
                ptx = mar_mu - 3 * mar_sigma
                pty = mar_mu + 3 * mar_sigma
                ptz = x[x < pty]
                plt.fill_between(ptz, 0, norm.pdf(ptz, mar_mu, mar_sigma), color='#0b559f', alpha=1.0)

                # mar_z = (MAR - mar_mu)/(mar_sigma)
                # ear_z = (ratio - mar_mu)/(ear_sigma)

                plt.text(0, 100, 'Z value: 3', fontsize=20, bbox=dict(facecolor='red', alpha=0.5))
                plt.text(0, 125, 'Mean value: %s'%(mar_mu), fontsize=20, bbox=dict(facecolor='red', alpha=0.5))
                plt.text(0, 150, 'Standard deviation value: %s'%(mar_sigma), fontsize=20, bbox=dict(facecolor='red', alpha=0.5))

                ptz1 = x[x > pty]
                plt.fill_between(ptz1, 0, norm.pdf(ptz1, mar_mu, mar_sigma), color='tab:orange', alpha=1.0)

                # plt.legend(['mean'])
                plt.xlabel("MAR")
                plt.ylabel("Probability Distribution Function Value")
                plt.title("MAR Distribution")


                # plt.fill_between(pty, x_max, color='#2b7bba', alpha='1.0')
                # plt.hist(ear_arr)
                plt.show()

                # ear_z1 = stats.NormalDist(mu=ear_mu, sigma=ear_sigma)
                ear_mu = stats.mean(ear_arr)
                ear_sigma = stats.stdev(ear_arr)

                ear_z = (ratio - ear_mu)/(mar_sigma)

                mar_mu = stats.mean(mar_arr)
                mar_sigma = stats.stdev(mar_arr)

                # mar_z1 = stats.NormalDist(mu=mar_mu, sigma=mar_sigma)
                mar_z = (MAR - mar_mu)/(mar_sigma)

                #print(mar_mu)
                #print(mar_sigma)
                #print(mar_z)

                utils.colorBackgroundText(frame, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                          utils.YELLOW)
                utils.colorBackgroundText(frame, f'MAR : {round(MAR, 2)}', FONTS, 0.7, (30, 150), 2, utils.PINK,
                                          utils.YELLOW)

                if ear_z > 3:
                    CEF_COUNTER += 1

                    if CEF_COUNTER <= 3:
                        utils.colorBackgroundText(frame, f'1', FONTS, 5, (430, 200), 6, utils.WHITE, utils.RED, pad_x=4,
                                                  pad_y=6, )
                    elif CEF_COUNTER <= 20:
                        utils.colorBackgroundText(frame, f'2', FONTS, 5, (430, 200), 6, utils.WHITE, utils.RED, pad_x=4,
                                                  pad_y=6, )
                    elif CEF_COUNTER <= CLOSED_EYES_FRAME:
                        utils.colorBackgroundText(frame, f'3', FONTS, 5, (430, 200), 6, utils.WHITE, utils.RED, pad_x=4,
                                                  pad_y=6, )
                    if CEF_COUNTER > CLOSED_EYES_FRAME:
                        utils.colorBackgroundText(frame, f'Drowsiness Alert...', FONTS, 1.7, (250, 150),
                                                  2, utils.YELLOW, pad_x=4, pad_y=6, )
                        sound = mixer.Sound('warning.wav')
                        sound.play()
                        pygame.time.wait(2000)

                else:
                    CEF_COUNTER = 0

                if mar_z > 3.3:
                    CMF_COUNTER += 1

                    if CMF_COUNTER <= 3:
                        utils.colorBackgroundText(frame, f'1', FONTS, 5, (430, 200), 6, utils.WHITE, utils.RED, pad_x=4,
                                                  pad_y=6, )
                    elif CMF_COUNTER <= 20:
                        utils.colorBackgroundText(frame, f'2', FONTS, 5, (430, 200), 6, utils.WHITE, utils.RED, pad_x=4,
                                                  pad_y=6, )
                    elif CMF_COUNTER <= CLOSED_EYES_FRAME:
                        utils.colorBackgroundText(frame, f'3', FONTS, 5, (430, 200), 6, utils.WHITE, utils.RED, pad_x=4,
                                                  pad_y=6, )
                    if CMF_COUNTER > CLOSED_EYES_FRAME:
                        utils.colorBackgroundText(frame, f'Drowsiness Alert...', FONTS, 1.7, (250, 150),
                                                  2, utils.YELLOW, pad_x=4, pad_y=6, )
                        sound = mixer.Sound('warning.wav')
                        sound.play()
                        pygame.time.wait(2000)
                else:
                    CMF_COUNTER = 0

            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in UPPER_LIPS], dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in LOWER_LIPS], dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)

        # calculating  frame per seconds FPS
        end_time = time.time() - start_time
        fps = frame_counter / end_time

        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                         textThickness=2)

        frame = cv.resize(frame, (1280, 720))
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()

