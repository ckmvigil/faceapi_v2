from numpy.core.fromnumeric import shape
import streamlit as st
import pandas as pd
import skimage.io
import cv2
import mediapipe as mp
import math
import base64
import csv
import os
import numpy as np

imageSize = (1024, 1024)

#dictinary for disatnce labels
distance_label = {"jawContourLine2": 1,
                  "jawContourLine3": 2, "ear_ear_distance": 3, "eye_eye_distance": 4, "eyebrow_chin_distance": 5,
                  "mouth_distance": 6, "virtualline_distance": 7, "upperhead_distance": 8, "middlehead_distance": 9,
                  "right_ear_nose_distance": 10, "left_ear_nose_distance": 11
                  }

#dictionary for colours                  
colour_dict = {"1": (245, 30, 30), "2": (0, 128, 0), "3": (255, 255, 51), "4": (102, 51, 255), "5": (255,0,127), 
"6": (0, 255, 128), "7": (0,0,0), "8":(102, 204, 51), "9": (255, 255, 255), "10": (153,0,153), "11":(255, 128, 0)
               }


# function for getting coordinates of the key point
def extractCoordinates(results, landmark_number):
    x = int(
        results.multi_face_landmarks[0].landmark[landmark_number].x * imageSize[0])
    y = int(
        results.multi_face_landmarks[0].landmark[landmark_number].y * imageSize[1])
    return [x, y]


# drawing a circle using OpenCV
def drawCircle(image, location, i=1):
    cv2.circle(image, location, 5, (0, 0, 255), -1)
    # cv2.putText(image,str(i), location, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)


# drawing the lines
def drawPolylines(image, pts):
    cv2.polylines(image, [pts], isClosed=False, color=(0, 255, 0), thickness=1)


# drawing red color polygon
def drawRedPolylines(image, pts):
    cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=1)


# drawing the arrow
def drawArrow(image, start, end):
    cv2.arrowedLine(image, start, end, (0, 110, 0), 2)
    cv2.arrowedLine(image, end, start, (0, 110, 0), 2)


# draw distance label arrow
def drawdistlabel(image, start, end,s):
    cv2.arrowedLine(image, start, end, (colour_dict[s]), 2)
    cv2.arrowedLine(image, end, start, (colour_dict[s]), 2)

# finding the angle trunc

def angle_trunc(a):
    while a < 0.0:
        a += math.pi * 2
    return a

# finding the angle between two points


def findAngle(start, end):
    x1, y1 = start
    x2, y2 = end
    deltaY = y2 - y1
    deltaX = x2 - x1
    return math.degrees(angle_trunc(math.atan2(deltaY, deltaX)))
    
# for calculating z angles
zangle_list=[]
def Zangle(image, start,end,k):
    if(k==2):
        cv2.arrowedLine(image, start, end,(128,0,0), 2)
        cv2.arrowedLine(image, end, start,(128,0,0), 2) 
    if(k==1):
         cv2.arrowedLine(image, start, end,(0,0,255), 2)
         cv2.arrowedLine(image, end, start,(0,0,255), 2)
    if(k==5):
         cv2.arrowedLine(image, start, end,(0,0,0), 2)
         cv2.arrowedLine(image, end, start,(0,0,0), 2)

    zangle_list.append(findAngle(start,end))
   
# for finding chin ratio and lip ratio

chin_ratio2=[]
def chin_ratio1(a,b,c):
    chin_ratio2.append(abs(a[0]-b[0])/abs(a[0]-c[0]))
    
 
# drawing the arrow


def drawArrowRed(image, start, end):
    cv2.arrowedLine(image, start, end, (255, 0, 0), 1)
    cv2.arrowedLine(image, end, start, (255, 0, 0), 1)

# finding distance between two points



def findDistance(start, end):
    x1, y1 = start
    x2, y2 = end
    return (((x2 - x1)**2) + ((y2 - y1)**2))**(1/2)

# finding the distance covered by the list of points


def findDistance_poly(pts):
    jaw_distance = 0
    for i in range(len(pts) - 1):
        jaw_distance += findDistance(pts[i], pts[i+1])
    return jaw_distance
    
# drawing line with the slope and single point


def drawLinePointSlope(image, point, angle, distance):

    # getting the slope of the line
    slope = math.tan(math.radians(angle))

    # current coordinate
    x_coordinate, y_coordinate = point

    # new coordinates
    x_new_coordinate_1 = x_coordinate + (int(distance)/math.sqrt(1+slope**2))
    x_new_coordinate_2 = x_coordinate - (int(distance)/math.sqrt(1+slope**2))
    y_new_coordinate_1 = y_coordinate + slope*(x_new_coordinate_1-x_coordinate)
    y_new_coordinate_2 = y_coordinate + slope*(x_new_coordinate_2-x_coordinate)

    # new coordinate positions
    start = (int(x_new_coordinate_1), int(y_new_coordinate_1))
    end = (int(x_new_coordinate_2), int(y_new_coordinate_2))

    # drawing the lines
    cv2.arrowedLine(image, start, end, (255, 0, 0), 1)
    cv2.arrowedLine(image, end, start, (255, 0, 0), 1)

# Printing text on the line


def addText(image, start, end, s):
    a = distance_label[s]
    a = str(a)
    drawdistlabel(image, start, end,a)
    x_coordinate = int((start[0] + end[0])/(2.4))
    y_coordinate = int(start[1])
    if(s == "eyebrow_chin_distance"):
        x_coordinate = int(start[0])
        y_coordinate = int((start[1] + end[1])/2)
        y_coordinate += 30

    elif(s == "right_ear_nose_distance"):
        y_coordinate = y_coordinate + 30
    cv2.putText(image, a, (x_coordinate, y_coordinate),
                cv2.FONT_HERSHEY_SIMPLEX, 2, colour_dict[a], 2, cv2.LINE_AA)

# The main function which gives us the complete results


def main(image):

    # facemesh configuration
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # running the facemesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,

        # maximum number of faces it should detect
        max_num_faces=3,

        # minimum confidence
            min_detection_confidence=0.3) as face_mesh:

        # update the image size
        global imageSize
        imageSize = (image.shape[1], image.shape[0])

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(image)

    # Keypoints mapping for faster results
    keypoints_mapping = {"right_ear": 234,
                         "left_ear": 454,
                         "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
                         "left_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
                         "jaw_line": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454],
                         "upper_head": [10, 8],
                         "middle_head": [8, 1],
                         "bottom_head": [164, 152],
                         "right_ear_to_nose": [234, 5],
                         "nose_to_left_ear": [5, 454],
                         "right_eyebrow": 107,
                         "left_eyebrow": 336,
                         "right_corner_of_mouth": 61,
                         "left_corner_of_mouth": 291,
                         "upper_lip_to_lower_lip": [0, 17],
                         "right_eye_cord": [159, 145],
                         "left_eye_cord": [386, 374],
                         "head": [10, 1],
                         "vertical_line": [10, 152],
                         "right_ear_to_nose": [234, 5],
                         "left_ear_to_nose": [5, 454],
                         "mouth": [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37],
                         "jawContourLine2": [132, 164, 361],
                         "jawContourLine3": [172, 17, 397],
                         "vertical_line_more": [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
                         }

    # making copies of image for processing
    clone = image.copy()
    keypoints = image.copy()
    distance_mapping = image.copy()
    z_img=image.copy()
    # distance storing dictionary
    distance_dict = {}

    # angle storying dictionary
    angle_dict = {}

    # new distance dictionary
    new_distance_dict = {}

    # printing the key point using the drawcircle function
    for i in range(468):
        location = extractCoordinates(results, i)
        drawCircle(keypoints, location, i)

    # ----------------------------------- The second part --------------------------------------

# the eye keypoint circle
    pts = []
    for item in keypoints_mapping["right_eye"]:
        current = extractCoordinates(results, item)
        pts.append(current)
    pts = np.array(pts)
    drawRedPolylines(clone, pts)
    pts = []
    for item in keypoints_mapping["left_eye"]:
        current = extractCoordinates(results, item)
        pts.append(current)
    pts = np.array(pts)
    drawRedPolylines(clone, pts)

    # the mouth
    pts = []
    for item in keypoints_mapping["mouth"]:
        current = extractCoordinates(results, item)
        pts.append(current)
    pts = np.array(pts)
    drawRedPolylines(clone, pts)

    # jawContour Line 2
    jawContourLine2Point1 = extractCoordinates(
        results, keypoints_mapping["jawContourLine2"][0])
    jawContourLine2Point2 = extractCoordinates(
        results, keypoints_mapping["jawContourLine2"][2])
    drawArrow(clone, jawContourLine2Point1, jawContourLine2Point2)
    addText(distance_mapping, jawContourLine2Point1,
            jawContourLine2Point2, "jawContourLine2")

    jaw_contour_spread_width_2 = findDistance(extractCoordinates(
        results, keypoints_mapping["jawContourLine2"][0]), extractCoordinates(results, keypoints_mapping["jawContourLine2"][2]))
    new_distance_dict["jaw_contour_spread_width_2"] = jaw_contour_spread_width_2

    # jawContour Line 3
    jawContourLine3Point1 = extractCoordinates(
        results, keypoints_mapping["jawContourLine3"][0])
    jawContourLine3Point2 = extractCoordinates(
        results, keypoints_mapping["jawContourLine3"][2])
    drawArrow(clone, jawContourLine3Point1, jawContourLine3Point2)
    addText(distance_mapping, jawContourLine3Point1,
            jawContourLine3Point2, "jawContourLine3")
    jaw_contour_spread_width_3 = findDistance(extractCoordinates(
        results, keypoints_mapping["jawContourLine3"][0]), extractCoordinates(results, keypoints_mapping["jawContourLine3"][2]))
    new_distance_dict["jaw_contour_spread_width_3"] = jaw_contour_spread_width_3

    # left and right inner eye
    drawArrowRed(clone, extractCoordinates(results, keypoints_mapping["left_eye"][0]), extractCoordinates(
        results, keypoints_mapping["right_eye"][8]))
    left_right_inner_eye_width = findDistance(extractCoordinates(
        results, keypoints_mapping["left_eye"][0]), extractCoordinates(results, keypoints_mapping["right_eye"][8]))
    new_distance_dict["left_right_inner_eye_width"] = left_right_inner_eye_width

    # Vertical lines

    # Center Contour1 to chin
    # drawArrowRed(clone, extractCoordinates(results,keypoints_mapping["vertical_line_more"][6]), extractCoordinates(results,keypoints_mapping["vertical_line_more"][27]))

    # nose bottom to chin
    drawArrowRed(clone, extractCoordinates(results, keypoints_mapping["vertical_line_more"][13]), extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][27]))
    nose_bottom_to_chin = findDistance(extractCoordinates(results, keypoints_mapping["vertical_line_more"][13]), extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][27]))
    start1=extractCoordinates(results, keypoints_mapping["vertical_line_more"][10])
    start2=extractCoordinates(results, keypoints_mapping["vertical_line_more"][13])
    end1=extractCoordinates(results, keypoints_mapping["vertical_line_more"][26])
    Zangle(z_img,start1,start2,1)
    Zangle(z_img,start1,end1,2)
    Zangle(z_img,start2,end1,1)
    

    new_distance_dict["nose_bottom_to_chin"] = nose_bottom_to_chin

    # eyebrow center to nose bottom
    drawArrowRed(clone, extractCoordinates(results, keypoints_mapping["vertical_line_more"][2]), extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][13]))
    eyebrow_center_to_nose_bottom = findDistance(extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][2]), extractCoordinates(results, keypoints_mapping["vertical_line_more"][13]))
    new_distance_dict["eyebrow_center_to_nose_bottom"] = eyebrow_center_to_nose_bottom

    # nose bottom to mouth bottom
    drawArrowRed(clone, extractCoordinates(results, keypoints_mapping["vertical_line_more"][13]), extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][22]))
    nose_bottom_to_mouth_bottom = findDistance(extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][13]), extractCoordinates(results, keypoints_mapping["vertical_line_more"][22]))
    new_distance_dict["nose_bottom_to_mouth_bottom"] = nose_bottom_to_mouth_bottom

    # mouth length
    drawArrowRed(clone, extractCoordinates(results, keypoints_mapping["vertical_line_more"][15]), extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][22]))
    mouth_vertical_length = findDistance(extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][15]), extractCoordinates(results, keypoints_mapping["vertical_line_more"][22]))
    new_distance_dict["mouth_vertical_length"] = mouth_vertical_length
    # ------------------------------------------- second part ended -----------------------------------------

    # right and left ear implementations
    right_ear = extractCoordinates(results, keypoints_mapping["right_ear"])
    left_ear = extractCoordinates(results, keypoints_mapping["left_ear"])
    drawCircle(clone, right_ear)
    drawCircle(clone, left_ear)
    drawArrow(clone, right_ear, left_ear)
    addText(distance_mapping, right_ear, left_ear, "ear_ear_distance")
    ear_ear_distance = findDistance(right_ear, left_ear)
    ear_ear_angle = findAngle(right_ear, left_ear)
    distance_dict["ear_to_ear_distance"] = ear_ear_distance
    angle_dict["ear_to_ear_angle"] = ear_ear_angle

    # Contour lines
    drawLinePointSlope(clone, extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][13]), ear_ear_angle, ear_ear_distance*(0.4))
    drawLinePointSlope(clone, extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][22]), ear_ear_angle, ear_ear_distance*(0.3))

    # eye implementations
    right_eye_1 = extractCoordinates(
        results, keypoints_mapping["right_eye"][3])
    right_eye_2 = extractCoordinates(
        results, keypoints_mapping["right_eye"][5])
    right_eye_3 = extractCoordinates(
        results, keypoints_mapping["right_eye"][11])
    right_eye_4 = extractCoordinates(
        results, keypoints_mapping["right_eye"][13])
    left_eye_1 = extractCoordinates(results, keypoints_mapping["left_eye"][3])
    left_eye_2 = extractCoordinates(results, keypoints_mapping["left_eye"][5])
    left_eye_3 = extractCoordinates(results, keypoints_mapping["left_eye"][11])
    left_eye_4 = extractCoordinates(results, keypoints_mapping["left_eye"][13])
    right_eye = [(right_eye_1[0] + right_eye_2[0] + right_eye_3[0] + right_eye_4[0]) //
                 4, (right_eye_1[1] + right_eye_2[1] + right_eye_3[1] + right_eye_4[1])//4]
    left_eye = [(left_eye_1[0] + left_eye_2[0] + left_eye_3[0] + left_eye_4[0]) //
                4, (left_eye_1[1] + left_eye_2[1] + left_eye_3[1] + left_eye_4[1])//4]
    drawCircle(clone, right_eye)
    drawCircle(clone, left_eye)
    drawArrow(clone, right_eye, left_eye)
    addText(distance_mapping, right_eye, left_eye, "eye_eye_distance")
    eye_eye_distance = findDistance(right_eye, left_eye)
    eye_eye_angle = findAngle(right_eye, left_eye)
    distance_dict["eye_to_eye_distance"] = eye_eye_distance
    angle_dict["eye_to_eye_angle"] = eye_eye_angle

    # eyebrow center to chin implementations
    right_eyebrow = extractCoordinates(
        results, keypoints_mapping["right_eyebrow"])
    left_eyebrow = extractCoordinates(
        results, keypoints_mapping["left_eyebrow"])
    chin = extractCoordinates(results, keypoints_mapping["bottom_head"][1])
    center_eyebrow = [(right_eyebrow[0] + left_eyebrow[0]) //
                      2, (right_eyebrow[1] + left_eyebrow[1])//2]
    drawCircle(clone, center_eyebrow)
    drawCircle(clone, chin)
    drawArrow(clone, center_eyebrow, chin)
    addText(distance_mapping, center_eyebrow, chin, "eyebrow_chin_distance")
    drawArrow(z_img,center_eyebrow,[(right_eyebrow[0] + left_eyebrow[0])//2,chin[1]])
    Zangle(z_img,start2,center_eyebrow,5)
    drawArrow(z_img,right_eye_3,[right_eye_3[0],chin[1]])
    chin_ratio1([right_eye_3[0],chin[1]],chin,[(right_eyebrow[0] + left_eyebrow[0])//2,chin[1]])


    eyebrow_chin_distance = findDistance(center_eyebrow, chin)
    eyebrow_chin_angle = findAngle(center_eyebrow, chin)
    distance_dict["eyebrow_to_chin_distance"] = eyebrow_chin_distance
    angle_dict["eyebrow_to_chin_angle"] = eyebrow_chin_angle

    # jawline implementations
    pts = []
    for item in keypoints_mapping["jaw_line"]:
        current = extractCoordinates(results, item)
        pts.append(current)
    pts = np.array(pts)
    drawPolylines(clone, pts)
    jawline_distance = findDistance_poly(pts)
    distance_dict["jawline_distance"] = jawline_distance

    # right mouth cornet to left mouth corner implementations
    right_mouth = extractCoordinates(
        results, keypoints_mapping["right_corner_of_mouth"])
    left_mouth = extractCoordinates(
        results, keypoints_mapping["left_corner_of_mouth"])
    drawCircle(clone, right_mouth)
    drawCircle(clone, left_mouth)
    drawArrow(clone, right_mouth, left_mouth)
    addText(distance_mapping, left_mouth, right_mouth, "mouth_distance")
    right_left_mouth_distance = findDistance(right_mouth, left_mouth)
    right_left_mouth_angle = findAngle(right_mouth, left_mouth)
    distance_dict["right_left_mouth_distance"] = right_left_mouth_distance
    angle_dict["right_left_mouth_angle"] = right_left_mouth_angle

    # virtual line implementations
    y_added = int((5*clone.shape[1])/100)
    x_added = int((10*clone.shape[1])/100)
    right_virtualline = [right_ear[0]-x_added, chin[1]+y_added]
    left_virtualline = [left_ear[0]+x_added, chin[1]+y_added]
    drawCircle(clone, right_virtualline)
    drawCircle(clone, left_virtualline)
    drawArrow(clone, right_virtualline, left_virtualline)
    addText(distance_mapping, left_virtualline,
            right_virtualline, "virtualline_distance")
    virtualline_distance = findDistance(right_virtualline, left_virtualline)
    virtualline_angle = findAngle(right_virtualline, left_virtualline)
    distance_dict["virtualline_distance"] = virtualline_distance
    angle_dict["virtualline_angle"] = virtualline_angle

    # right eyeball implementations
    right_eye_1 = extractCoordinates(
        results, keypoints_mapping["right_eye"][12])
    right_eye_2 = extractCoordinates(
        results, keypoints_mapping["right_eye"][4])
    right_eyeball_horizontal_distance = findDistance(right_eye_1, right_eye_2)
    right_eyeball_horizontal_angle = findAngle(right_eye_1, right_eye_2)
    distance_dict["right_eyeball_horizontal_distance"] = right_eyeball_horizontal_distance
    angle_dict["right_eyeball_horizontal_angle"] = right_eyeball_horizontal_angle
    right_eye_3 = extractCoordinates(
        results, keypoints_mapping["right_eye"][3])
    right_eye_4 = extractCoordinates(
        results, keypoints_mapping["right_eye"][5])
    right_eye_5 = extractCoordinates(
        results, keypoints_mapping["right_eye"][11])
    right_eye_6 = extractCoordinates(
        results, keypoints_mapping["right_eye"][13])
    center_right_eye_1 = [(right_eye_3[0]+right_eye_6[0]) //
                          2, (right_eye_3[1]+right_eye_6[1])//2]
    center_right_eye_2 = [(right_eye_4[0]+right_eye_5[0]) //
                          2, (right_eye_4[1]+right_eye_5[1])//2]
    right_eyeball_vertical_distance = findDistance(
        center_right_eye_1, center_right_eye_2)
    right_eyeball_vertical_angle = findAngle(
        center_right_eye_1, center_right_eye_2)
    distance_dict["right_eyeball_vertical_distance"] = right_eyeball_vertical_distance
    angle_dict["right_eyeball_vertical_angle"] = right_eyeball_vertical_angle

    # left eyeball implementations
    left_eye_1 = extractCoordinates(results, keypoints_mapping["left_eye"][12])
    left_eye_2 = extractCoordinates(results, keypoints_mapping["left_eye"][4])
    left_eyeball_horizontal_distance = findDistance(left_eye_1, left_eye_2)
    left_eyeball_horizontal_angle = findAngle(left_eye_1, left_eye_2)
    distance_dict["left_eyeball_horizontal_distance"] = left_eyeball_horizontal_distance
    angle_dict["left_eyeball_horizontal_angle"] = left_eyeball_horizontal_angle
    left_eye_3 = extractCoordinates(results, keypoints_mapping["left_eye"][3])
    left_eye_4 = extractCoordinates(results, keypoints_mapping["left_eye"][5])
    left_eye_5 = extractCoordinates(results, keypoints_mapping["left_eye"][11])
    left_eye_6 = extractCoordinates(results, keypoints_mapping["left_eye"][13])
    center_left_eye_1 = [(left_eye_3[0]+left_eye_6[0])//2,
                         (left_eye_3[1]+left_eye_6[1])//2]
    center_left_eye_2 = [(left_eye_4[0]+left_eye_5[0])//2,
                         (left_eye_4[1]+left_eye_5[1])//2]
    left_eyeball_vertical_distance = findDistance(
        center_left_eye_1, center_left_eye_2)
    left_eyeball_vertical_angle = findAngle(
        center_left_eye_1, center_left_eye_2)
    distance_dict["left_eyeball_vertical_distance"] = left_eyeball_vertical_distance
    angle_dict["left_eyeball_vertical_angle"] = left_eyeball_vertical_angle

    # upperhead implementations
    upperhead_1 = extractCoordinates(
        results, keypoints_mapping["upper_head"][0])
    upperhead_2 = extractCoordinates(
        results, keypoints_mapping["upper_head"][1])
    upperhead_distance = findDistance(upperhead_1, upperhead_2)
    upperhead_angle = findAngle(upperhead_1, upperhead_2)
    addText(distance_mapping, upperhead_1, upperhead_2, "upperhead_distance")
    distance_dict["upperhead_distance"] = upperhead_distance
    angle_dict["upperhead_angle"] = upperhead_angle

    # middlehead implementations
    middlehead_1 = extractCoordinates(
        results, keypoints_mapping["middle_head"][0])
    middlehead_2 = extractCoordinates(
        results, keypoints_mapping["middle_head"][1])
    middlehead_distance = findDistance(middlehead_1, middlehead_2)
    middlehead_angle = findAngle(middlehead_1, middlehead_2)
    addText(distance_mapping, middlehead_1,middlehead_2, "middlehead_distance")
    
    distance_dict["middlehead_distance"] = middlehead_distance
    angle_dict["middlehead_angle"] = middlehead_angle

    # bottomhead implementations
    bottomhead_1 = extractCoordinates(
        results, keypoints_mapping["bottom_head"][0])
    bottomhead_2 = extractCoordinates(
        results, keypoints_mapping["bottom_head"][1])
    bottomhead_distance = findDistance(bottomhead_1, bottomhead_2)
    bottomhead_angle = findAngle(bottomhead_1, bottomhead_2)
    # addText(distance_mapping,bottomhead_1, bottomhead_2,"bottomhead_distance")
    distance_dict["bottomhead_distance"] = bottomhead_distance
    angle_dict["bottomhead_angle"] = bottomhead_angle

    # right ear to nose implementations
    right_ear = extractCoordinates(
        results, keypoints_mapping["right_ear_to_nose"][0])
    nose = extractCoordinates(
        results, keypoints_mapping["right_ear_to_nose"][1])
    right_ear_nose_distance = findDistance(right_ear, nose)
    
    right_ear_nose_angle = findAngle(right_ear, nose)
    right_nose = extractCoordinates(
        results, keypoints_mapping["right_ear_to_nose"][1])
    rightear_to_nose_distance = findDistance(right_ear, right_nose)
    distance_dict["rightear_to_nose_distance"] = rightear_to_nose_distance
    addText(distance_mapping, right_ear, right_nose, "right_ear_nose_distance")
    Zangle(z_img,right_nose,right_ear,2)
    distance_dict["right_ear_nose_distance"] = right_ear_nose_distance
    angle_dict["right_ear_nose_angle"] = right_ear_nose_angle

    # left ear to nose implementations
    left_ear = extractCoordinates(
        results, keypoints_mapping["nose_to_left_ear"][0])
    nose = extractCoordinates(
        results, keypoints_mapping["nose_to_left_ear"][1])
    left_ear_nose_distance = findDistance(left_ear, nose)
    left_ear_nose_angle = findAngle(left_ear, nose)
    left_nose = extractCoordinates(
        results, keypoints_mapping["left_ear_to_nose"][1])
    leftear_to_nose_distance = findDistance(left_ear, left_nose)
    addText(distance_mapping, left_ear, left_nose, "left_ear_nose_distance")
    #Zangle(z_img,left_ear,left_nose)
    distance_dict["leftear_to_nose_distance"] = leftear_to_nose_distance
    distance_dict["left_ear_nose_distance"] = left_ear_nose_distance
    angle_dict["left_ear_nose_angle"] = left_ear_nose_angle

    # upper lip to lower lip implementations
    top_lip = extractCoordinates(
        results, keypoints_mapping["upper_lip_to_lower_lip"][0])
    bottom_lip = extractCoordinates(
        results, keypoints_mapping["upper_lip_to_lower_lip"][1])
    drawArrow(z_img,bottom_lip,[bottom_lip[0],chin[1]])
    chin_ratio1([right_eye_3[0],chin[1]],chin,[bottom_lip[0],chin[1]])
    lip_to_lip_distance = findDistance(top_lip, bottom_lip)
    lip_to_lip_angle = findAngle(top_lip, bottom_lip)
    distance_dict["lip_to_lip_distance"] = lip_to_lip_distance
    angle_dict["lip_to_lip_angle"] = lip_to_lip_angle

    # right eye implementations
    right_eye_top = extractCoordinates(
        results, keypoints_mapping["right_eye_cord"][0])
    right_eye_bottom = extractCoordinates(
        results, keypoints_mapping["right_eye_cord"][1])
    righteye_to_righteye_distance = findDistance(
        right_eye_top, right_eye_bottom)
    righteye_to_righteye_angle = findAngle(right_eye_top, right_eye_bottom)
    distance_dict["righteye_to_righteye_distance"] = righteye_to_righteye_distance
    angle_dict["righteye_to_righteye_angle"] = righteye_to_righteye_angle

    # left eye implementations
    left_eye_top = extractCoordinates(
        results, keypoints_mapping["left_eye_cord"][0])
    left_eye_bottom = extractCoordinates(
        results, keypoints_mapping["left_eye_cord"][1])
    lefteye_to_lefteye_distance = findDistance(left_eye_top, left_eye_bottom)
    lefteye_to_lefteye_angle = findAngle(left_eye_top, left_eye_bottom)
    distance_dict["lefteye_to_lefteye_distance"] = lefteye_to_lefteye_distance
    angle_dict["lefteye_to_lefteye_angle"] = lefteye_to_lefteye_angle

    # head implementations
    head_1 = extractCoordinates(results, keypoints_mapping["head"][0])
    head_2 = extractCoordinates(results, keypoints_mapping["head"][1])
    head_distance = findDistance(head_1, head_2)
    head_angle = findAngle(head_1, head_2)
    # addText(distance_mapping,head_1,head_2,"head_distance")
    distance_dict["head_distance"] = head_distance
    angle_dict["head_angle"] = head_angle

    # vertical line implementations
    vertical_1 = extractCoordinates(
        results, keypoints_mapping["vertical_line"][0])
    vertical_2 = extractCoordinates(
        results, keypoints_mapping["vertical_line"][1])
    vertical_distance = findDistance(head_1, head_2)
    vertical_angle = findAngle(head_1, head_2)
    distance_dict["vertical_distance"] = vertical_distance
    angle_dict["vertical_angle"] = vertical_angle

    # returning all the results
    return distance_dict, angle_dict, clone, distance_mapping,z_img, keypoints, new_distance_dict

# The main streamlit function

# the side bar
with st.sidebar:
    st.title('CKM VIGIL Face API')
    st.subheader("Facial Landmark Detection")
    # choosing the app mode
    app_mode = st.selectbox("Please select from the following", [
                            "Company Info", "Project Demo", "Project Details"])

# company info mode
if app_mode == "Company Info":
    st.header("CKM VIGIL Pvt. Ltd.")
    st.subheader(
        "The company based out of India working on different aspects of Computer Vision.")
    st.write(
        "For more details visit our website [ckmvigil.in](https://ckmvigil.in/)")

# project demo mode
elif app_mode == "Project Demo":

    st.subheader("Please upload your image")

    # images upload
    image_1 = st.file_uploader("Choose an Image 1", type=[
                               "jpg", "png", "jpeg"], key="image1")
    image_2 = st.file_uploader("Choose an Image 2", type=[
                               "jpg", "png", "jpeg"], key='image2')

    # functions for image 1
    if image_1 is not None:
        image_1 = skimage.io.imread(image_1)

        # optimizing the channels of the image to three channels
        if image_1.shape[2] > 3:
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGBA2RGB)
        if len(image_1.shape) == 2:
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB)

    # functions for image 2
    if image_2 is not None:
        image_2 = skimage.io.imread(image_2)

        # optimizing the channels of the image to three channels
        if image_2.shape[2] > 3:
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGBA2RGB)
        if len(image_2.shape) == 2:
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_GRAY2RGB)

    # option select dropdown for implementing the image results
    #options_selected = st.sidebar.multiselect('Which result you want to get', [
    #                                          "Keypoints", "Characteristics", "Distance", "Angle", "Mouth Position", "Eyes Position", "Head Position", "Face Position", "New Results", "Distance Labels","z angle"])
    options_selected = st.sidebar.multiselect('Which result you want to get', [
                                              "Keypoints", "z angle"])
    # declearing the two columns of the complete screen
    cols = st.columns(2)

    if image_1 is not None:
        cols[0].subheader("Image 1")
        cols[0].image(image_1, caption='Original Image')
    if image_2 is not None:
        cols[1].subheader("Image 2")
        cols[1].image(image_2, caption='Original Image')
    if len(options_selected) != 0:
        st.header("Results")

    # declearing the two columns of the complete screen
    col = st.columns(2)

    # facemesh configuration
    mp_face_detection = mp.solutions.face_detection

    if image_1 is not None:
        # st.image(image_1, caption='Original Image')

        # running the face detection model for getting results
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(image_1)

        # st.write(results.detections)
        # functions for image 1 if the face is founded
        if results.detections:
            distance_dict, angle_dict, clone, distance_mapping, z_img ,keypoints, new_distance_dict = main(
                image_1)
            emotion_image = image_1

            # showing the image with keypoints
            if "Keypoints" in options_selected:
                col[0].image(keypoints, caption='Image 1 with Keypoints')

            # # showing the image with characteristics
            # if "Characteristics" in options_selected:
            #     col[0].image(clone, caption="Characteristics of image 1")

            # # showing the distance graph
            # if "Distance" in options_selected:
            #     st.subheader("The distance between key points in image 1")
            #     st.write("All the distances are in pixel")
            #     df = pd.DataFrame(distance_dict, index=["distance"])
            #     csv=df.to_csv('results_1.csv', mode='a')
            #     df = df.T
            #     st.area_chart(df)
            #     csv = df.to_csv(index=False)

            #showing z angle
    
            if "z angle" in options_selected:
                st.write("All the angles are in degree")      
                st.write("Z angle is the angle between RED LINES")  
                st.write("GREEN LINES are the perpendicular lines from PUPIL and CENTER OF HEAD")            
                col[0].image(z_img, caption="Z angle of image 1")
                st.write("z angle is",zangle_list[4]-zangle_list[1])
                st.write("angle between BLUE LINES is ",zangle_list[2]-zangle_list[0]+180)
                st.write("BLACK LINE is the line between subnasal point and center of head")    
                st.write("the ratio of distance between perpendicular from pupil to chin and perpendicular lines is", chin_ratio2[0])
                st.write("the ratio of distance between perpendicular from pupil to chin and perpendicular from pupil and lower lip is", chin_ratio2[1])
            # # showing the angle graph
            # if "Angle" in options_selected:
            #     st.subheader("The important angles of image 1")
            #     st.write("All the angles are in degree")
            #     df = pd.DataFrame(angle_dict, index=["angle"])
            #     df.to_csv('results_1.csv', mode='a')
            #     df = df.T
            #     st.area_chart(df)

            # # results for mouth opening
            # if ("Mouth Position" in options_selected) and image_2 is None:
            #     st.subheader(
            #         "Please upload the second image also, for getting results.")

            # # results for eye position
            # if ("Eyes Position" in options_selected) and image_2 is None:
            #     st.subheader(
            #         "Please upload the second image also, for getting results.")

            # # results for head position
            # if ("Head Position" in options_selected) and image_2 is None:
            #     st.subheader(
            #         "Please upload the second image also, for getting results.")

            # # results for face position
            # if ("Face Position" in options_selected) and image_2 is None:
            #     st.subheader(
            #         "Please upload the second image also, for getting results.")

            # # showing the new distance graph
            # if "New Results" in options_selected:
            #     st.subheader("The new distances in image 1")
            #     st.write("All the distances are in pixel")
            #     df = pd.DataFrame(new_distance_dict, index=["distance"])
            #     df.to_csv('results_1.csv', mode='a')
            #     df = df.T
            #     st.area_chart(df)
            # if "Distance Labels" in options_selected:
            #     col[0].image(distance_mapping, caption="Distance of image 1")

        # if no face is detected then show the error
        else:
            st.error("No face detected in image 1")

    # second image functions
    if image_2 is not None:

        # getting the face detection results
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(image_2)

        # if image 2 contains a face
        if results.detections:
            distance_dict_1, angle_dict_1, clone, z_img,  distance_mapping_1, keypoints, new_distance_dict_1 = main(
                image_2)
            emotion_image = image_2

            # showing the keypoints
            if "Keypoints" in options_selected:
                col[1].image(keypoints, caption='Image 2 with Keypoints')

            # # showing the characteristics
            # if "Characteristics" in options_selected:
            #     col[1].image(clone, caption="Characteristics of image 2")

            # # showing the distance graph
            # if "Distance" in options_selected:
            #     st.subheader("The distance between key points of image 2")
            #     st.write("All the distances are in pixel")
            #     df = pd.DataFrame(distance_dict_1, index=["distance"])
            #     df = df.T
            #     st.area_chart(df)

            # # showing the angle graph
            # if "Angle" in options_selected:
            #     st.subheader("The important angles of image 2")
            #     st.write("All the angles are in degree")
            #     df = pd.DataFrame(angle_dict, index=["angle"])
            #     df = df.T
            #     st.area_chart(df)

            # results for mouth opening
            # if "Mouth Position" in options_selected and image_1 is not None:
            #     if ((distance_dict["lip_to_lip_distance"]) < (distance_dict_1["lip_to_lip_distance"])):
            #         st.subheader(
            #             "Mouth is open in second image with respect to image 1")
            #     elif((distance_dict["lip_to_lip_distance"]) > (distance_dict_1["lip_to_lip_distance"])):
            #         st.subheader(
            #             "Mouth is closed in second image with respect to image 1")
            #     else:
            #         st.subheader("Error 404: No changes found!")
                    

            # # results for eye position
            # if "Eyes Position" in options_selected and image_1 is not None:
            #     x = (distance_dict["righteye_to_righteye_distance"] +
            #          distance_dict["lefteye_to_lefteye_distance"])
            #     y = distance_dict_1["righteye_to_righteye_distance"] + \
            #         distance_dict_1["lefteye_to_lefteye_distance"]
            #     if (x < y):
            #         st.subheader(
            #             "Eyes are open in second image with respect to image 1")
            #     else:
            #         st.subheader(
            #             "Eyes are closed in second image with respect to image 1")

            # # results for head position
            # if "Head Position" in options_selected and image_1 is not None:
            #     a = (distance_dict["rightear_to_nose_distance"]
            #          ) / (distance_dict["leftear_to_nose_distance"])
            #     b = (distance_dict_1["rightear_to_nose_distance"]
            #          ) / (distance_dict_1["leftear_to_nose_distance"])

            #     if(b > a):
            #         st.subheader(
            #             "Face is turned to left in second image with respect to image 1")
            #     elif(a > b):
            #         st.subheader(
            #             "Face is turned to right in second image with respect to image 1")
            #     else:
            #         st.subheader(
            #             "The head is in same position in second image with respect to image 1")

            # # results for the face position
            # if "Face Position" in options_selected and image_1 is not None:
            #     c = (distance_dict["head_distance"]) / \
            #         (distance_dict["bottomhead_distance"])
            #     d = (distance_dict_1["head_distance"]) / \
            #         (distance_dict_1["bottomhead_distance"])

            #     if(c > d):
            #         st.subheader(
            #             "Face is upwards in second image with respect to image 1")
            #     elif(d > c):
            #         st.subheader(
            #             "Face is downwards in second image with respect to image 1")
            #     else:
            #         st.subheader(
            #             "The face is in same position in second image with respect to image 1")

            #     if(((angle_dict["vertical_angle"])-(angle_dict_1["vertical_angle"])) > 6):
            #         st.subheader(
            #             "The Face has tilted to right in second image with respect to image 1")
            #     elif(((angle_dict["vertical_angle"])-(angle_dict_1["vertical_angle"])) < -6):
            #         st.subheader(
            #             "The Face has tilted to left in second image with respect to image 1")
            #     else:
            #         st.subheader(
            #             "No major change in the tilting of face in second image with respect to image 1")

            # # showing the new distance graph
            # if "New Results" in options_selected:
            #     st.subheader("The new distances in image 2")
            #     st.write("All the distances are in pixel")
            #     df = pd.DataFrame(new_distance_dict_1, index=["distance"])
            #     df = df.T
            #     st.area_chart(df)

            # if "Distance Mapping" in options_selected:
            #     col[1].image(distance_mapping_1, caption='Distance Mapping')

        # if no face is found in image 2 then show the error
        else:
            st.error("No face detected in image 2")

# project details page
elif app_mode == "Project Details":
    st.subheader(
        "CKM VIGIL Face API is solution that estimates 468 3D face landmarks in real-time. It only requires a simple face image.")
    st.write(
        "Check our more projects on [ckmvigil.in/project](https://ckmvigil.in/project)")
