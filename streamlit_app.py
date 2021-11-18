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
distance_label = {"blue":1,"black":2,"red":3,"lime":4,"redd":5}

#dictionary for colours                  
colour_dict = {"1": (0,0,255),"2":(0,0,0),"3":(128,0,0),"4":(0, 110, 0),"5":(128,0,0)}


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

# finding the angle between two points


def findAngle(start, end):
    x1, y1 = start
    x2, y2 = end
    deltaY = y2 - y1
    deltaX = x2 - x1
    return math.degrees(angle_trunc(math.atan2(deltaY, deltaX)))
    

   
# for finding chin ratio and lip ratio

chin_ratio2=[]
def chin_ratio1(a,b,c):
    if(round(abs(a[0]-b[0])/abs(a[0]-c[0]),1)==0.50):
        
        chin_ratio2.append("Tip of chin is at 1/2 times the distance between pupil perpendicular and head perpendicular line")
    if(round(abs(a[0]-b[0])/abs(a[0]-c[0]),1)<0.50):
        f="Tip of chin lies at the left of midpoint of pupil perpendicular and head perpendicular line by- "+ str(round(0.50-round(abs(a[0]-b[0])/abs(a[0]-c[0]),2),2))+" times the distance between the lines"
    
        chin_ratio2.append(f)

    if(round(abs(a[0]-b[0])/abs(a[0]-c[0]),1)>0.50):
        g="Tip of chin lies at the right of midpoint of pupil perpendicular and head perpendicular line by- "+ str(round(round(abs(a[0]-b[0])/abs(a[0]-c[0]),2)-0.50,2))+" times the distance between the lines"
        chin_ratio2.append(g)

def lip_ratio1(a,b,c):
    if(round(abs(a[0]-b[0])/abs(a[0]-c[0]),2)==0.33):

        chin_ratio2.append("-Lower lip line is at 1/3 times the distance between pupil perpendicular and head perpendicular line")
    if(round(abs(a[0]-b[0])/abs(a[0]-c[0]),2)<0.33):
        h="Lower lip line lies at the left of 1/3rd of pupil perpendicular and head perpendicular line point by- "+ str(0.33-round(abs(a[0]-b[0])/abs(a[0]-c[0]),2))+" times the distance between the lines"
        chin_ratio2.append(h)

    if(round(abs(a[0]-b[0])/abs(a[0]-c[0]),2)>0.33):
        i="Lower lip line lies at the right of 1/3rd of pupil perpendicular and head perpendicular line point by- "+ str(round(abs(a[0]-b[0])/abs(a[0]-c[0]),2)-0.33)+" times the distance between the lines"
        chin_ratio2.append(i)

    
 
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
def linedraw(image,start,angle,length):
    
    x=int(round(start[0]+length * math.cos(angle*math.pi/180.00)))
    y=int(round(start[1]+length * math.sin(angle*math.pi/180.00)))
    end=[x,y]
    cv2.line(image, start, end,(128,0,0), 2)



def addText(image, start, end, s):
    a = distance_label[s]
    a = str(a)
  
    x_coordinate = int((start[0] + end[0])/(2.4))
    y_coordinate = int(start[1])
    if(s == "eyebrow_chin_distance"):
        x_coordinate = int(start[0])
        y_coordinate = int((start[1] + end[1])/2)
        y_coordinate += 30
    thickness = int((imageSize[1])*(0.001))
    if thickness == 0:
        thickness += 1

    elif(s == "right_ear_nose_distance"):
        y_coordinate = y_coordinate + 30
    cv2.putText(image, a, (x_coordinate, y_coordinate),
                cv2.FONT_HERSHEY_SIMPLEX, thickness, colour_dict[a], 2, cv2.LINE_AA)

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
            min_detection_confidence=0.1) as face_mesh:

        # update the image size
        global imageSize
        imageSize = (image.shape[1], image.shape[0])

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(image)



    keypoints = image.copy()
    distance_mapping = image.copy()
    z_img=image.copy()
    # distance storing dictionary
    distance_dict = {}


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
  
    pts = []
    for item in keypoints_mapping["left_eye"]:
        current = extractCoordinates(results, item)
        pts.append(current)
    pts = np.array(pts)


    # the mouth
    pts = []
    for item in keypoints_mapping["mouth"]:
        current = extractCoordinates(results, item)
        pts.append(current)
    pts = np.array(pts)



    # left and right inner eye
    
    left_right_inner_eye_width = findDistance(extractCoordinates(
        results, keypoints_mapping["left_eye"][0]), extractCoordinates(results, keypoints_mapping["right_eye"][8]))

    # Vertical lines

    # Center Contour1 to chinaddText
    # drawArrowRed(clone, extractCoordinates(results,keypoints_mapping["vertical_line_more"][6]), extractCoordinates(results,keypoints_mapping["vertical_line_more"][27]))

    # nose bottom to chin
   
    nose_bottom_to_chin = findDistance(extractCoordinates(results, keypoints_mapping["vertical_line_more"][13]), extractCoordinates(
        results, keypoints_mapping["vertical_line_more"][27]))
    start1=extractCoordinates(results, keypoints_mapping["vertical_line_more"][10])
    start2=extractCoordinates(results, keypoints_mapping["vertical_line_more"][13])
    end1=extractCoordinates(results, keypoints_mapping["vertical_line_more"][26])
    Zangle(z_img,start1,start2,1)
    Zangle(z_img,[start1[0],start1[1]],end1,2)
    Zangle(z_img,start2,end1,1)
    addText(z_img,start2,end1,"blue")
    addText(z_img,end1,[start1[0],start1[1]],"redd")
    

# ---------------------------------------- second part ended -----------------------------------------

    # right and left ear implementations
    right_ear = extractCoordinates(results, keypoints_mapping["right_ear"])
    left_ear = extractCoordinates(results, keypoints_mapping["left_ear"])


    ear_ear_distance = findDistance(right_ear, left_ear)
    ear_ear_angle = findAngle(right_ear, left_ear)
  
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

 
    eye_eye_distance = findDistance(right_eye, left_eye)
    eye_eye_angle = findAngle(right_eye, left_eye)

    # eyebrow center to chin implementations
    right_eyebrow = extractCoordinates(
        results, keypoints_mapping["right_eyebrow"])
    left_eyebrow = extractCoordinates(
        results, keypoints_mapping["left_eyebrow"])
    chin = extractCoordinates(results, keypoints_mapping["bottom_head"][1])
    center_eyebrow = [(right_eyebrow[0] + left_eyebrow[0]) //
                      2, (right_eyebrow[1] + left_eyebrow[1])//2]

 
    drawArrow(z_img,center_eyebrow,[(right_eyebrow[0] + left_eyebrow[0])//2,chin[1]])
    Zangle(z_img,start2,center_eyebrow,5)
    addText(z_img,center_eyebrow,start2,"black")
    drawArrow(z_img,right_eye_3,[right_eye_3[0],chin[1]])
    addText(z_img,right_eye_3,[right_eye_3[0],chin[1]],"lime")
    chin_ratio1([right_eye_3[0],chin[1]],chin,[(right_eyebrow[0] + left_eyebrow[0])//2,chin[1]])


    eyebrow_chin_distance = findDistance(center_eyebrow, chin)
    eyebrow_chin_angle = findAngle(center_eyebrow, chin)
    

    # jawline implementations
 

    # right mouth cornet to left mouth corner implementations
    right_mouth = extractCoordinates(
        results, keypoints_mapping["right_corner_of_mouth"])
    left_mouth = extractCoordinates(
        results, keypoints_mapping["left_corner_of_mouth"])


    right_left_mouth_distance = findDistance(right_mouth, left_mouth)
    right_left_mouth_angle = findAngle(right_mouth, left_mouth)


 

    # right eyeball implementations
    right_eye_1 = extractCoordinates(
        results, keypoints_mapping["right_eye"][12])
    right_eye_2 = extractCoordinates(
        results, keypoints_mapping["right_eye"][4])
    right_eyeball_horizontal_distance = findDistance(right_eye_1, right_eye_2)
    right_eyeball_horizontal_angle = findAngle(right_eye_1, right_eye_2)
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

    # left eyeball implementations
    left_eye_1 = extractCoordinates(results, keypoints_mapping["left_eye"][12])
    left_eye_2 = extractCoordinates(results, keypoints_mapping["left_eye"][4])
    left_eyeball_horizontal_distance = findDistance(left_eye_1, left_eye_2)
    left_eyeball_horizontal_angle = findAngle(left_eye_1, left_eye_2)
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
    left_nose = extractCoordinates(
        results, keypoints_mapping["left_ear_to_nose"][1])
   
    r_ear = extractCoordinates(results, keypoints_mapping["right_ear_to_nose"][0])
    l_ear = extractCoordinates(results, keypoints_mapping["nose_to_left_ear"][0])

    m=findDistance(right_nose,r_ear)
    l=findDistance(left_nose,l_ear)
    if(m<=l):
        Zangle(z_img,[left_nose[0],left_nose[1]],l_ear,2)
        addText(z_img,[left_nose[0],left_nose[1]],l_ear,"red")
        linedraw(z_img,left_nose,zangle_list[4],l+(int(imageSize[0]*0.3)))
    else:
        Zangle(z_img,[right_nose[0],right_nose[1]],r_ear,2)
        addText(z_img,[right_nose[0],right_nose[1]],r_ear,"red")
        linedraw(z_img,r_ear,180.00+zangle_list[4],m+(int(imageSize[0]*0.3)))

    # z angle
    yy=findDistance(start1,end1)
    aa=zangle_list[1]+180.00
    linedraw(z_img,end1,aa,yy+(int(imageSize[0]*0.3)))
    
    

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


    # upper lip to lower lip implementations
    top_lip = extractCoordinates(
        results, keypoints_mapping["upper_lip_to_lower_lip"][0])
    bottom_lip = extractCoordinates(
        results, keypoints_mapping["upper_lip_to_lower_lip"][1])
    drawArrow(z_img,bottom_lip,[bottom_lip[0],chin[1]])
    lip_ratio1([right_eye_3[0],chin[1]],[bottom_lip[0],chin[1]],[(right_eyebrow[0] + left_eyebrow[0])//2,chin[1]])
    lip_to_lip_distance = findDistance(top_lip, bottom_lip)
    lip_to_lip_angle = findAngle(top_lip, bottom_lip)

    # right eye implementations
    right_eye_top = extractCoordinates(
        results, keypoints_mapping["right_eye_cord"][0])
    right_eye_bottom = extractCoordinates(
        results, keypoints_mapping["right_eye_cord"][1])
    righteye_to_righteye_distance = findDistance(
        right_eye_top, right_eye_bottom)
    righteye_to_righteye_angle = findAngle(right_eye_top, right_eye_bottom)

    # left eye implementations
    left_eye_top = extractCoordinates(
        results, keypoints_mapping["left_eye_cord"][0])
    left_eye_bottom = extractCoordinates(
        results, keypoints_mapping["left_eye_cord"][1])
    lefteye_to_lefteye_distance = findDistance(left_eye_top, left_eye_bottom)
    lefteye_to_lefteye_angle = findAngle(left_eye_top, left_eye_bottom)

    # head implementations
    head_1 = extractCoordinates(results, keypoints_mapping["head"][0])
    head_2 = extractCoordinates(results, keypoints_mapping["head"][1])

    # vertical line implementations
    vertical_1 = extractCoordinates(
        results, keypoints_mapping["vertical_line"][0])
    vertical_2 = extractCoordinates(
        results, keypoints_mapping["vertical_line"][1])
    vertical_distance = findDistance(head_1, head_2)
    vertical_angle = findAngle(head_1, head_2)

    distance_dict["sub_nasal_point_to_chin_distance"] = findDistance(start2,end1)
    distance_dict["center_eyebrow_to_sub_nasal_point"] = findDistance(center_eyebrow,start2)
    distance_dict["right_ear_to_nose_tip"] = findDistance(right_nose,right_ear)    
    distance_dict["pupil_to_chin"] = findDistance(right_eye_3,[right_eye_3[0],chin[1]])
    distance_dict["nose_tip_to_chin_distance"] = findDistance(start1,end1)


    # returning all the results
    return distance_dict, distance_mapping,z_img, keypoints
# for calculating z angles
zangle_list=[]
def Zangle(image, start,end,k):
   
    if(k==2):
      pass
    if(k==1):
         cv2.line(image, start, end,(0,0,255), 2)
         cv2.line(image, end, start,(0,0,255), 2)
    if(k==5):
         cv2.line(image, start, end,(0,0,0), 2)
         cv2.line(image, end, start,(0,0,0), 2)

    zangle_list.append(round(findAngle(start,end),2))
# The main streamlit function

# the side bar
with st.sidebar:
    st.title('CKM VIGIL Face API')
    st.subheader("Side face Detection")
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
   # image_2 = st.file_uploader("Choose an Image 2", type=[
                            #    "jpg", "png", "jpeg"], key='image2')

    # functions for image 1
    if image_1 is not None:
        image_1 = skimage.io.imread(image_1)

        # optimizing the channels of the image to three channels
        if len(image_1.shape) == 2:
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB)
        if image_1.shape[2] > 3:
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGBA2RGB)
        

    # option select dropdown for implementing the image results
 
    options_selected = st.sidebar.multiselect('Which result you want to get', [
                                              "Keypoints", "Z Angle"])
    # declearing the two columns of the complete screen
    cols = st.columns(2)

    if image_1 is not None:
        cols[0].subheader("Image 1")
        cols[0].image(image_1, caption='Original Image')
  
    if len(options_selected) != 0:
        st.header("Results")

    # declearing the two columns of the complete screen
    col = st.columns(2)

    # facemesh configuration
    mp_face_detection = mp.solutions.face_detection

    if image_1 is not None:
        # st.image(image_1, caption='Original Image')

        # running the face detection model for getting results
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1) as face_detection:
            results = face_detection.process(image_1)

        # st.write(results.detections)
        # functions for image 1 if the face is founded
        if results.detections:
            distance_dict, distance_mapping, z_img ,keypoints = main(
                image_1)
            emotion_image = image_1

            # showing the image with keypoints
            if "Keypoints" in options_selected:
                col[0].image(keypoints, caption='Image 1 with Keypoints')

            #showing Z Angle
            st.write("All the angles are in degree")
            title1 = '<p style="font-family:Courier; color:Red;">Z Angle is the angle between RED LINES</p>'
            title2 = '<p style="font-family:Courier; color:Green;">GREEN LINES are the perpendicular lines from PUPIL, CENTER OF HEAD and LOWER LIP</p>'
            title3 = '<p style="font-family:Courier; color:Gray;">Z Angle is :</p>'
            title4 = '<p style="font-family:Courier; color:Blue;">Angle between BLUE LINES is:</p>'
            title5 = '<p style="font-family:Courier; color:Gray;">BLACK LINE is the line between subnasal point and center of head</p>'
            if "Z Angle" in options_selected:
                #with st.expander("INFORMATION :",expanded=False):

                st.markdown(title1, unsafe_allow_html=True)
                st.markdown(title2, unsafe_allow_html=True)
                st.markdown(title3, unsafe_allow_html=True)
                if(round(zangle_list[4]-zangle_list[1],2)<=90.00):
                    st.info(str(round(zangle_list[4]-zangle_list[1],2)))
                else:
                    st.info(str(round(180.00-zangle_list[4]+zangle_list[1],2)))
                    
                st.write(zangle_list[1])
                st.write(zangle_list[4])

    
        
              
                st.markdown(title4, unsafe_allow_html=True)
                if(round(zangle_list[0]-zangle_list[2]+180.00,2)>=90.00 and round(zangle_list[0]-zangle_list[2]+180.00,2)<=180.00):
                    st.info(str(round(zangle_list[0]-zangle_list[2]+180.00,2)))
                else:
                     st.info(str(round(zangle_list[2]-zangle_list[0]+180.00,2)))
               


                st.markdown(title5, unsafe_allow_html=True)
                st.markdown((chin_ratio2[0]))
                st.markdown(chin_ratio2[1])
                st.write("\n")


                     
                col[0].image(z_img, caption="Z Angle of image 1")
                st.write("Distances")
                key_values = distance_label.values()
                key_list = distance_dict.keys()
                distance_dict_list = distance_dict.values()
                distance_label_list = distance_label.keys()
                label_table = pd.DataFrame(list(zip(key_list,distance_dict_list,distance_label_list,key_values)))
                st.table(label_table)

    
                df = pd.DataFrame(colour_dict, index=["Red","Green", "Blue"])
                st.write("Label Color Codes")
                st.table(df)
                
        # if no face is detected then show the error
        else:
            st.error("No face detected in image 1")

   
# project details page
elif app_mode == "Project Details":
    st.subheader(
        "CKM VIGIL Face API is solution that estimates 468 3D face landmarks in real-time. It only requires a simple face image.")
    st.write(
        "Check our more projects on [ckmvigil.in/project](https://ckmvigil.in/project)")