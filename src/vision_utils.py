import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit
from time import sleep
import cv2.aruco as aruco
import math

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) #import Arcuo Dictionary
parameters =  aruco.DetectorParameters_create()
ext_pixels = 85 #how much to dilate obstacle 
TH_Poly=0.025 #threshold for Polyfit
camera_matrix = np.array([[1007.8578, 0., 627.9454], [0., 1010.2664, 326.2529], [0., 0., 1.]], dtype=np.float32)
distortion = np.array([1.21870288e-01, -7.16272780e-01, -7.01033059e-03, -3.83577419e-04, 9.43910859e-01], dtype=np.float32)

#convert rotation matrix to euler angles
def rotationMatrixToEulerAngles(rvecs):
    '''
    convert rotation matrix to euler angles
    params: rvecs: rotation vector for Arcuo marker
    output: x,y,z: converted euler angle for Arcuo marker in arena space
    '''
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvecs, R)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    #print('dst:', R)
    x = x*180.0/3.141592653589793
    y = y*180.0/3.141592653589793
    z = z*180.0/3.141592653589793
    return x,y,z

def add_alpha_channel(img):
    '''
    add an alpha channel to Thymio icon to decide where to overlay
    params: img, input of img to add alpha channel
    output: img_new: img with all white marked with 0 and others with 1 in the fourth channel
    '''
    b_channel, g_channel, r_channel = cv2.split(img) # split channels
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # create Alpha channels
    alpha_channel[b_channel+g_channel+b_channel==0]=0
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # merge channels
    return img_new
def rotate_image(image, angle):
    '''
    rotate Thymio icon according to angle
    params: image, input of Thymio icon
    params: angle, rotation of Thymio
    output: result: rotated Thymio image
    '''
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 0.5)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

#merge Thymio icon and annotated map
def merge_img(map_img, thymio_img, y1, y2, x1, x2):
    '''function to overlay thymio_icon onto the graph
    jpg_img is thymio icon i'''
    if map_img.shape[2] == 3:
        map_img = add_alpha_channel(map_img)
    if thymio_img.shape[2] == 3:
        thymio_img = add_alpha_channel(thymio_img)
    yy1 = 0
    yy2 = thymio_img.shape[0]
    xx1 = 0
    xx2 = thymio_img.shape[1]
 
    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > map_img.shape[1]:
        xx2 = thymio_img.shape[1] - (x2 - map_img.shape[1])
        x2 = map_img.shape[1]
    if y2 > map_img.shape[0]:
        yy2 = thymio_img.shape[0] - (y2 - map_img.shape[0])
        y2 = map_img.shape[0]
        
    alpha_thymio = thymio_img[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_map = 1 - alpha_thymio
    for c in range(0,3):
        map_img[y1:y2, x1:x2, c] = ((alpha_map*map_img[y1:y2,x1:x2,c]) + (alpha_thymio*thymio_img[yy1:yy2,xx1:xx2,c]))
    return map_img

#get info for Thymio start position, Thymio direction, Thymio target, list of obstacle corners, occupancy grid, abstracted obstacle map
#details are described in the report
def localisation_cam_all(frame):
    '''
    function for taking current frame and returning all information for gloval navigation as output
    params: frame, captured frame, 720*1280*3 uint8 RGB image
    output: Thymio_center: Thymio start position, list of [x,y]
    output: Thymio_target: Thymio target, list of [x,y]
    output: obs_corners: list of obstacle corners, list of list of obstacle vertices, 
    output: obstacles_mask: occupancy grid, 720*1280 grids with occupancy. 0 is free and 1 is occupied
    output: img_out: abstracted obstacle map, 720*1280*3 uint8 RGB image with obstacles with red and all others white
    '''
    red_lower = np.array([140, 100, 160])
    red_upper = np.array([185, 180, 255])
    # red_lower = np.array([140, 120, 150])
    # red_upper = np.array([180, 180, 210])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
    corner_points = []  
    Thymio_start=[-1,-1]
    Thymio_target=[-1,-1]
    Thymio_dir = -1
    obs_corners = []
    obstacles_mask=[]
    warpedimg=[]
    img_out=np.ones((720,1280,3),np.uint8)*255
    cam_OK = True
    try:
        for i in range(1,5):  
            corner_points.append(corners[ids.tolist().index([i])][0][0].tolist())
        pts1 = np.float32(corner_points)
        pts2 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
        transform = cv2.getPerspectiveTransform(pts1, pts2)
        warpedimg = cv2.warpPerspective(frame, transform, (1280, 720))
        corners, ids, _ = aruco.detectMarkers(warpedimg,aruco_dict,parameters=parameters)
        Thymio_corner = corners[ids.tolist().index([0])]
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(Thymio_corner, 0.05, camera_matrix, distortion)
        Thymio_dir = -rotationMatrixToEulerAngles(rvec[0])[2]+180
        Thymio_center = np.mean(Thymio_corner, axis=1).tolist()[0]
        Thymio_center.reverse()
        Thymio_target = np.mean(corners[ids.tolist().index([5])], axis=1).tolist()[0]
        Thymio_target.reverse()
        HSV = cv2.cvtColor(warpedimg, cv2.COLOR_RGB2HSV)
        HSV_blur = cv2.GaussianBlur(HSV, (7, 7), 0)
        red_mask=cv2.inRange(HSV_blur,red_lower,red_upper)
        red_mask_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8),iterations=2)
        obstacles_mask = cv2.morphologyEx(red_mask_closed, cv2.MORPH_OPEN, np.ones((10,10), np.uint8),iterations=2)
        obstacles_mask = cv2.dilate(obstacles_mask, np.ones((int(ext_pixels*2-1),int(ext_pixels*2-1)), np.uint8), iterations=1)
        contours, _ = cv2.findContours(obstacles_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 1800:#drop spots
                temp = []
                epsilon = TH_Poly*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                for point in approx:
                    point_original = list(point[0])
                    temp.append([point_original[1],point_original[0]])
                obs_corners.append(temp)


        obstacles_mask = cv2.morphologyEx(red_mask_closed, cv2.MORPH_OPEN, np.ones((10,10), np.uint8),iterations=2)
        obstacles_mask[obstacles_mask==255]=1
        _, mask = cv2.threshold(red_mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
        img_out[mask==255] = [200,20,20]
        Thymio_start = Thymio_center
    except Exception as e:
        cam_OK=False
        print(e)
    return cam_OK, Thymio_start, Thymio_dir, Thymio_target, obs_corners, obstacles_mask, img_out

def localisation_cam(frame):
    '''
    function for taking current frame and returning all information for gloval navigation as output
    params: frame, captured frame, 720*1280*3 uint8 RGB image
    output: Thymio_center: Thymio start position, list of [x,y]
    output: Thymio_target: Thymio target, list of [x,y]
    output: warpedimg: perspectively transformed 720*1280*3 uint8 RGB image for annotation purposes
    '''
    # red_lower = np.array([140, 120, 150])
    # red_upper = np.array([185, 180, 210])
    # ret, frame = cap.read()
    # while(not ret):
    #     cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #     cap.set(cv2.CAP_PROP_EXPOSURE, -4)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
    corner_points = []  
    #marker1=topleft   marker2=topright marker3=bottomleft marker4=bottomright
    Thymio_center=[-1,-1]
    Thymio_dir = -1
    warpedimg=[]
    cam_OK = True
    try:
        for i in range(1,5):  
            corner_points.append(corners[ids.tolist().index([i])][0][0].tolist())
        pts1 = np.float32(corner_points)
        pts2 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
        transform = cv2.getPerspectiveTransform(pts1, pts2)
        warpedimg = cv2.warpPerspective(frame, transform, (1280, 720))
        corners, ids, _ = aruco.detectMarkers(warpedimg,aruco_dict,parameters=parameters)
        Thymio_corner = corners[ids.tolist().index([0])]
        rvec, _, _ = aruco.estimatePoseSingleMarkers(Thymio_corner, 0.05, camera_matrix, distortion)
        Thymio_dir = -rotationMatrixToEulerAngles(rvec[0])[2]+180
        Thymio_center = np.mean(Thymio_corner, axis=1).tolist()[0]
        # Thymio_center = [int(Thymio_center[0]),int(Thymio_center[1])]
        Thymio_center.reverse()
    except:
        cam_OK=False
    return cam_OK, Thymio_center, Thymio_dir, warpedimg

def swap(obs):
    '''
    swap x and y for matplotlib plotting
    params: obs, obstacle vertices (y,x)
    params: obs_coords_swap, swapped obstacle vertices (x,y)
    '''
    obs_coords_swap = []
    for i in range(len(obs)):
        obs_coords_swap.append(np.copy(obs[i]))
        obs_coords_swap[i][:, [1, 0]] = obs_coords_swap[i][:, [0, 1]]
    return obs_coords_swap

def obs_corners_point(obs_corners_swap):
    '''
    convert swapped corners to list
    params: obs_corners_swap: swapped obstacle corner in (x,y)
    output: point_list: obstacle vertices in point list
    '''
    tmp = []
    for i in obs_corners_swap:
        tmp.append(i.tolist())
    point_list = [tuple(item) for sublist in tmp for item in sublist]
    return point_list

def overlay_vis_graph(Thymio_start, Thymio_center, Thymio_target, obs_corners, optimal_path, warpedimg_clean):
    '''
    overlay the Thymio and path onto abstract map or photo
    params: Thymio_start: start position of Thymio
    params: Thymio_center: current position of Thymio
    params: Thymio_target: target of Thymio position
    params: obs_corners: obstacle boundary vertices
    params: optimal_path: optimal path from global planning
    params: warpedimg_clean: abstract map or perspectively transferred image for overlay plotting
    '''
    obs_corners_swap = swap(obs_corners)
    point_list = obs_corners_point(obs_corners_swap)
    warpedimg_clean_cpy = warpedimg_clean.copy()
    cv2.polylines(warpedimg_clean_cpy, obs_corners_swap, isClosed=True, color=(0, 114, 189), thickness=8)
    for point in point_list:
        cv2.circle(warpedimg_clean_cpy, point, 5, color=(217, 83, 25), thickness=10)
    cv2.circle(warpedimg_clean_cpy, tuple([Thymio_start[1],Thymio_start[0]]), 5, color=(126, 47, 142), thickness=12)
    cv2.circle(warpedimg_clean_cpy, tuple([Thymio_center[1],Thymio_center[0]]), 5, color=(126, 47, 142), thickness=16)
    cv2.circle(warpedimg_clean_cpy, tuple([Thymio_target[1],Thymio_target[0]]), 5, color=(126, 47, 142), thickness=12)
    for i in range(len(optimal_path)-1):
        cv2.line(warpedimg_clean_cpy,tuple([int(optimal_path[i][1][1]),int(optimal_path[i][1][0])]),
                tuple([int(optimal_path[i+1][1][1]),int(optimal_path[i+1][1][0])]),
                color=(119, 172, 48),
                thickness=12)
    cv2.putText(warpedimg_clean_cpy, 'Start', tuple([Thymio_start[1],Thymio_start[0]]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color = (0,0,0),thickness = 7)
    cv2.putText(warpedimg_clean_cpy, 'Robot', tuple([Thymio_center[1],Thymio_center[0]]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color = (0,0,0),thickness = 7)
    cv2.putText(warpedimg_clean_cpy, 'Goal', tuple([Thymio_target[1],Thymio_target[0]]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color = (0,0,0),thickness = 7)
    return(warpedimg_clean_cpy)