import numpy as np
import cv2
from math import cos, sin, acos
from scipy.spatial.transform import Rotation as R
 
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    # convert to radians
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
    
def crop_face(img, pt2d):
    x_min = min(pt2d[0, :])
    y_min = min(pt2d[1, :])
    x_max = max(pt2d[0, :])
    y_max = max(pt2d[1, :])
    
    Lx = abs(x_max - x_min)
    Ly = abs(y_max - y_min)
    Lmax = max(Lx, Ly) * 1.5
    center_x = x_min + Lx // 2
    center_y = y_min + Ly // 2
    
    x_min = center_x - Lmax // 3
    x_max = center_x + Lmax // 3
    y_min = center_y - Lmax // 2
    y_max = center_y + Lmax // 3
    
    if x_min < 0:
        y_max -= abs(x_min)
        x_min = 0
    if y_min < 0:
        x_max -= abs(y_min)
        y_min = 0
    if x_max > img.shape[1]:
        y_min += abs(x_max - img.shape[1])
        x_max = img.shape[1]
    if y_max > img.shape[0]:
        x_min += abs(y_max - img.shape[0])
        y_max = img.shape[0]
    
    return img[int(y_min):int(y_max), int(x_min):int(x_max)]

def convert_ypr_to_rvec(yaw, pitch, roll):
    r = R.from_euler('xyz',[pitch, yaw, roll], degrees=True)
    rotvec = r.as_rotvec()
    return rotvec[0], rotvec[1], rotvec[2]