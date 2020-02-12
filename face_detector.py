from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

def detect_landmarks(gray):
    rects = detector(gray, 1)
    rect = rects[0] if len(rects) > 0 else dlib.rectangle(left=0, top=0, right=gray.shape[0], bottom=gray.shape[1])
    
    shape = predictor(gray, rect)
    return face_utils.shape_to_np(shape).reshape((2,68))

