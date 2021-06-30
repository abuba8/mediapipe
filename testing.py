import cv2
import mediapipe as mp 
import time
import hand_tracking_module as htm
import facial_landmarks as lm
#import face_detection as fd

cap = cv2.VideoCapture(0)
detector = htm.handDetector()
lm_detector = lm.faceLandmarks()
#fd_detector = fd.faceDetection()

while True:
	success, img = cap.read()
	img = detector.findHands(img)
	lmlist = detector.findPosition(img, draw = False)
	img = lm_detector.findFace(img)
	#img = fd.detector.detectFace(img)
	

	cv2.imshow("Image", img)
	cv2.waitKey(1)