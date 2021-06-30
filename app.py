import cv2
import mediapipe as mp 
import time

cap = cv2.VideoCapture(0)

#necessary
mp_hands = mp.solutions.hands 
#instance, default params static = false, tracking and detection 0.5, no of hands 2
hands = mp_hands.Hands()
#to draw/connect the dots
mp_draw = mp.solutions.drawing_utils

'''
#for fps
pTime = 0
cTime = 0
'''


while True:
	success, img = cap.read()
	#converting image to RGB
	img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#results detects hands
	results = hands.process(img_RGB)
	#results.multi_hand_landmarks prints on console if the hand is detected on screen
	#print(results.multi_hand_landmarks)
	h,w,c = img.shape
	height = h
	width = w
	#print(h, w, c)
	#if results exists
	if results.multi_hand_landmarks:
		for h in results.multi_hand_landmarks: #for each hand 
			for id, lm in enumerate(h.landmark):
				cx, cy = int(lm.x * width), int(lm.y * height)
				if id == 4:
					cv2.circle(img, (cx,cy), 7, (255,255,255), cv2.FILLED)
				
			mp_draw.draw_landmarks(img, h, mp_hands.HAND_CONNECTIONS, 
			mp_draw.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 3)) #first 2 params draw dots, 3rd param connects dots

	'''
	#fps formula
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	#displaying fps on screen
	cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
	'''

	cv2.imshow("Image", img)
	cv2.waitKey(1)