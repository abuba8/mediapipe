import cv2
import mediapipe as mp 
import time

class handDetector():
	def __init__(self, mode=False, max_hands=2, detection_confidence = 0.5,track_confidence = 0.5):
		self.mode = mode
		self.max_hands = max_hands
		self.detection_confidence = detection_confidence
		self.track_confidence = track_confidence

		self.mp_hands = mp.solutions.hands 
		self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_confidence, self.track_confidence)
		self.mp_draw = mp.solutions.drawing_utils



	def findHands(self, img, draw = True):
		img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(img_RGB)
		if self.results.multi_hand_landmarks:
			for h in self.results.multi_hand_landmarks: #for each hand 
				if draw:
					self.mp_draw.draw_landmarks(img, h, self.mp_hands.HAND_CONNECTIONS, self.mp_draw.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 3)) #first 2 params draw dots, 3rd param connects dots

		return img

	def findPosition(self, img, hand_number=0, draw=True):
		lm_list = []
		h,w,c = img.shape
		height = h
		width = w
		if self.results.multi_hand_landmarks:
			my_hand = self.results.multi_hand_landmarks[hand_number]
			for id, lm in enumerate(my_hand.landmark):
				cx,cy = int(lm.x * width), int(lm.y * height)
				lm_list.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx,cy), 7, (255,255,255), cv2.FILLED)

		return lm_list
		



'''
def main():
	cap = cv2.VideoCapture(0)
	detector = handDetector()
	while True:
		success, img = cap.read()
		img = detector.findHands(img)
		lmlist = detector.findPosition(img)
		if len(lmlist) != 0:
			print(lmlist[4])

		cv2.imshow("Image", img)
		cv2.waitKey(1)

if __name__ == "__main__":
	main()
'''