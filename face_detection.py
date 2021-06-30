import cv2
import mediapipe as mp 


class faceDetection():
    def __init__(self, min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence,

        self.mp_face_detection = mp.solutions.face_detection
        self.facedetection = self.mp_face_detection.FaceDetection(self.min_detection_confidence)
        self.mp_drawing = mp.solutions.drawing_utils


    def detectFace(self, img, draw = True):
        self.img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img)
        img.flags.writeable = True
        if self.results.detections:
            for detection in self.results.detections:  
                if draw:
                    self.mp_drawing.draw_detection(img, detection)
        return img
