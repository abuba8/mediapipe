import cv2
import mediapipe as mp 


class faceLandmarks():
    def __init__(self, mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode,
        self.max_num_faces = max_num_faces,
        self.min_detection_confidence = min_detection_confidence,
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)


    def findFace(self, img, draw = True):
        self.img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img)
        img.flags.writeable = True
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:  
                if draw:
                    self.mp_draw.draw_landmarks(
                                image=img,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACE_CONNECTIONS,
                                landmark_drawing_spec=self.drawing_spec,
                                connection_drawing_spec=self.drawing_spec,
                                )
        return img