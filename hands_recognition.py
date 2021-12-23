import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import mediapipe as mp
import time

class ASL:
    def __init__(self):
        self.load()
        pass
    
    def load(self):
        # Load the gesture recognizer model
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        self.pTime = 0
        self.cTime = 0
        
        self.model = load_model('mp_hand_gesture')

        # Load class names
        f = open('gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()
        print(self.classNames)
        
    def run(self):
        # initialize mediapipe

        frameWidth = 720
        frameHeight = 720
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, frameWidth)
        self.capture.set(4, frameHeight)

        while True:
            success, frame = self.capture.read()
            frame= cv2.flip(frame,1)
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            #print(results.multi_hand_landmarks)

            if results.multi_hand_landmarks:
                landmarks = []
                for handLms in results.multi_hand_landmarks: 
                #handLMs are 21 points. so we need conection too-->self.mpHands.HAND_CONNECTIONS
                    for id, lm in enumerate(handLms.landmark):
                        #print(id, lm)
                        #lm = x,y cordinate of each landmark in float numbers. lm.x, lm.y methods
                        #So, need to covert in integer
                        h, w, c =frame.shape
            #                 print(frame.shape)
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarks.append([cx, cy])
                        #print(id, cx, cy)
                        if id == 4: #(To draw 4th point)
                            cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS) #drawing points and lines(=handconections)

            #             print(results.)
                    if len(landmarks)==21:
                        prediction = self.model.predict([landmarks])


                        classID = np.argmax(prediction)
                        className = self.classNames[classID]
                        print(className)

            #Write frame rate
            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            cv2.putText(frame, "FPS= " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 0), 1)

            cv2.imshow('image', frame)
            if cv2.waitKey(1)==ord('q'):
                break

ASL().run()