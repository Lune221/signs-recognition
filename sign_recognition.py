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
        pass
    
    def paste_img(self,img1,img2):

        img2 = cv2.resize(img2,(100,100))
        # I want to put logo on top-left corner, So I create a ROI
        rows,cols,channels = img2.shape
        roi = img1[0:rows, 0:cols ]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg)
        img1[0:rows, 0:cols ] = dst

        return img1
    def run(self):
        # initialize mediapipe
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils
        
                # Load the gesture recognizer model
        model = load_model('mp_hand_gesture')
        model_asl = tf.keras.models.load_model("my_asl_model_local")

        # Load class names
        f = open('gesture.names', 'r')
        classNames = f.read().split('\n')
        f.close()

        frameWidth = 720
        frameHeight = 720
        cap = cv2.VideoCapture(0)
        cap.set(3, frameWidth)
        cap.set(4, frameHeight)
        mpHands=mp.solutions.hands
        hands=mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils
        margin = 150

        pTime = 0
        cTime = 0
        center = []
        prediction = ""
        while True:
            try:
                success, img = cap.read()
                img= cv2.flip(img,1)
                img_ = img.copy()
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)
                #print(results.multi_hand_landmarks)

                if results.multi_hand_landmarks:
                    landmarks = []
                    for handLms in results.multi_hand_landmarks: 
                    #handLMs are 21 points. so we need conection too-->mpHands.HAND_CONNECTIONS

                        for id, lm in enumerate(handLms.landmark):
                            #print(id, lm)
                            #lm = x,y cordinate of each landmark in float numbers. lm.x, lm.y methods
                            #So, need to covert in integer
                            h, w, c =img.shape
            #                 print(img.shape)
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            landmarks.append([cx, cy])
                            #print(id, cx, cy)

                            if id == 9: #(To draw 4th point)
                                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                                #A
                                cv2.circle(img, (cx-margin, cy-margin), 15, (255, 0, 255), cv2.FILLED)
                                #B
                                cv2.circle(img, (cx+margin, cy-margin), 15, (255, 0, 255), cv2.FILLED)
                                #C
                                cv2.circle(img, (cx+margin, cy+margin), 15, (255, 0, 255), cv2.FILLED)
                                #D
                                cv2.circle(img, (cx-margin, cy+margin), 15, (255, 0, 255), cv2.FILLED)

                                center.append(cx)
                                center.append(cy)


                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #drawing points and lines(=handconections)

            #             print(results.)
                        if len(landmarks)==21 and len(center)>0:
            #                 prediction = model.predict([landmarks])

                            cropped_image = img_[center[0]-margin:center[0]+margin, center[1]-margin:center[1]+margin]
                            
                            res_crop = cv2.resize(cropped_image,dsize=(64,64))
                            res_crop = np.expand_dims(res_crop,axis=0)
                            prediction = np.argmax(model_asl.predict([res_crop])[0])
            #               

                            classID = np.argmax(prediction)

                            className = classNames[classID]
                            center = []
                            
                            img = self.paste_img(img,cropped_image)
                            print(prediction)
         
                #Write frame rate
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, "FPS= " + str(int(fps))+"Predict"+str(prediction), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 0), 1)

                cv2.imshow('image', img)
                if cv2.waitKey(1)==ord('q'):
                    break
            except Exception as e:
                print(e)
                
ASL().run()