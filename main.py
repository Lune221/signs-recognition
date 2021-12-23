__author__ = 'lune'
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import mediapipe as mp
import time

LETTERS = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","DEL","NOTHING","SPACE"]

class Indicator(Button):
    def __init__(self) -> None:
        super().__init__()
        self.on_color = (0, 1, 0, 1)
        self.off_color = (1, 0, 0, 1)
        self.background_color = self.off_color
        self.size_hint_y = None
        self.height = 20
    def on(self):
        self.background_color = self.on_color
    def off(self):
        self.background_color = self.off_color
        
class MyLabel(Label):
    def on_size(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0, 0, 0, 0.45)
            Rectangle(pos=self.pos, size=self.size)
            
class TranslationBox(BoxLayout):
    def __init__(self, translate) -> None:
        super().__init__()
        self.orientation = 'vertical'
        self.translating = False
        self.off_color = (0, 1, 0, 1)
        self.on_color = (1, 0, 0, 1)
        
        self.label = MyLabel(
            text='',
            pos=(20, 20),
            size_hint=(1, 0.5),
        )

        # self.add_widget(Widget(size_hint_y=None, height=220))
        self.add_widget(self.label)
        
        buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        
        self.trans = Button(text="Traduire", background_color=self.off_color)
        self.trans.bind(on_press=translate)
        self.trans.bind(on_press=self.changeTransButton)
        buttons.add_widget(self.trans)
        
        res = Button(text="Reinitialiser")
        res.bind(on_press=self.reset)
        buttons.add_widget(res)
        
        self.add_widget(buttons)
    def setTranlation(self, prediction):
        letter = LETTERS[prediction]
        print(letter, prediction)
        if letter == 'DEL':
            self.label.text = self.label.text[:-1]
        elif letter == 'NOTHING':
            pass
        elif letter == 'SPACE':
            self.label.text += ' '
        else:
            self.label.text += letter
            
    def reset(self, instance):
        self.label.text = ""
    
    def changeTransButton(self, instance):
        self.translating = not self.translating
        
        if self.translating :
            self.trans.background_color = self.on_color
            self.trans.text = "Arrêter"
        else:
            self.trans.background_color = self.off_color
            self.trans.text = "Traduire"
        

class CamApp(App):

    def __init__(self) -> None:
        super().__init__()
        self.hand_presence = False
        self.start_translation = False
        self.cropped_image = ""
        self.center = []
        self.margin = 150
        self.load()
        
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        self.img1=Image(size_hint_y=2)
        self.indicator = Indicator()
        self.translationBox = TranslationBox(self.translate)
        layout = BoxLayout(orientation='vertical')
        
        layout.add_widget(self.img1)
        layout.add_widget(self.indicator)
        layout.add_widget(self.translationBox)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        # cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def paste_img(self,img1,img2):
        if img2.size == 0: return
        
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
    
    def update(self, dt):
        # display image from cam in opencv window
        ret, frame = self.capture.read()
        prediction = ""
        frame= cv2.flip(frame,1)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        img_ = frame.copy()
        if results.multi_hand_landmarks:
            landmarks = []
            for handLms in results.multi_hand_landmarks: 
                for id, lm in enumerate(handLms.landmark):

                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([cx, cy])
                    if id == 9: #(To draw 4th point)
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                        #A
                        cv2.circle(frame, (cx-self.margin, cy-self.margin), 15, (255, 0, 255), cv2.FILLED)
                        #B
                        cv2.circle(frame, (cx+self.margin, cy-self.margin), 15, (255, 0, 255), cv2.FILLED)
                        #C
                        cv2.circle(frame, (cx+self.margin, cy+self.margin), 15, (255, 0, 255), cv2.FILLED)
                        #D
                        cv2.circle(frame, (cx-self.margin, cy+self.margin), 15, (255, 0, 255), cv2.FILLED)

                        self.center.append(cx)
                        self.center.append(cy)
                self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS) #drawing points and lines(=handconections)

                if len(landmarks)==21 and len(self.center) > 0:
                    self.hand_presence = True
                    self.cropped_image = img_[self.center[0]-self.margin:self.center[0]+self.margin, self.center[1]-self.margin:self.center[1]+self.margin]
                    
                    
                    # res_crop = cv2.resize(self.cropped_image,dsize=(64,64))
                    # res_crop = np.expand_dims(res_crop,axis=0)
                    # prediction = np.argmax(self.model_asl.predict([res_crop])[0])
                    # self.translationBox.setTranlation(str(prediction))

                    self.center = []
                    
                    frame = self.paste_img(frame,self.cropped_image)
                    
                else:
                    self.hand_presence = False

        #Write frame rate
        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        # cv2.putText(frame, "FPS= " + str(int(fps)) + " Predict: "+str(prediction), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 0), 1)

        # cv2.imshow('image', frame)
        if cv2.waitKey(1)==ord('q'):
            return
        
        buf1 = cv2.flip(frame, 0)
        if buf1 is not None:
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.img1.texture = texture1

    def translate(self, instance):
        self.start_translation = not self.start_translation
        if self.start_translation :
            self.translationEvent = Clock.schedule_interval(self.captureAndTranslate, 1.5)
        else:
            Clock.unschedule(self.translationEvent)
            self.indicator.off()
                
    def captureAndTranslate(self, dt):
        self.indicator.on()
        if self.hand_presence == True:
            res_crop = cv2.resize(self.cropped_image,dsize=(64,64))
            res_crop = np.expand_dims(res_crop,axis=0)
            prediction = np.argmax(self.model_asl.predict([res_crop])[0])
            self.translationBox.setTranlation(prediction)

        # self.indicator.off()
    
    def load(self):
        # Load the gesture recognizer model
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        self.pTime = 0
        self.cTime = 0
        
        self.model_asl = tf.keras.models.load_model("my_asl_model_local")
           
if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()