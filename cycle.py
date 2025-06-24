import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math
import os
import platform
import ctypes

import pyttsx3 as pyt
from threading import Thread
import threading
from voiceModule import VoicePlay
from allMethods import allmethods
from face_detect import HeadPoseEstimator
# from body_position import bodyPosition


from vajrasana import vajrasana
from gomukhasana import gomukhasana
from sarvangasana import sarvangasana

def prevent_sleep():
    if platform.system() == 'Windows' :  # Windows
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x80000000)
    elif platform.system() == 'Darwin' :  # macOS
        os.system("caffeinate -d &")
    elif platform.system() == 'Linux' : # Linux
        os.system("xdg-screensaver reset")  # Works for most Linux distros

# Call this function at the start of your application
prevent_sleep()
# **************************************************************************************


class DisplayAll:
    
    def __init__(self,mode = False,max_hands=2,mindetectconf=0.5,mintrcackconf=0.5):
        # c += 1
        # print(c)
        # print('again......................................................................')


        self.mode = mode
        self.mindetectconf = mindetectconf
        self.mintrackconf = mintrcackconf
        self.max_hands = max_hands

        self.mpHand = mp.solutions.hands
        self.hand = self.mpHand.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.mindetectconf,
            min_tracking_confidence=self.mintrackconf
        )
        # self.mpDraw = mp.solutions.drawing_utils

        self.results = None
        self.mode = mode
        self.mindetectconf = mindetectconf
        self.mintrcackconf = mintrcackconf

        self.mpPose = mp.solutions.pose
        self.pose= self.mpPose.Pose(
            static_image_mode = self.mode,
            min_detection_confidence = self.mindetectconf,
            min_tracking_confidence = self.mintrcackconf
        )
        self.mpDraw= mp.solutions.drawing_utils

        self.lmlist = []
        self.vajrasana = vajrasana()
        self.gomukhasana = gomukhasana()
        self.sarvangasana = sarvangasana()
        self.all_methods = allmethods()
        self.voice = VoicePlay()

        # self.stop_ = False

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        #shoulder hip
        self.MAX_BACK_LIMT = 50
        self.MIN_BACK_LIMT = 0
        
        #hip knee
        self.MAX_THIGH_LIMT = 45
        self.MIN_THIGH_LIMT = 0

    def pose_positions(self,frames,draw = True):

        imgRB = cv.cvtColor(frames,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRB)
        

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frames,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

    def pose_landmarks(self,frames,draw = True):

        self.lmlist =[]

        if self.results.pose_landmarks:

            self.h,self.w,cl = frames.shape

            for id,poselms in enumerate(self.results.pose_landmarks.landmark):
                px,py ,pz= (poselms.x * self.w) , (poselms.y * self.h),(poselms.z)

                self.lmlist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lmlist
    

    def find_hands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hand.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)

        return img

    def get_landmarks(self, img, hand_no=0, draw=True):
        lms_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            h, w, c = img.shape

            for id, lm in enumerate(my_hand.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                lms_list.append([id, px, py])

                if draw:
                    cv.circle(img, (px, py), 2, (0, 255, 255), 2)

        return lms_list
    
    def vajrasana_cycle(self,frames,height,width,draw = False):

        self.va_correct = None
        
        check_side = self.all_methods.standing_side_view_detect(frames=frames,llist=pose_llist,height=height,width=width)
        vajrasana_slope = self.vajrasana.slope(frames=frames,llist=pose_llist,height=height,width=width)

        if not pose_llist or pose_llist == 0:
                pass

        elif len(pose_llist) != 0:

            #Left side
            if check_side == "left":
                left_vajrasana_angles = self.vajrasana.left_vajrasana(frames=frames,llist=pose_llist,elbow=(11,13,15), hip=(11,23,25), knee=(23,25,27), shoulder=(13,11,23),right_knee1=(24,26,28),draw=False)
                sitting_position = self.vajrasana.check_sitting(frames=frames,llist=pose_llist,height=height,width=width)
                if sitting_position:
                    wrong_left = self.vajrasana.wrong_vajrasana_left(frames=frames)
                self.va_correct = self.vajrasana.left_vajrasana_name(frames=frames)

            #Right Side
            elif check_side == "right":
                right_vajrasana_angles = self.vajrasana.right_vajrasana(frames=frames,llist=pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee1=(23,25,27),draw=False)
                sitting_position = self.vajrasana.check_sitting(frames=frames,llist=pose_llist,height=height,width=width)
                if sitting_position:
                    wrong_right = self.vajrasana.wront_right_vajrasana(frames=frames)
                self.va_correct = self.vajrasana.right_vajrasana_name(frames=frames)

            elif check_side == "left side cross position":
                self.voice.playAudio(["turn total left side"],play=True)

            elif check_side == "right side cross position":
                self.voice.playAudio(["turn total right side"],play=True)

            else:
                self.voice.playAudio(["turn left side or right side and do vajrasana"],play=True)

        if self.va_correct:
            return True

    def gomukhasana_cycle(self,frames,height,width,draw = False):

        self.go_correct = None

        check_sitting_position = self.gomukhasana.check_sitting(frames=frames,llist=pose_llist,height=height,width=width)
        side_view_detect = self.all_methods.standing_side_view_detect(frames=frames,llist=pose_llist,height=height,width=width)

        if not pose_llist or pose_llist == 0:
                pass

        elif len(pose_llist) != 0:

            if check_sitting_position:
                if len(pose_llist) != 0:
                    if side_view_detect == "forward":

                        self.gomukhasana.gomukhasana(frames=frames,llist=pose_llist,left_elbow=(11,13,15), left_hip=(11,23,25), left_knee=(23,25,27), left_shoulder=(13,11,23),right_elbow=(12,14,16), right_hip=(12,24,26),right_knee= (24,26,28), right_shoulder=(14,12,24),draw=False)
                        self.gomukhasana.check_slopes(frames=frames,lmlist=pose_llist,height=height,width=width)
                        self.gomukhasana.wrong_gomukhasana(frames=frames)
                        self.go_correct =  self.gomukhasana.gomukhasana_name(frames=frames)
                        
                    else:
                        self.voice.playAudio(["you must face the camera"],play=True)

        if self.go_correct:
            return True
        
    def sarvangasana_cycle(self,frames):

        if len(pose_llist) != 0:

            self.sarvangasana.right_sarvangasana(frames=frames,llist=pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),right_shoulder_ear=(23,11,7),draw=False)
            wrong_right = self.sarvangasana.wrong_right(frames=frames)
            correct = self.sarvangasana.right_sarvangasana_name(frames)

        # elif side_view == "left":

            self.sarvangasana.left_sarvangasana(frames=frames,llist=pose_llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),right_knee= (24,26,28),left_shoulder_ear=(24,12,8),draw=True)
            wrong_left = self.sarvangasana.wrong_left(frames=frames)
            correct = self.sarvangasana.left_sarvangasana_name(frames)



def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980)  #height

    finder="second_images"

    my_list= os.listdir(finder)

    overLay=dict({})

    modules = ["gomukhasana","sarvangasana","vajrasana"]
    
    for i, img in enumerate(my_list):
        
        if i < len(my_list):
            image = cv.imread(f'{finder}/{img}')
            image = cv.resize(image, (300, 300))
            
            overLay[modules[i]] = image         

    detect = DisplayAll()
    all_methods = allmethods()
    voice=VoicePlay()
    thumb_detect = False
    module  = "gomukhasana"
    print("hello")
    voice_detect = False
    flag = False

    while True:

        isTrue, video = video_capture.read()
        height , width , cl = video.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        global pose_llist

        img = cv.imread("videos/ashtanga2.jpg")


        detect.find_hands(video, False)
        # detect.pose_positions(video,False)
        hand_llist = detect.get_landmarks(video, draw=True)
        # pose_llist = detect.pose_landmarks(video,False)
        

        if not thumb_detect:
                cv.putText(video, "Show Thumb Up", (30, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
                voice.playAudio(["show thumb up to start the exercise"],play=True)


        if not flag:
            detect.pose_positions(video,False)
            pose_llist = detect.pose_landmarks(video,False)

            if len(hand_llist) != 0:  # Ensure we have enough landmark points
                thumb_tip = hand_llist[3]
                thumb_tip1 = hand_llist[4]
                index_tip = hand_llist[5]
                index_tip1 = hand_llist[6]

                
                if not thumb_detect and thumb_tip[2] < index_tip[2] and thumb_tip[2] < index_tip1[2] and thumb_tip1[2] < thumb_tip[2]:
                    # voice.stopEngine()
                    module = "gomukhasana"
                    thumb_detect = True
                            

            if thumb_detect:
                # all_methods.stop()

                for mod in overLay:

                    if module == "vajrasana" and not voice_detect and mod == module:
                        
                        image = overLay[mod]
                        h, w, _ = image.shape
                        video[0:h, 0:w] = image

                        flag = detect.vajrasana_cycle(video,height=height,width=width,draw=True)
                        
                    
                    elif module == "gomukhasana" and not voice_detect and mod == module:
                    
                        image = overLay[mod]
                        h, w, _ = image.shape
                        video[0:h, 0:w] = image
                        correct = detect.gomukhasana_cycle(video,height=height,width=width,draw=True)
                        if correct:
                            module = "sarvangasana"

                    elif module == "sarvangasana" and not voice_detect and mod == module:
                    
                        image = overLay[mod]
                        h, w, _ = image.shape
                        video[0:h, 0:w] = image
                        correct = detect.sarvangasana_cycle(video)
                        if correct:
                            module = "vajrasana"
                        

            cv.imshow("video",video)

            if cv.waitKey(10) & 0xFF == ord('d'):
                break
    
    video_capture.release()
    cv.destroyAllWindows()
# main()