import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math

import pyttsx3 as pyt
from threading import Thread
import threading
from voiceModule import VoicePlay
from allMethods import allmethods
# from body_position import bodyPosition


class treepose:
    
    def __init__(self,mode = False,mindetectconf=0.5,mintrcackconf=0.5):
        # c += 1
        # print(c)
        # print('again......................................................................')
        self.results = None
        self.mode = mode
        self.mindetectconf = mindetectconf
        self.mintrcackconf = mintrcackconf

        self.engine = pyt.init()
        voices = self.engine.getProperty("voices")
        self.engine.setProperty('voice',voices[1].id)
        self.engine.setProperty("rate", 115)

        self.mpPose = mp.solutions.pose
        self.pose= self.mpPose.Pose(
            static_image_mode = self.mode,
            min_detection_confidence = self.mindetectconf,
            min_tracking_confidence = self.mintrcackconf
        )
        self.mpDraw= mp.solutions.drawing_utils

        self.lmlist = []
        self.m_lmlist = []
        self.angle =0
        self.voice_thread = None
        self.voice_detect = False
        self.lock = threading.Lock()
        self.voice = VoicePlay()
        self.all_methods = allmethods()
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
                px,py ,pz=int (poselms.x * self.w) , int(poselms.y * self.h),(poselms.z)

                self.lmlist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lmlist
    
    def treepose(self, frames,llist, left_elbow, left_hip, left_knee,left_shoulder,right_elbow, right_hip, right_knee, right_shoulder,draw =True):

        
        self.left_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=left_elbow,lmList=llist)
        self.left_hip, hip_coords = self.all_methods.calculate_angle(frames=frames,points= left_hip,lmList=llist)
        self.left_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= left_knee,lmList=llist)       
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=left_shoulder, lmList=llist)

        self.right_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=right_elbow,lmList=llist)
        self.right_hip, hip_coords = self.all_methods.calculate_angle(frames=frames, points=right_hip,lmList=llist)
        self.right_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= right_knee,lmList=llist)
        self.right_shoulder,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=right_shoulder, lmList=llist)

        if draw:
            if elbow_coords:
                cv.putText(frames, str(int(self.right_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.right_hip)), (hip_coords[2]-20, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.right_knee)), (knee_coords[2]-20, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            if points_cor7:
                cv.putText(frames, str(int(self.right_shoulder)), (points_cor7[2]+10, points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

            if elbow_coords:
                cv.putText(frames, str(int(self.left_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.left_hip)), (hip_coords[2]+10, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.left_knee)), (knee_coords[2]+10, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            if points_cor8:
                cv.putText(frames, str(int(self.left_shoulder)), (points_cor8[2]+10, points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)


        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder#self.head_position


        
        
        # if draw:
           
    def right_treepose(self,frames,llist,elbow, hip, knee, shoulder,draw=True):

        # self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        # self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        # self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        # self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        # if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
        #     return 
        
        self.right_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
        self.right_hip, hip_coords = self.all_methods.calculate_angle(frames=frames, points=hip,lmList=llist)
        self.right_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)
        self.right_shoulder,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)

        if draw:
            if elbow_coords:
                cv.putText(frames, str(int(self.right_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.right_hip)), (hip_coords[2]-20, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.right_knee)), (knee_coords[2]-20, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            if points_cor7:
                cv.putText(frames, str(int(self.right_shoulder)), (points_cor7[2]+10, points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder
    
        
    def check_stand(self,frames,llist,height,width):

        
        self.view_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,hip_points=(11,23,25),leg_points=(23,25,27),elbow_points=(11,13,15),height=height,width=width)

        # return self.view_position

        if self.view_position == "standing":

            # self.voice.playAudio(["you can start your yoga","stand in a left side or right side position and do treepose"],play=True)
            return True
        
        elif self.view_position != "standing":
            self.voice.playAudio(["you must be in standing position"],play=True)


    def treepose_name(self,frames):

            correct = (
                    self.left_knee and 160 <= self.left_knee <= 180 and
                    self.right_knee and 30 <= self.right_knee <= 50 and
                    self.left_elbow and 150 <= self.left_elbow <= 180 and
                    self.right_elbow and 150 <= self.right_elbow <= 180 and
                    self.left_shoulder and 150 <= self.left_shoulder <= 180 and
                    self.right_shoulder and 150 <= self.right_shoulder <= 180 and
                    self.left_hip and 160 <= self.left_hip <= 180 and
                    self.right_hip and 160 <= self.right_hip <= 180 
                   )
            
            if correct:

                self.voice.playAudio(["you completed your yoga pose corrected"],play=True)


def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    global  llist
    detect = treepose()
    all_methods = allmethods()
    voice = VoicePlay()
    
    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        img = cv.imread("images/image1.webp")
        # resized_img = cv.resize(img, None, fx=3.0, fy=3.0, interpolation=cv.INTER_LINEAR)
        
        detect.pose_positions(frames,draw=False)
        llist = detect.pose_landmarks(frames,False)
    
        if not llist or llist == 0:
            pass

        elif len(llist) != 0:

            #Left side
            # if side_view == "left":
                detect.treepose(frames=frames,llist=llist,left_elbow=(11,13,15), left_hip=(11,23,25), left_knee=(23,25,27), left_shoulder=(13,11,23),right_elbow=(12,14,16), right_hip=(12,24,26),right_knee= (24,26,28), right_shoulder=(14,12,24),draw=True)
                # stand_position = detect.check_stand(frames=frames,llist=llist,height=detect.h,width=detect.w)
                # if stand_position:

                    # wrong_left = detect.wrong_treepose(frames=frames)
                treepose_name = detect.treepose_name(frames=frames)


        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()