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
from face_detect import HeadPoseEstimator
# from body_position import bodyPosition
from abstract_class import yoga_exercise


class natrajasana(yoga_exercise):
    
    def __init__(self,mode = False,mindetectconf=0.5,mintrcackconf=0.5):
        # c += 1
        # print(c)
        # print('again......................................................................')
        self.results = None
        self.mode = mode
        self.mindetectconf = mindetectconf
        self.mintrcackconf = mintrcackconf

        self.check_stand = False
        self.initial_position = False
        self.side_detect = False
        self.initial_count = 0

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
        self.head_pose = HeadPoseEstimator()
        # self.stop_ = False

        
        self.forward_count = 0
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
    
    def left_natrajasana(self, frames,llist, elbow, hip, knee,shoulder,right_knee1,right_elbow,draw =True):

        self.head = self.head_pose_estimator.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose_estimator.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose_estimator.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose_estimator.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 

        
        self.left_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
        self.left_hip, hip_coords = self.all_methods.calculate_angle(frames=frames,points= hip,lmList=llist)
        self.left_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)       
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.right_knee1, right_knee1_coords = self.all_methods.calculate_angle(frames=frames,points= right_knee1,lmList=llist)
        self.right_elbow1, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=right_elbow,lmList=llist)
        
        if draw:

            cv.putText(frames,f'l_elbow{str(self.left_elbow)}',(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_hip{str(self.left_hip)}',(10,80),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_knee{str(self.left_knee)}',(10,120),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_shoulder{str(self.left_shoulder)}',(10,160),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_knee1{str(self.right_knee1)}',(10,200),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_elbow{str(self.right_elbow1)}',(10,240),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.right_knee1,self.right_elbow1,self.head_position #self.left_knee_y,self.left_wrist_y,self.ground_left,self.ground_left_min

    def right_natrajasana(self,frames,llist,elbow, hip, knee, shoulder,left_knee1,left_elbow,draw=True):

        self.head = self.head_pose_estimator.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose_estimator.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose_estimator.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose_estimator.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        self.right_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
        self.right_hip, hip_coords = self.all_methods.calculate_angle(frames=frames, points=hip,lmList=llist)
        self.right_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)
        self.right_shoulder,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.left_knee1, left_knee1_coords = self.all_methods.calculate_angle(frames=frames,points= left_knee1,lmList=llist)
        self.left_elbow1, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=left_elbow,lmList=llist)

        if draw:

            cv.putText(frames,f'r_elbow{str(self.right_elbow)}',(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_hip{str(self.right_hip)}',(10,80),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_knee{str(self.right_knee)}',(10,120),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_shoulder{str(self.right_shoulder)}',(10,160),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_knee1{str(self.left_knee1)}',(10,200),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_elbow{str(self.left_elbow1)}',(10,240),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)

        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,left_knee1,self.left_elbow1,self.head_position#self.right_knee_y,self.right_wrist_y,self.ground_right,self.ground_right_min
    
    def wrong_left(self,frames,llist,height,width):
        
        if not llist:
            return 
        
        left_knee_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width)
        
        initial_pos = (self.left_hip and 160 <= self.left_hip <= 180 and
                        self.left_knee and 160 <= self.left_knee <= 180 and
                       self.right_knee1 and 160 <= self.right_knee1 <= 180)

        if not self.check_stand and self.all_methods.is_person_standing_sitting == "standing":
            self.all_methods.reset_after_40_sec()
            check = self.all_methods.play_after_40_sec(["you are in standing position please turn your body at any side"],llist=llist)
            if check:
                self.check_stand = True
                
        elif self.check_stand and not self.side_detect:
            
            if self.all_methods.standing_side_view_detect == "left":
                self.side_detect = True
                
            else:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["please turn your body which side you are comfortable"],llist= llist)
                
        elif self.side_detect and not self.initial_position:#and 
            if not initial_pos:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are not in initial position please be in standing"],llist=llist)
            
            elif initial_pos:
                self.initial_position = True
                
            
        elif self.initial_position:
            
            if initial_pos:
                initial_voices = ["please do natrajasana","please lift your left leg up"]
            
                if self.initial_count < len(initial_voices):
                    self.all_methods.reset_after_40_sec()
                    voice = self.all_methods.play_after_40_sec([initial_voices[self.initial_count]],llist=llist)
                    if voice:
                        self.initial_count += 1
                else:
                    self.initial_count = 1
                    
            else:
                middle_foot = (self.all_methods.l_ankle_y + self.all_methods.l_toe_y) // 2
                
                if self.all_methods.l_knee_x < self.all_methods.l_hip_x:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please keep your leg back side"],llist=llist)
                    
                elif self.all_methods.l_ankle_x < self.all_methods.l_hip_x:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please keep your left ankle back side"],llist=llist)
                
                elif middle_foot > self.all_methods.l_hip_y :
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please lift your left leg up ankle back side, cross your hip,hold your toe with left hand"],llist=llist)
                    
                else:
                    
                    to_hold_foot = abs(int((self.all_methods.l_toe_y - self.all_methods.l_wrist_y)))
                    
                    if self.all_methods.l_wrist_x < self.all_methods.l_shoulder_x:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["keep your lower hand back side"],llist=llist)
                    
                    elif to_hold_foot and to_hold_foot > 100:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["please hold your left hand with left toe"])
                        
                    else:
                        if self.left_knee and 0 <= self.left_knee <= 59:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please raise up left leg , maintain lower and upper leg 90 degrees"],llist=llist)
                            
                        elif self.left_knee and left_knee_hip  and 21 <= left_knee_hip <= 90:
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your left upper leg in flat position"],llist=llist)
                            
                        elif self.left_knee and 101 <= self.left_knee <= 180:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please bend your left leg , maintain lower and upper leg 90 degrees"],llist=llist)
                            
                        else:
                            if self.right_knee1 and 0 <= self.right_knee1 <= 159:   
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please keep your right leg straight"],llist=llist)

                            else:
                                if self.left_hip and 0 <= self.left_hip <= 99:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["you are bending too much please slight relax your hip"],llist=llist)
                                    
                                elif self.left_hip and 141 <= self.left_hip <= 180:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please raise  your left leg"],llist=llist)
                                    
                                else:
                                    if self.left_elbow and 0 <= self.left_elbow <= 159:
                                       self.all_methods.reset_after_40_sec()
                                       self.all_methods.play_after_40_sec(["keep your left elbow straight and hold your toe"],llist=llist)
                                        
                                    else:
                                        if self.left_shoulder and 0 <= self.left_shoulder <= 60:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please raise your left hand up"],llist=llist)
                                            
                                        elif self.left_shoulder and 100 <= self.left_shoulder <= 180:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please down your left hand"],llist=llist)
                                            
                                        else:
                                            if self.all_methods.r_elbow_x > self.all_methods.l_hip_x:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["keep your right hand in forward side"],llist=llist)  
                                            
                                            else:
                                                if self.right_elbow1 and 0 <= self.right_elbow1 <= 159:
                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["keep your right hand straight"],llist=llist)
                                               
                                                else:
                                                    self.initial_count = 0
                                                    return True
            
    def wrong_right(self,frames,llist,height,width):
        
        if not llist:
            return 
        
        right_knee_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width)
        
        initial_pos = (self.right_hip and 160 <= self.right_hip <= 180 and
                        self.right_knee and 160 <= self.right_knee <= 180 and
                       self.left_knee1 and 160 <= self.left_knee1 <= 180)

        if not self.check_stand and self.all_methods.is_person_standing_sitting == "standing":
            self.all_methods.reset_after_40_sec()
            check = self.all_methods.play_after_40_sec(["you are in standing position please turn your body at any side"],llist=llist)
            if check:
                self.check_stand = True
                
        elif self.check_stand and not self.side_detect:
            
            if self.all_methods.standing_side_view_detect == "right":
                self.side_detect = True
                
            else:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["please turn your body which side you are comfortable"],llist= llist)
                
        elif self.side_detect and not self.initial_position:#and 
            if not initial_pos:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are not in initial position please be in standing"],llist=llist)
            
            elif initial_pos:
                self.initial_position = True
                
            
        elif self.initial_position:
            
            if initial_pos:
                initial_voices = ["please do natrajasana","please lift your right leg up"]
            
                if self.initial_count < len(initial_voices):
                    self.all_methods.reset_after_40_sec()
                    voice = self.all_methods.play_after_40_sec([initial_voices[self.initial_count]],llist=llist)
                    if voice:
                        self.initial_count += 1
                else:
                    self.initial_count = 1
                    
            else:
                middle_foot = (self.all_methods.l_ankle_y + self.all_methods.l_toe_y) // 2
                
                if self.all_methods.r_knee_x > self.all_methods.r_hip_x:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please keep your leg back side"],llist=llist)
                    
                elif self.all_methods.r_ankle_x > self.all_methods.r_hip_x:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please keep your right ankle back side"],llist=llist)
                
                elif middle_foot > self.all_methods.r_hip_y :
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please lift your right leg up ankle back side, cross your hip,hold your toe with right hand"],llist=llist)
                    
                else:
                    
                    to_hold_foot = abs(int((self.all_methods.l_toe_y - self.all_methods.l_wrist_y)))
                    
                    if self.all_methods.r_wrist_x > self.all_methods.r_shoulder_x:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["keep your right lower hand back side"],llist=llist)
                    
                    elif to_hold_foot and to_hold_foot > 100:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["please hold your right hand with right toe"])
                        
                    else:
                        if self.right_knee and 0 <= self.right_knee <= 59:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please raise up right leg , maintain lower and upper leg 90 degrees"],llist=llist)
                            
                        elif self.right_knee and right_knee_hip  and 21 <= right_knee_hip <= 90:
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your right upper leg in flat position"],llist=llist)
                            
                        elif self.right_knee and 101 <= self.right_knee <= 180:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please bend your right leg , maintain lower and upper leg 90 degrees"],llist=llist)
                            
                        else:
                            if self.left_knee1 and 0 <= self.left_knee1 <= 159:   
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please keep your right leg straight"],llist=llist)

                            else:
                                if self.right_hip and 0 <= self.right_hip <= 99:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["you are bending too much please slight relax your hip"],llist=llist)
                                    
                                elif self.right_hip and 141 <= self.right_hip <= 180:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please raise  your right leg"],llist=llist)
                                    
                                else:
                                    if self.right_elbow and 0 <= self.right_elbow <= 159:
                                       self.all_methods.reset_after_40_sec()
                                       self.all_methods.play_after_40_sec(["keep your right elbow straight and hold your toe"],llist=llist)
                                        
                                    else:
                                        if self.right_shoulder and 0 <= self.right_shoulder <= 60:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please raise your right hand up"],llist=llist)
                                            
                                        elif self.right_shoulder and 100 <= self.right_shoulder <= 180:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please down your right hand"],llist=llist)
                                            
                                        else:
                                            if self.all_methods.l_elbow_x < self.all_methods.r_hip_x:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["keep your right hand in forward side"],llist=llist)  
                                            
                                            else:
                                                if self.left_elbow1 and 0 <= self.left_elbow1 <= 159:
                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["keep your right hand straight"],llist=llist)
                                               
                                                else:
                                                    self.initial_count = 0
                                                    return True
                                                
    def check_standing(self,frames,llist,height,width):

        standing_position = self.all_methods.is_person_standing_standing(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if standing_position == "standing":
            
            return True 
        

        elif standing_position != "standing":
            # self.all_methods.reset_after_40_sec()
            # self.all_methods.play_after_40_sec(["please be in standing position","   ","this yoga may started in standing position"],llist=llist)
            return False
                                                
    def left_reverse(self,frames):
        
        self.l_r_count = 0
        l_r_voice = ["good stay in same position ","very good get relax and come to stand position"]
        
        if (self.left_knee and 60 <= self.left_knee <= 100) and (self.right_knee1 and 160 <= self.right_knee1 <= 180):
            
            if l_r_voice < len(l_r_voice):
                self.all_methods.reset_after_40_sec()
                l_r = self.all_methods.play_after_40_sec([l_r_voice[self.l_r_count]],llist=llist)
                if l_r:
                    self.l_r_count += 1
                    
            else:
                self.l_r_count = 1
                
        if self.left_knee and 0 <= self.left_knee <= 159:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be staright your left leg"],llist=llist)
         
        if self.all_methods.is_person_standing_sitting == "standing":    
            self.all_methods.reset_after_40_sec()
            voice = self.all_methods.play_after_40_sec(["good job your yoga is perfectly completed"],llist=llist)
            if voice:
                return True
                
    def right_reverse(self,frames):
        self.r_r_count = 0
        r_r_voice = ["good stay in same position ","very good get relax and come to stand position"]
        
        if (self.right_knee and 60 <= self.right_knee <= 100) and (self.left_knee1 and 160 <= self.left_knee1 <= 180):
            
            if r_r_voice < len(r_r_voice):
                self.all_methods.reset_after_40_sec()
                r_r = self.all_methods.play_after_40_sec([r_r_voice[self.r_r_count]],llist=llist)
                if r_r:
                    self.r_r_count += 1
                    
            else:
                self.r_r_count = 1
                
        if self.right_knee and 0 <= self.right_knee <= 159:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be staright your right leg"],llist=llist)
        
        if self.all_methods.is_person_standing_sitting == "standing":    
            self.all_methods.reset_after_40_sec()
            voice = self.all_methods.play_after_40_sec(["good job your yoga is perfectly completed"],llist=llist)
            if voice:
                return True
    
    def left_natrajasana_name(self,frames):
        
        correct = (self.left_knee and 60 <= self.left_knee <= 100 and
                   self.right_knee1 and 160 <= self.right_knee1 <= 180 and
                   self.left_hip and 100 <= self.left_hip <= 140 and
                   self.left_elbow and 160 <= self.left_elbow <= 180 and
                   self.right_elbow1 and 160 <= self.right_elbow1 <= 180 and
                   self.left_shoulder and 61 <= self.left_shoulder <= 99 and
                   self.all_methods.r_elbow_x < self.all_methods.l_hip_x and
                   self.all_methods.l_wrist_x > self.all_methods.l_shoulder_x and
                   self.all_methods.l_ankle_x > self.all_methods.l_hip_x and
                   self.all_methods.l_knee_x > self.all_methods.l_hip_x)
        
        if correct:
            return True
        
        
    def right_natrajasana_name(self,frames):
        
        correct = (self.right_knee and 60 <= self.right_knee <= 100 and
                   self.left_knee1 and 160 <= self.left_knee1 <= 180 and
                   self.right_hip and 100 <= self.right_hip <= 140 and
                   self.right_elbow and 160 <= self.right_elbow <= 180 and
                   self.left_elbow1 and 160 <= self.left_elbow1 <= 180 and
                   self.right_shoulder and 61 <= self.right_shoulder <= 99 and
                   self.all_methods.l_elbow_x > self.all_methods.r_hip_x and
                   self.all_methods.r_wrist_x < self.all_methods.r_shoulder_x and
                   self.all_methods.r_ankle_x < self.all_methods.r_hip_x and
                   self.all_methods.r_knee_x < self.all_methods.r_hip_x)
        
        if correct:
            return True

def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720) 
    global llist
    all_methods = allmethods()
    detect = natrajasana()
    flag = False
    
    check_stand = False
    ready_for_exercise = False
    reverse_yoga = False
    checking_wrong = False

    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        img1 = cv.imread("images/image3.webp")
        img = cv.resize(img1, None, fx=2.0, fy=2.0, interpolation=cv.INTER_LINEAR)
        detect.pose_positions(frames,draw = False)
        llist = detect.pose_landmarks(frames,draw=False)
        stand = detect.check_standing(frames=frames,llist=llist,height=height,width=width)
        
        if len(llist) == 0:
            return

        if len(llist) != 0:
            
            if not flag:
            
                if not check_stand:
                    if stand:
                        check_stand = True
                        
                if check_stand:
                    side_view = all_methods.standing_side_view_detect(frames,llist=llist,height=height,width=width)
                    if side_view == "right":
                        
                        detect.right_natrajasana(frames,llist,elbow=(12,14,16), hip=(12,24,26), knee=(24,26,28), shoulder=(14,12,24),left_knee1=(23,25,27),left_elbow=(11,13,15),draw=False)
                        if not checking_wrong:
                            wrong_right = detect.wrong_right(frames=frames)
                            if wrong_right:
                                checking_wrong = True
                        
                        elif checking_wrong and not reverse_yoga:
                            correct = detect.right_natrajasana_name(frames=frames)
                            if correct:
                                reverse_yoga = True
                            
                        elif reverse_yoga:
                                reverse = detect.right_reverse(frames=frames)
                                if reverse:
                                    flag = True

                    elif side_view =="left":
                        
                        detect.left_natrajasana(frames,llist,elbow=(11,13,15),hip= (11,23,25),knee= (23,25,27), shoulder=(13,11,23),right_knee1=(24,26,28),right_elbow=(12,14,16),draw=False)
                        if not checking_wrong:
                            wrong_right = detect.wrong_right(frames=frames)
                            if wrong_right:
                                checking_wrong = True
                        
                        elif checking_wrong and not reverse_yoga:
                            correct = detect.right_natrajasana_name(frames=frames)
                            if correct:
                                reverse_yoga = True
                            
                        elif reverse_yoga:
                                reverse = detect.right_reverse(frames=frames)
                                if reverse:
                                    flag = True

        cv.imshow("video",frames)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()   