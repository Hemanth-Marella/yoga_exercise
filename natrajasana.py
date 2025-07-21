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

        self.results = None
        self.mode = mode
        self.mindetectconf = mindetectconf
        self.mintrcackconf = mintrcackconf

        self.check_stand = False
        self.initial_position = False
        self.initial_count = 0
        self.first_pose_detected = False
        

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
    
    def natrajasana(self, frames,llist, l_elbow, l_hip, l_knee,l_shoulder,r_elbow,r_hip,r_knee,r_shoulder,l_draw =True,r_draw = True,l_v_draw = True,r_v_draw = True):
        
        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        self.left_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=l_elbow,lmList=llist,draw=l_draw)
        self.left_hip, hip_coords = self.all_methods.calculate_angle(frames=frames,points= l_hip,lmList=llist,draw=l_draw)
        self.left_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= l_knee,lmList=llist,draw=l_draw)       
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=l_shoulder, lmList=llist,draw=l_draw)

        self.right_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=r_elbow,lmList=llist,draw=r_draw)
        self.right_hip, hip_coords = self.all_methods.calculate_angle(frames=frames, points=r_hip,lmList=llist,draw=r_draw)
        self.right_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= r_knee,lmList=llist,draw=r_draw)
        self.right_shoulder,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=r_shoulder, lmList=llist,draw=r_draw)

        if l_v_draw:

            cv.putText(frames,f'l_elbow{str(self.left_elbow)}',(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_hip{str(self.left_hip)}',(10,80),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_knee{str(self.left_knee)}',(10,120),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_shoulder{str(self.left_shoulder)}',(10,160),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
        if r_v_draw:
            cv.putText(frames,f'r_elbow{str(self.right_elbow)}',(10,200),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_hip{str(self.right_hip)}',(10,220),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_knee{str(self.right_knee)}',(10,240),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_shoulder{str(self.right_shoulder)}',(10,260),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)


    def side_view_detect(self,frames,llist):

        self.all_methods.all_x_values(frames=frames,llist=llist)
        self.all_methods.all_y_values(frames=frames,llist=llist)
        self.all_methods.all_z_values(frames=frames,llist=llist)

        result = None
        
        # self.all_methods.all_z_values(frames=frames,llist=llist)
        # print(self.all_methods.l_ankle_z)
        
        shoulder_diff_z = abs(self.all_methods.l_shoulder_z - self.all_methods.r_shoulder_z)
        hip_diff_z = abs(self.all_methods.l_hip_z - self.all_methods.r_hip_z)

        
        if shoulder_diff_z > 0.05 and hip_diff_z > 0.05:
            
            if self.all_methods.l_shoulder_z < self.all_methods.r_shoulder_z:
                # if self.all_methods.l_hip_z < self.all_methods.r_hip_z:
                    result = "left"
                
            elif self.all_methods.r_shoulder_z < self.all_methods.l_shoulder_z:
                # if self.all_methods.r_hip_z < self.all_methods.l_hip_z:
                    result = "right"
        else:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please turn your body which side you are comfortable"],llist=llist)
            result = "forward" 

        cv.putText(frames,result,(10,50),cv.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)

        return result
    
    def wrong_left(self,frames,llist,height,width):

        self.all_methods.all_y_values(frames=frames,llist=llist)
        self.all_methods.all_x_values(frames=frames,llist=llist)
        self.all_methods.all_z_values(frames=frames,llist=llist)

        self.side = self.side_view_detect(frames=frames,llist=llist)
        if not llist:
            return 
        
        left_knee_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=True)
        
        check_stand = (self.left_hip and 160 <= self.left_hip <= 180 and
                        self.left_knee and 160 <= self.left_knee <= 180 and
                       self.right_knee and 160 <= self.right_knee <= 180)
        
        self.l_touch_hand_foot_x = abs(int(self.all_methods.l_wrist_x - self.all_methods.l_ankle_x))
        self.l_touch_hand_foot_y = abs(int(self.all_methods.l_wrist_y - self.all_methods.l_ankle_y))
        
        # cv.putText(frames,str(self.l_touch_hand_foot_y),(10,100),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        
        if not self.check_stand:#and 
            
            if not check_stand:
                # print("hello")
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are not in standing position please be in standing"],llist=llist)
            
            elif check_stand:
                self.check_stand = True     
            
        if self.check_stand and not self.initial_position:
                
            if check_stand:
                initial_voices = ["you are in initial position please do natrajasana","please, fold ,and, lift your left leg up"]
            
                if self.initial_count < len(initial_voices):
                    self.all_methods.reset_after_40_sec()
                    voice = self.all_methods.play_after_40_sec([initial_voices[self.initial_count]],llist=llist)
                    if voice:
                        self.initial_count += 1
                else:
                    self.initial_count = 1
                    
            else:

                if self.left_knee and 0<= self.left_knee <= 24 :
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["your left leg is bending too much , please slight stretch your left leg"],llist=llist)

                elif self.left_knee and 51 <= self.left_knee <= 180:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please bend your left leg slight"],llist=llist)
                
                else:
                    if self.all_methods.r_wrist_y > self.all_methods.nose_y:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["please, raise your right hand up"],llist=llist)

                    else:
                        if self.all_methods.l_ankle_x < self.all_methods.l_hip_x:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please keep your left ankle back side"],llist=llist)

                        else:

                            if self.all_methods.l_elbow_x < self.all_methods.l_hip_x:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please keep your left hand back side, and hold your foot"],llist=llist)

                            else:
                                if (self.left_elbow and 0 <= self.left_elbow <= 159):
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please keep your left hand straight"],llist=llist)
                                
                                else:

                                    if self.l_touch_hand_foot_x >= 50 and self.l_touch_hand_foot_y >= 100:
                                        # cv.putText(frames,str(self.l_touch_hand_foot_x,self.l_touch_hand_foot_y),(10,100),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please touch your left foot with left hand"],llist=llist)

                                    else:
                                        if not self.first_pose_detected:
                                            if (self.left_knee and 25 <= self.left_knee <= 50 and
                                                self.left_hip and 160 <= self.left_hip <= 180 and
                                                self.left_shoulder and 15 <= self.left_shoulder <= 40 and
                                                self.right_knee and 160 <= self.right_knee <= 180 and
                                                self.left_elbow and 160 <= self.left_elbow <= 180 and
                                                self.all_methods.r_wrist_y < self.all_methods.nose_y 
                                                ):

                                                self.all_methods.reset_after_40_sec()

                                                first_pose = self.all_methods.play_after_40_sec(["good , you completed, half pose of this yoga"],llist=llist)
                                                if first_pose:
                                                    self.first_pose_detected = True


                    if self.first_pose_detected:

                        # self.l_touch_hand_foot_x = abs(int(self.all_methods.l_wrist_x - self.all_methods.l_ankle_x))
                        # self.l_touch_hand_foot_y = abs(int(self.all_methods.l_wrist_y - self.all_methods.l_ankle_y))

                        if self.l_touch_hand_foot_x >= 50 and self.l_touch_hand_foot_y >= 80:

                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please touch your left foot with left hand"],llist=llist)

                        if self.left_hip and 0 <= self.left_hip <= 50:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["you are bending too much , please  bend slightly"],llist=llist)

                        elif self.left_hip and 51 <= self.left_hip <= 109:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["good , you are at correct way, please bend slightly"],llist=llist)        
                        
                        elif self.left_hip and 141 <= self.left_hip <= 160:   
                            self.hip_count =0 
                            hip_list = ["good, you almost done ,please,lift your left leg little more","good,please, raise your left leg little more"]
                        
                            if self.hip_count < len(hip_list):
                                self.all_methods.reset_after_40_sec()
                                hip_voice = self.all_methods.play_after_40_sec([hip_list[self.hip_count]],llist=llist)
                                if hip_voice:
                                    self.hip_count += 1
                            else:
                                self.hip_count = 0

                        elif self.left_hip and 161 <= self.left_hip <= 180:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["carefully ,and , slowly lift your right leg , with help of your hand"],llist=llist)
                        
                        else:

                            same_axis_for_knee_hip = abs(int(self.all_methods.l_knee_y - self.all_methods.l_hip_y))
                            # cv.putText(frames,f'axis{str(same_axis_for_knee_hip)}',(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
                            if same_axis_for_knee_hip >= 100:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please , ensure your upper leg is parallel to ground"],llist=llist)
                                
                            else:
                                if self.all_methods.l_ankle_y > self.all_methods.l_hip_y:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please raise your right ankle , and , cross your hip in up"],llist=llist)

                                if self.left_knee and 0 <= self.left_knee <= 34:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please raise up left leg , maintain lower and upper leg 90 degrees"],llist=llist)
                                
                                elif self.left_knee and 35 <= self.left_knee <= 70 :
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec([" you almost good,move some outwards your left leg"],llist=llist)

                                elif self.left_knee and 111 <= self.left_knee <= 150:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["good , please make some bend your left leg."],llist=llist)
                                    
                                elif self.left_knee and 151 <= self.left_knee <= 180:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please bend your right leg ,and make 90 degrees between "],llist=llist)
                                else:
                                    
                                    if self.left_elbow and 0 <= self.left_elbow <= 89:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please , keep your left elbow straight, hold your toe"],llist=llist)

                                    elif self.left_elbow and 90 <= self.left_elbow <= 159:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["keep your left elbow straight, and hold your toe"],llist=llist)  
                                        
                                    else:
                                        
                                 
                                            # if self.left_shoulder and 0 <= self.left_shoulder <= 14:
                                            #     self.all_methods.reset_after_40_sec()
                                            #     self.all_methods.play_after_40_sec(["please raise your left hand up"],llist=llist)
                                                
                                            # elif self.left_shoulder and 70 <= self.left_shoulder <= 180:
                                            #     self.all_methods.reset_after_40_sec()
                                            #     self.all_methods.play_after_40_sec(["please down your left hand"],llist=llist)
                                                
                                            if self.right_shoulder and 0 <= self.right_shoulder <= 109:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please raise your right hand up"],llist=llist)
                                                
                                            elif self.right_shoulder and 151 <= self.right_shoulder <= 180:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please down your right hand"],llist=llist)
                                                
                                            else:
                                                if self.all_methods.r_elbow_x > self.all_methods.l_hip_x:
                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["keep your right hand in forward side"],llist=llist)  
                                                
                                                else:
                                                    if self.right_elbow and 0 <= self.right_elbow <= 159:
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["keep your right hand straight"],llist=llist)
                                                
                                                    if self.right_knee and 0 <= self.right_knee <= 80:   
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["please keep your right leg straight"],llist=llist)
                                                    
                                                    elif self.right_knee and 81 <= self.right_knee <= 180:
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["move little straight your right leg"],llist=llist)
                                                    
                                                    else:
                                                        if self.head_position != "left":

                                                            self.all_methods.reset_after_40_sec()
                                                            self.all_methods.play_after_40_sec(["please, turn your head to left side"],llist=llist)
                                                        
                                                        else:
                                                            self.initial_count = 0
                                                            return True
                
    def wrong_right(self,frames,llist,height,width):
        
        if not llist:
            return 

        self.all_methods.all_y_values(frames=frames,llist=llist)
        self.all_methods.all_x_values(frames=frames,llist=llist)
        self.all_methods.all_z_values(frames=frames,llist=llist)
        
        self.side = self.side_view_detect(frames=frames,llist=llist)
        if not llist:
            return 
        
        right_knee_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=True)
        
        check_stand = (self.right_hip and 160 <= self.right_hip <= 180 and
                        self.right_knee and 160 <= self.right_knee <= 180 and
                       self.left_knee and 160 <= self.left_knee <= 180)
        
        self.r_touch_hand_foot_x = abs(int(self.all_methods.r_wrist_x - self.all_methods.r_ankle_x))
        self.r_touch_hand_foot_y = abs(int(self.all_methods.r_wrist_y - self.all_methods.r_ankle_y))
        
                
        if not self.check_stand:#and 
            
            if not check_stand:
                # print("hello")
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are not in standing position please be in standing"],llist=llist)
            
            elif check_stand:
                self.check_stand = True     
            
        if self.check_stand and not self.initial_position:
                
            if check_stand:
                initial_voices = ["you are in initial position please do natrajasana","please, fold ,and, lift your right leg up"]
            
                if self.initial_count < len(initial_voices):
                    self.all_methods.reset_after_40_sec()
                    voice = self.all_methods.play_after_40_sec([initial_voices[self.initial_count]],llist=llist)
                    if voice:
                        self.initial_count += 1
                else:
                    self.initial_count = 1
                    
            else:

                if self.right_knee and 0<= self.right_knee <= 24 :
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["your right leg is bending too much , please slight stretch your right leg"],llist=llist)
                else:
                    if self.right_knee and 51 <= self.right_knee <= 180:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["please bend your right leg slight"],llist=llist)
                    
                    else:

                        if self.all_methods.r_elbow_x > self.all_methods.r_hip_x:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please keep your right hand back side, and hold your foot"],llist=llist)

                        else:
                    
                            if self.all_methods.r_ankle_x > self.all_methods.r_hip_x:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please keep your right ankle back side"],llist=llist)

                            else:

                                if self.all_methods.l_wrist_y > self.all_methods.nose_y:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please raise your left hand up"],llist=llist)

                                else:

                                    if (self.right_elbow and 0 <= self.right_elbow <= 159):
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please keep your right hand straight"],llist=llist)

                                    else:

                                        if self.r_touch_hand_foot_x >= 50 and self.r_touch_hand_foot_y >= 100:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please touch your right foot with right hand"],llist=llist)
                # cv.putText()
                                        else:
                                            if not self.first_pose_detected:
                                                if (self.right_knee and 25 <= self.right_knee <= 50 and
                                                    self.right_hip and 160 <= self.right_hip <= 180 and
                                                    self.right_shoulder and 15 <= self.right_shoulder <= 30 and
                                                    self.left_knee and 160 <= self.left_knee <= 180 and
                                                    self.right_elbow and 160 <= self.right_elbow <= 180 ):

                                                    self.all_methods.reset_after_40_sec()

                                                    first_pose = self.all_methods.play_after_40_sec(["good , you completed half pose of this yoga "],llist=llist)
                                                    if first_pose:
                                                        self.first_pose_detected = True

                    if self.first_pose_detected:

                        if self.r_touch_hand_foot_x >= 50 and self.r_touch_hand_foot_y >= 100:

                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please touch your right foot with right hand"],llist=llist)

                        if self.right_hip and 0 <= self.right_hip <= 50:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["You are bending too much. Please bend slightly."],llist=llist)

                        elif self.right_hip and 51 <= self.right_hip <= 109:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["good , you are at correct way ,bend slight"],llist=llist)

                        elif self.right_hip and 141 <= self.right_hip <= 160:   
                            self.hip_count =0 
                            hip_list = ["good, you almost done ,please,lift your right leg little more","good,please, raise your right leg little more"]
                        
                            if self.hip_count < len(hip_list):
                                self.all_methods.reset_after_40_sec()
                                hip_voice = self.all_methods.play_after_40_sec([hip_list[self.hip_count]],llist=llist)
                                if hip_voice:
                                    self.hip_count += 1
                            else:
                                self.hip_count = 0

                        elif self.right_hip and 161 <= self.right_hip <= 180:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["carefully ,and , slowly lift your right leg , with help of your hand"],llist=llist)

                        else:

                            same_axis_for_knee_hip = abs(int(self.all_methods.r_knee_y - self.all_methods.r_hip_y))
                            if same_axis_for_knee_hip >= 100:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please , ensure your upper leg is parallel to ground"],llist=llist)
                                cv.putText(frames,f'axis{str(same_axis_for_knee_hip)}',(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
                            else:
                                if self.all_methods.r_ankle_y > self.all_methods.r_hip_y :
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please raise your right leg ankle , and , cross your hip in up"],llist=llist)

                                elif self.right_knee and 0 <= self.right_knee <= 34:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["Please ,stretch your right leg to form a 90-degree angle."],llist=llist)

                                elif self.right_knee and 35 <= self.right_knee <= 70:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["good, you almost close, move some outward your lower leg."],llist=llist)
                                    
                                elif self.right_knee and 111 <= self.right_knee <= 150:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["good , please make some bend your right leg."],llist=llist)

                                elif self.right_knee and 151 <= self.right_knee <= 180:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please bend your right leg ,and make 90 degrees between "],llist=llist)
                                    
                                else:
                                    
                                    if self.right_elbow and 0 <= self.right_elbow <= 89:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please , keep your right elbow straight, hold your toe"],llist=llist)

                                    elif self.right_elbow and 90 <= self.right_elbow <= 159:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["keep your right elbow straight, and hold your toe"],llist=llist)  
                                    
                                    else:
                                        # if self.right_shoulder and 0 <= self.right_shoulder <= 14:
                                        #     self.all_methods.reset_after_40_sec()
                                        #     self.all_methods.play_after_40_sec(["please raise your right hand up"],llist=llist)
                                            
                                        # elif self.right_shoulder and 70 <= self.right_shoulder <= 180:
                                        #     self.all_methods.reset_after_40_sec()
                                        #     self.all_methods.play_after_40_sec(["please down your right hand"],llist=llist)
                                            
                                        if self.left_shoulder and 0 <= self.left_shoulder <= 109:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please, raise your left hand up"],llist=llist)
                                            
                                        elif self.left_shoulder and 151 <= self.left_shoulder <= 180:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please, down your left hand"],llist=llist)
                                            
                                        else:
                                            if self.all_methods.l_elbow_x < self.all_methods.r_hip_x:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["keep your left hand in forward side"],llist=llist)  
                                            
                                            else:
                                                if self.left_elbow and 0 <= self.left_elbow <= 159:
                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["keep your left hand straight"],llist=llist)

                                                else:
                                            
                                                    if self.left_knee and 0 <= self.left_knee <= 89:   
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["please keep your left leg straight"],llist=llist)
                                                    
                                                    elif self.left_knee and 90 <= self.left_knee <= 159:   
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["move little straight your left leg"],llist=llist)

                                                    else:
                                                        if self.head_position != "right":
                                                            self.all_methods.reset_after_40_sec()
                                                            self.all_methods.play_after_40_sec(["please,turn your head to right side"],llist=llist)
                                                        else:
                                                            # self.initial_count = 0
                                                            return True
                                            
    def check_standing(self,frames,llist,height,width):

        if len(llist) is None:
            return 

        standing_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if standing_position == "standing":
            # print("hello")
            return True 

        elif standing_position != "standing":
            self.all_methods.reset_after_40_sec()
            if self.all_methods.play_after_40_sec(["please be in standing position","   ","this yoga may started in standing position"],llist=llist):
                return False
                                                
    def left_reverse(self,frames):
        
        self.l_r_count = 0
        l_r_voice = ["good, stay in same position ","very good get relax and come to half pose","please come to half pose"]
        
        if (
            self.left_knee and 61 <= self.left_knee <= 120 ):
            
            if self.l_r_count < len(l_r_voice):
                self.all_methods.reset_after_40_sec()
                l_r = self.all_methods.play_after_40_sec([l_r_voice[self.l_r_count]],llist=llist)
                if l_r:
                    self.l_r_count += 1
                    # return True

            else:
                self.l_r_count = 1

        elif (self.left_knee and 25 <= self.left_knee <= 50 and
            self.left_hip and 160 <= self.left_hip <= 180):

            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please come to stand position","please back to initial position"],llist=llist)
                
        elif self.left_knee and 51 <= self.left_knee <= 159:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be straight your left leg"],llist=llist)
         
        elif self.right_knee and 0 <= self.right_knee <= 159:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be straight your right leg"],llist=llist)
        
        elif self.all_methods.is_person_standing_sitting == "standing":    
            self.all_methods.reset_after_40_sec()
            voice = self.all_methods.play_after_40_sec(["good job your yoga is perfectly completed"],llist=llist)
            if voice:
                return True
                
    def right_reverse(self,frames):

        self.r_r_count = 0
        r_r_voice = ["good, stay in same position ","very good get relax and come to half pose","please back to half pose"]
        
        if ( self.right_knee and 61 <= self.right_knee <= 120 ):
            
            if self.r_r_count < len(r_r_voice):
                self.all_methods.reset_after_40_sec()
                r_r = self.all_methods.play_after_40_sec([r_r_voice[self.r_r_count]],llist=llist)
                if r_r:
                    self.r_r_count += 1
                    # return True
                    
            else:
                self.r_r_count = 1

        elif (self.right_knee and 25 <= self.right_knee <= 50 and
            self.right_hip and 160 <= self.right_hip <= 180 ):

            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please come to stand position","please back to initial position"],llist=llist)
                
                
        elif self.right_knee and 51 <= self.right_knee <= 159:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be straight your right leg"],llist=llist)
        
        elif self.left_knee and 0 <= self.left_knee <= 159:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be straight your right leg"],llist=llist)

        if self.all_methods.is_person_standing_sitting == "standing":    
            self.all_methods.reset_after_40_sec()
            voice = self.all_methods.play_after_40_sec(["good job your yoga is perfectly completed"],llist=llist)
            if voice:
                return True
    
    def left_natrajasana_name(self,frames):
        
        correct = (self.left_knee and 71 <= self.left_knee <= 110 and
                   self.right_knee and 160 <= self.right_knee <= 180 and
                   self.left_hip and 110 <= self.left_hip <= 140 and
                   self.left_elbow and 160 <= self.left_elbow <= 180 and
                   self.right_elbow and 160 <= self.right_elbow <= 180 and
                #    self.left_shoulder and 15 <= self.left_shoulder <= 69 and
                   self.right_shoulder and 110 <= self.right_shoulder <= 150 and
                   self.all_methods.r_elbow_x < self.all_methods.l_hip_x and
                   self.all_methods.l_wrist_x > self.all_methods.l_shoulder_x and
                   self.all_methods.l_ankle_x > self.all_methods.l_hip_x and
                   self.all_methods.l_knee_x > self.all_methods.l_hip_x)
        
        if correct:
                return True
        
        
    def right_natrajasana_name(self,frames):
        
        correct = (self.right_knee and 71 <= self.right_knee <= 110 and
                   self.left_knee and 160 <= self.left_knee <= 180 and
                   self.right_hip and 110 <= self.right_hip <= 140 and
                   self.right_elbow and 160 <= self.right_elbow <= 180 and
                   self.left_elbow and 160 <= self.left_elbow <= 180 and
                #    self.right_shoulder and 15 <= self.right_shoulder <= 69 and
                   self.left_shoulder and 110 <= self.left_shoulder <= 150 and
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
        
        img = cv.imread("images/natrajasana.jpg")

        if not flag:
            
            detect.pose_positions(frames,draw = False)
            llist = detect.pose_landmarks(frames,draw=False)
            # detect.natrajasana(frames,llist,l_elbow=(12,14,16), l_hip=(12,24,26), l_knee=(24,26,28), l_shoulder=(14,12,24),r_elbow=(11,13,15),r_hip= (11,23,25),r_knee= (23,25,27), r_shoulder=(13,11,23),l_draw=True ,r_draw=True)

            if not check_stand:
                stand = detect.check_standing(frames=frames,llist=llist,height=height,width=width)
                if stand:
                    check_stand = True
                    
            elif check_stand:
                side_view = detect.side_view_detect(frames,llist=llist)

                if side_view == "forward":
                    all_methods.reset_after_40_sec()
                    all_methods.play_after_40_sec(["please turn total body ,  either left, or, right side"],llist=llist)

                elif side_view == "right":
                    
                    detect.natrajasana(frames,llist,r_elbow=(12,14,16), r_hip=(12,24,26), r_knee=(24,26,28), r_shoulder=(14,12,24),l_elbow=(11,13,15),l_hip= (11,23,25),l_knee= (23,25,27), l_shoulder=(13,11,23),l_draw=False ,r_draw=True,r_v_draw=True,l_v_draw=False)
                    if not checking_wrong:
                        detect.side_view_detect(frames=frames,llist=llist)
                        wrong_right = detect.wrong_right(frames=frames,llist=llist,height=height,width=width)
                        if wrong_right:
                            checking_wrong = True
                    
                    if checking_wrong and not reverse_yoga:
                        correct = detect.right_natrajasana_name(frames=frames)
                        if correct:
                            reverse_yoga = True
                        
                    if reverse_yoga:
                            reverse = detect.right_reverse(frames=frames)
                            if reverse:
                                flag = True

                elif side_view =="left":
                    
                    detect.natrajasana(frames,llist,r_elbow=(12,14,16), r_hip=(12,24,26), r_knee=(24,26,28), r_shoulder=(14,12,24),l_elbow=(11,13,15),l_hip= (11,23,25),l_knee= (23,25,27), l_shoulder=(13,11,23),l_draw=True ,r_draw=False,l_v_draw=True,r_v_draw=False)
                    if not checking_wrong:
                        detect.side_view_detect(frames=frames,llist=llist)
                        wrong_right = detect.wrong_left(frames=frames,llist=llist,height=height,width=width)
                        if wrong_right:
                            checking_wrong = True
                    
                    if checking_wrong and not reverse_yoga:
                        correct = detect.right_natrajasana_name(frames=frames)
                        if correct:
                            # flag = True
                            reverse_yoga = True
                        
                    if reverse_yoga:
                            reverse = detect.right_reverse(frames=frames)
                            if reverse:
                                flag = True

        cv.imshow("video",frames)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()   