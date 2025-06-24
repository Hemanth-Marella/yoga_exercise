import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math

# import pyttsx3 as pyt
from threading import Thread
from allMethods import allmethods
from face_detect import HeadPoseEstimator
from voiceModule import VoicePlay
from body_position import bodyPosition

class usthrasana:
     
    def __init__(self,mode = False,mindetectconf=0.5,mintrcackconf=0.5):
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

        self.lpslist = []

        self.check_initial_position = False
        self.start_exercise = False

        self.angle =0
        self.all_methods = allmethods()
        self.voice = VoicePlay()
        self.body_position = bodyPosition()
        self.head_pose_estimator = HeadPoseEstimator()
        self.round = True

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        self.head_pose = HeadPoseEstimator()     

    def pose_positions(self,frames,draw = True):

        imgRB = cv.cvtColor(frames,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frames,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

    def pose_landmarks(self,frames,draw = True):

        self.lpslist =[]

        if self.results.pose_landmarks:

            self.h,self.w,cl = frames.shape

            for id,poselms in enumerate(self.results.pose_landmarks.landmark):
                px,py ,pz= int(poselms.x * self.w) , int(poselms.y * self.h),int(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lpslist

    def left_usthrasana(self, frames,llist, elbow, hip, knee,shoulder,right_knee1,right_elbow,draw =True):

        self.head = self.head_pose_estimator.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose_estimator.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose_estimator.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose_estimator.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        # self.left_wrist_y = llist[15][2]
        # self.left_knee_y = llist[25][2]
        # self.left_knee_min_y = self.left_knee_y - 80
        # self.ground_left = self.all_methods.ground_distance_left(frames=frames,lmlist=llist)
        # self.ground_left_min = self.ground_left - 50
        
        self.left_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
        self.left_hip, hip_coords = self.all_methods.calculate_angle(frames=frames,points= hip,lmList=llist)
        self.left_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)       
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.right_knee1, right_knee1_coords = self.all_methods.calculate_angle(frames=frames,points= right_knee1,lmList=llist)
        self.right_elbow1, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=right_elbow,lmList=llist)

        cv.putText(frames,f'l_elbow{str(self.left_elbow)}',(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
        cv.putText(frames,f'l_hip{str(self.left_hip)}',(10,80),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
        cv.putText(frames,f'l_knee{str(self.left_knee)}',(10,120),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
        cv.putText(frames,f'l_shoulder{str(self.left_shoulder)}',(10,160),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
        cv.putText(frames,f'r_knee1{str(self.right_knee1)}',(10,200),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
        cv.putText(frames,f'r_elbow{str(self.right_elbow1)}',(10,240),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)

        # if draw:
        #     if elbow_coords:
        #         cv.putText(frames, str(int(self.left_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        #     if hip_coords:
        #         cv.putText(frames, str(int(self.left_hip)), (hip_coords[2]+10, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        #     if knee_coords:
        #         cv.putText(frames, str(int(self.left_knee)), (knee_coords[2]+10, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        #     if points_cor8:
        #         cv.putText(frames, str(int(self.left_shoulder)), (points_cor8[2]+10, points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        #     if right_knee1_coords:
        #         cv.putText(frames, str(int(self.right_knee1)), (right_knee1_coords[2]-20, right_knee1_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.right_knee1,self.head_position #self.left_knee_y,self.left_wrist_y,self.ground_left,self.ground_left_min

    def right_usthrasana(self,frames,llist,elbow, hip, knee, shoulder,left_knee1,left_elbow,draw=True):

        self.head = self.head_pose_estimator.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose_estimator.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose_estimator.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose_estimator.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
    
        if draw:
        
            self.right_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
            self.right_hip, hip_coords = self.all_methods.calculate_angle(frames=frames, points=hip,lmList=llist)
            self.right_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)
            self.right_shoulder,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
            self.left_knee1, left_knee1_coords = self.all_methods.calculate_angle(frames=frames,points= left_knee1,lmList=llist)
            self.left_elbow1, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=left_elbow,lmList=llist)


        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,left_knee1,self.head_position#self.right_knee_y,self.right_wrist_y,self.ground_right,self.ground_right_min
    
    def wrong_left(self,frames,llist,height,width):

        left_shoulder_x,left_shoulder_y,left_shoulder_z = llist[11][1:]
        right_shoulder_x,right_shoulder_y,right_shoulder_z = llist[12][1:]
        tolerance = abs(left_shoulder_z - right_shoulder_z)

        #left slope condition
        self.left_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width,draw=False)
        self.left_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=False)
        self.left_hip_ankle = self.all_methods.slope(frames=frames,lmlist=llist,point1=25,point2=27,height=height,width=width,draw=False)

        flat_position = (self.left_hip_knee and 0 <= self.left_hip_knee <= 15 and 
                         self.left_hip_ankle and 0 <= self.left_hip_ankle <= 15)

        if not self.left_hip and not self.left_elbow and not self.left_knee and not self.left_shoulder and not self.right_knee1 and not left_shoulder_z and not right_shoulder_z:
            return None
        

        sleeping_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if not self.check_sleep_position and sleeping_position == "sleeping":
            
            self.check_sleep_position = True
            self.check_initial_position = True
            return 

        elif not self.check_initial_position and sleeping_position != "sleeping":
            self.all_methods.reset_voice()
            self.all_methods.play_voice(["please be in sleeping position","   ","this yoga may started in sleeping position"],llist=llist)
            return 
        
        if self.check_sleep_position:

            count = 0

            if tolerance <= 40:

                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are in side sleep position","please sleep in flat position"],llist=llist)

            elif not self.start_exercise and left_shoulder_z < right_shoulder_z and tolerance > 40 and flat_position:

                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["Lie on your back with legs extended"],llist=llist)


            elif not self.start_exercise and left_shoulder_z > right_shoulder_z and tolerance > 40 and flat_position:
                self.start_exercise = True

        
        elif self.start_exercise:

            if left_shoulder_z > right_shoulder_z and tolerance > 40 and flat_position:

                if count == 0:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are in initial position and start yoga mathyasana"],llist=llist)
                    count += 1
                
                elif count == 1:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please start yoga and fold your legs around 45 degress"],llist=llist)

            else:

                #check legs in 45 degress or not

                if self.left_knee and 0 <= self.left_knee <= 29:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["keep your left leg around 45 degress outwards"],llist=llist)

                elif self.left_knee and 51 <= self.left_knee <= 180:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please fold your legs inwards 45 degress"],llist=llist)

                else:
                    if self.right_knee1:
                        if self.right_knee1 and 0 <= self.right_knee1 <= 29:

            