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

class matyasana:
     
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
        self.l_detect_45 = False
        self.r_detect_45 = False
        self.l_knee_up = False
        self.r_knee_up = False
        self.l_check_leg_raise = False
        self.r_check_leg_raise = False
        self.l_leg_on_thigh = False
        self.r_leg_on_thigh = False
        self.upper_pose_detected = False
        self.ground_pose_detected = False

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

    def left_matyasana(self, frames,llist, elbow, hip, knee,shoulder,right_knee1,right_elbow,draw =True):

        self.head = self.head_pose_estimator.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose_estimator.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose_estimator.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose_estimator.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        # self.left_wrist_y = llist[15][2]
        # self.self.all_methods.l_knee_y = llist[25][2]
        # self.left_knee_min_y = self.self.all_methods.l_knee_y - 80
        # self.ground_left = self.all_methods.ground_distance_left(frames=frames,lmlist=llist)
        # self.ground_left_min = self.ground_left - 50
        
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
        
        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.right_knee1,self.head_position #self.self.all_methods.l_knee_y,self.left_wrist_y,self.ground_left,self.ground_left_min

    def right_matyasana(self,frames,llist,elbow, hip, knee, shoulder,left_knee1,left_elbow,draw=True):

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


        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,left_knee1,self.head_position#self.self.all_methods.r_knee_y,self.right_wrist_y,self.ground_right,self.ground_right_min
    
    def wrong_left(self,frames,llist,height,width):

        if len(llist) == 0:
            return
        count = 0

        nose_y = llist[0][2]
        # self.all_methods.l_ankle_y = llist[27][2]
        # self.all_methods.r_ankle_y = llist[28][2]
        hip_y = max(llist[23][2],llist[24][2])
        # self.all_methods.l_knee_y = llist[25][2]
        # self.all_methods.r_knee_y = llist[26][2]

        self.all_methods.all_x_values(frames=frames,llist=llist)

        left_leg_up = (self.all_methods.l_ankle_y < self.all_methods.r_knee_y)
        right_leg_up = (self.all_methods.r_ankle_y < self.all_methods.l_knee_y)

        left_ankle_on_thigh = (self.all_methods.l_ankle_y > self.all_methods.r_knee_y)
        right_ankle_on_thigh = (self.all_methods.r_ankle_y > self.all_methods.l_knee_y)

        right_thigh_y = (self.all_methods.r_knee_y + self.all_methods.r_hip_y) // 2
        right_thigh_x = (self.all_methods.r_knee_x + self.all_methods.r_hip_x) // 2

        distance = (math.sqrt((self.all_methods.l_ankle_x - right_thigh_x) ** 2 + (self.all_methods.l_ankle_y - right_thigh_y) ** 2))
        left_ankle_touches_to_right_thigh = ()
        right_ankle_touches_to_left_thigh = ()

        left_shoulder_x,left_shoulder_y,left_shoulder_z = llist[11][1:]
        right_shoulder_x,right_shoulder_y,right_shoulder_z = llist[12][1:]
        tolerance = abs(int(left_shoulder_z - right_shoulder_z))


        l_legs_on_ground = abs(int(self.all_methods.l_knee_y - hip_y))
        r_legs_on_ground = abs(int(self.all_methods.r_knee_y - hip_y))

        #right slope condition
        self.right_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=False)
        self.right_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=False)
        self.right_hip_ankle = self.all_methods.slope(frames=frames,lmlist=llist,point1=26,point2=28,height=height,width=width,draw=False)

        # flat_position = (self.left_hip_knee and 0 <= self.left_hip_knee <= 15 and 
        #                  self.left_hip_ankle and 0 <= self.left_hip_ankle <= 15)

        r_45 = ( 30 <= self.right_knee1 <= 50)
        l_45 = (30 <= self.left_knee <= 50)

        if not self.left_hip and not self.left_elbow and not self.left_knee and not self.left_shoulder and not self.right_knee1 and not left_shoulder_z and not right_shoulder_z:
            return None
        
        #check sleep POSITION
        sleeping_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if not self.check_sleep_position and sleeping_position == "sleeping":
            
            self.check_sleep_position = True
            self.check_initial_position = True
            return 

        elif not self.check_initial_position and sleeping_position != "sleeping":
            self.all_methods.reset_voice()
            self.all_methods.play_voice(["please be in sleeping position","   ","this yoga may started in sleeping position"],llist=llist)
            return 
        
        #CHECK FLAT POSITION
        if self.check_sleep_position and not self.start_exercise:

            count = 0

            if tolerance <= 40:

                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are in side sleep position","please sleep in flat position"],llist=llist)

            elif not self.start_exercise and left_shoulder_z < right_shoulder_z and tolerance > 40:

                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["Lie on your back with legs extended"],llist=llist)


            elif not self.start_exercise and left_shoulder_z > right_shoulder_z and tolerance > 40:
                self.start_exercise = True

        #CHECK INITIAL POSITION
        elif self.start_exercise:

            if left_shoulder_z > right_shoulder_z and tolerance > 40:

                if count == 0:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are in initial position and start yoga mathyasana"],llist=llist)
                    count += 1
                
                elif count == 1:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please start yoga and fold your right leg around 45 degress"],llist=llist)
                    count = 1

            else:
                #check legs in 45 degress or not
                if not self.r_detect_45:

                    if self.right_knee1:
                        if self.right_knee1 and 0 <= self.right_knee1 <= 29:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your right leg around 45 degress outwards"],llist=llist)

                        elif self.right_knee1 and 51 <= self.right_knee1 <= 180:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please fold your legs inwards 45 degress"],llist=llist)

                        elif r_45:
                            self.r_detect_45 = True

                elif self.r_detect_45 and not self.l_detect_45:

                    if self.left_knee and 0 <= self.left_knee <= 29:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["keep your left leg around 45 degress outwards"],llist=llist)

                    elif self.left_knee and 51 <= self.left_knee <= 180:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["please fold your legs inwards 45 degress"],llist=llist)

                    elif l_45:
                        self.l_detect_45 = True

                #CHECK LEFT LEG RAISE
                elif self.l_detect_45 and not self.l_check_leg_raise:

                    if l_45:

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["lift your left lower leg up"],llist=llist)

                    elif not left_leg_up and not l_45:

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["lift your left leg ankle above than right knee"],llist=llist)

                    elif left_leg_up and not l_45:

                        self.l_check_leg_raise = True

                #CHECK UPPER POSE DETECTED
                elif self.l_check_leg_raise and not self.upper_pose_detected:

                    if  not self.r_check_leg_raise:

                        if left_leg_up:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your left leg on thigh"],llist=llist)

                        #CHECK LEFT LEG IS CORRECT POSITION
                        if self.all_methods.l_ankle_y < hip_y and self.all_methods.r_knee_y < self.all_methods.l_ankle_y and self.all_methods.l_knee_y < hip_y:
                            
                            self.r_check_leg_raise = True

                    if self.r_check_leg_raise:
                        #CHECK LEFT LEG IS IN UP
                        if l_legs_on_ground <= 100:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["your left leg is on ground ,please keep it little above than your hip"],llist=llist)

                        elif self.all_methods.l_ankle_y >= hip_y:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please keep your left ankle on thigh"],llist=llist)

                        #CHECK RIGHT LEG IS IN 45 DEGREES
                        if r_45:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["Please raise your right leg slightly above your hip."],llist=llist)

                        #CHECK RIGHT LEG IS IN UP
                        elif r_legs_on_ground <= 100:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["your right leg is on ground please keep it little above than your hip"],llist=llist)

                        elif self.all_methods.r_ankle_y >= hip_y:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please keep your right ankle on thigh"],llist=llist)

                        #CHECK RIGHT LEG IN CORRECT POSITION
                        elif self.all_methods.r_ankle_y < hip_y and self.all_methods.l_knee_y < self.all_methods.r_ankle_y and self.all_methods.l_knee_y < hip_y:
                            self.upper_pose_detected = True

                elif self.upper_pose_detected:

                    #CHECK GROUND POSE DETECTED

                    if not self.ground_pose_detected and r_legs_on_ground > 100 and l_legs_on_ground > 100:

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["keep your legs on ground and do not get relax your legs"],llist=llist)

                    elif not self.ground_pose_detected and r_legs_on_ground <= 50:

                        self.ground_pose_detected = True

                    elif self.ground_pose_detected:

                        #CHECK CORRECT POSE WHILE LIFT ON GROUND using z values
                        if self.all_methods.r_knee_z > self.all_methods.r_ankle_z:

                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["your right ankle must be in above the left hip"],llist=llist)

                        elif self.all_methods.l_knee_z < self.all_methods.l_ankle_z:

                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["your left ankle must be in above the right hip"],llist=llist)

                        else:
                            #CHECK WHILE CHEST IS LIFT OR NOT
                            if self.right_shoulder_hip and 0 <= self.right_shoulder_hip <= 10:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["with support of arms and elbows lift your chest slightly"],llist=llist)

                            elif self.right_shoulder_hip and 40 <= self.right_shoulder_hip <= 90:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["with proper support of arms and elbows down your chest slightly"],llist=llist)

                            else:
                                #CHECK WHILE ELBOW IS IN CORRECT OR NOT
                                if self.right_elbow and 160 <= self.right_elbow <= 180:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["take proper support of elbows"],llist=llist)

                                elif self.right_elbow and 0 <= self.right_elbow <= 120:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["take proper support of elbows"],llist=llist)

                                else:
                                    #CHECK FACE IN RIGHT OR NOT
                                    if self.head_position != 'Right':

                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["touch your head on ground and raise up see right side"],llist=llist)

                                    else:
                                        #check  WRIST IS TOUCH TO TOE OR NOT
                                        r_touch_toe = abs(int(self.all_methods.r_toe_x - self.all_methods.l_wrist_x))
                                        if r_touch_toe >= 50 and self.all_methods.r_hip_y < self.all_methods.r_wrist_y:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["hold your right toe with left hand"],llist=llist)

                                        l_toe_touch = abs(int(self.all_methods.l_toe_x - self.all_methods.r_wrist_x))

                                        if l_toe_touch >= 50 and self.all_methods.r_hip_y < self.all_methods.l_wrist_y:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["hold your left toe with right hand"],llist=llist)

                                        else:
                                            
                                            r_hip_differ = abs(int(self.all_methods.r_hip_y - self.all_methods.r_knee_y))

                                            if r_hip_differ >= 70 :
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["sleep on head and hip and knee only"],llist=llist)

                                            else:
                                                return True
                                        
                                        

    def wrong_right(self,frames,llist,height,width):

        if len(llist) == 0:
            return
        count = 0

        nose_y = llist[0][2]
        hip_y = max(llist[23][2],llist[24][2])

        self.all_methods.all_x_values(frames=frames,llist=llist)

        left_leg_up = (self.all_methods.l_ankle_y < self.all_methods.r_knee_y)
        right_leg_up = (self.all_methods.r_ankle_y < self.all_methods.l_knee_y)

        left_ankle_on_thigh = (self.all_methods.l_ankle_y > self.all_methods.r_knee_y)
        right_ankle_on_thigh = (self.all_methods.r_ankle_y > self.all_methods.l_knee_y)

        right_thigh_y = (self.all_methods.r_knee_y + self.all_methods.r_hip_y) // 2
        right_thigh_x = (self.all_methods.r_knee_x + self.all_methods.r_hip_x) // 2

        distance = (math.sqrt((self.all_methods.l_ankle_x - right_thigh_x) ** 2 + (self.all_methods.l_ankle_y - right_thigh_y) ** 2))
        left_ankle_touches_to_right_thigh = ()
        right_ankle_touches_to_left_thigh = ()

        left_shoulder_x,left_shoulder_y,left_shoulder_z = llist[11][1:]
        right_shoulder_x,right_shoulder_y,right_shoulder_z = llist[12][1:]
        tolerance = abs(left_shoulder_z - right_shoulder_z)
        l_legs_on_ground = (self.all_methods.l_knee_y - hip_y)
        r_legs_on_ground = (self.all_methods.r_knee_y - hip_y)

        #left slope condition
        self.left_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width,draw=False)
        self.left_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=False)
        self.left_hip_ankle = self.all_methods.slope(frames=frames,lmlist=llist,point1=25,point2=27,height=height,width=width,draw=False)

        # flat_position = (self.left_hip_knee and 0 <= self.left_hip_knee <= 15 and 
        #                  self.left_hip_ankle and 0 <= self.left_hip_ankle <= 15)

        r_45 = ( 30 <= self.right_knee1 <= 50)
        l_45 = (30 <= self.left_knee <= 50)

        if not self.left_hip and not self.left_elbow and not self.left_knee and not self.left_shoulder and not self.right_knee1 and not left_shoulder_z and not right_shoulder_z:
            return None
        
        #check sleep POSITION
        sleeping_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if not self.check_sleep_position and sleeping_position == "sleeping":
            
            self.check_sleep_position = True
            self.check_initial_position = True
            return 

        elif not self.check_initial_position and sleeping_position != "sleeping":
            self.all_methods.reset_voice()
            self.all_methods.play_voice(["please be in sleeping position","   ","this yoga may started in sleeping position"],llist=llist)
            return 
        
        #CHECK FLAT POSITION
        if self.check_sleep_position:

            count = 0

            if tolerance <= 40:

                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are in side sleep position","please sleep in flat position"],llist=llist)

            elif not self.start_exercise and left_shoulder_z < right_shoulder_z and tolerance > 40:

                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["Lie on your back with legs extended"],llist=llist)


            elif not self.start_exercise and left_shoulder_z < right_shoulder_z and tolerance > 40:
                self.start_exercise = True

        #CHECK INITIAL POSITION
        elif self.start_exercise:

            if left_shoulder_z < right_shoulder_z and tolerance > 40:

                if count == 0:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are in initial position and start yoga mathyasana"],llist=llist)
                    count += 1
                
                elif count == 1:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please start yoga and fold your right leg around 45 degress"],llist=llist)
                    count = 1

            else:
                #check legs in 45 degress or not

                if not self.r_detect_45:

                    if self.right_knee1:
                        if self.right_knee1 and 0 <= self.right_knee1 <= 29:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your right leg around 45 degress outwards"],llist=llist)

                        elif self.right_knee1 and 51 <= self.right_knee1 <= 180:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please fold your legs inwards 45 degress"],llist=llist)

                        elif r_45:
                            self.r_detect_45 = True

                elif self.r_detect_45 and not self.l_detect_45:

                    if self.left_knee and 0 <= self.left_knee <= 29:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["keep your left leg around 45 degress outwards"],llist=llist)

                    elif self.left_knee and 51 <= self.left_knee <= 180:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["please fold your legs inwards 45 degress"],llist=llist)

                    elif l_45:
                        self.l_detect_45 = True

                #CHECK RIGHT LEG RAISE
                elif r_45 and not self.r_check_leg_raise:

                    if r_45:

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["lift your RIGHT lower leg up"],llist=llist)

                    elif not right_leg_up and not r_45:

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["lift your RIGHT leg ankle above than right knee"],llist=llist)

                    elif right_leg_up and not r_45:

                        self.r_check_leg_raise = True

                #CHECK UPPER POSE DETECTED
                elif self.r_check_leg_raise and not self.upper_pose_detected:

                    if  not self.l_check_leg_raise:

                        if right_leg_up:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your right leg on thigh"],llist=llist)

                        #CHECK LEFT LEG IS CORRECT POSITION
                        if self.all_methods.r_ankle_y < hip_y and self.all_methods.l_knee_y < self.all_methods.r_ankle_y and self.all_methods.r_knee_y < hip_y:
                            
                            self.r_check_leg_raise = True

                    if self.r_check_leg_raise:
                        #CHECK RIGHT LEG IS IN UP
                        if self.all_methods.r_knee_y >= hip_y:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["your right leg is on ground please keep it little above than your hip"],llist=llist)

                        elif self.all_methods.r_ankle_y >= hip_y:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please keep your right ankle on thigh"],llist=llist)

                        #CHECK LEFT LEG IS IN 45 DEGREES
                        if l_45:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["Please raise your left leg slightly above your hip."],llist=llist)

                        #CHECK LEFT LEG IS IN UP
                        elif self.all_methods.l_knee_y >= hip_y:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["your left leg is on ground please keep it little above than your hip"],llist=llist)

                        elif self.all_methods.l_ankle_y >= hip_y:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please keep your left ankle on thigh"],llist=llist)

                        #CHECK LEFT LEG IN CORRECT POSITION
                        elif self.all_methods.l_ankle_y < hip_y and self.all_methods.r_knee_y < self.all_methods.l_ankle_y and self.all_methods.r_knee_y < hip_y:
                            self.upper_pose_detected = True

                elif self.upper_pose_detected:

                    #CHECK GROUND POSE DETECTED

                    if not self.ground_pose_detected and self.all_methods.l_knee_y < hip_y and self.all_methods.r_knee_y < hip_y:

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["keep your legs on ground and do not get relax your legs"],llist=llist)

                    elif not self.ground_pose_detected and r_legs_on_ground <= 50:

                        self.ground_pose_detected = True

                    elif self.ground_pose_detected:

                        #CHECK CORRECT POSE WHILE LIFT ON GROUND
                        if self.all_methods.l_knee_z > self.all_methods.l_ankle_z:

                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["your left leg ankle must be in above the left hip"],llist=llist)

                        elif self.all_methods.r_knee_z < self.all_methods.r_ankle_z:

                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["your right leg ankle must be in above the right hip"],llist=llist)

                        else:
                            #CHECK WHILE SHOULDER IS LIFT OR NOT
                            if self.left_shoulder_hip and 0 <= self.left_shoulder_hip <= 10:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["with support of arms and elbows lift your chest slightly"],llist=llist)

                            elif self.left_shoulder_hip and 40 <= self.left_shoulder_hip <= 90:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["with proper support of arms and elbows down your chest slightly"],llist=llist)
                            else:

                                # #CHECK LEFT HIP ANGLE
                                # if self.left_hip and 0 <= self.left_hip <= 90:
                                #     self.all_methods.reset_after_40_sec()
                                #     self.all_methods.play_after_40_sec(["your legs must be in padmasanaa"],llist=llist)

                                # elif self.left_hip and 140 <= self.left_hip <= 180:
                                #     self.all_methods.reset_after_40_sec()
                                #     self.all_methods.play_after_40_sec(["your legs must be in padmasana"],llist=llist)
                                #CHECK WHILE ELBOW IS IN CORRECT OR NOT
                                if self.left_elbow and 160 <= self.left_elbow <= 180:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["take proper support of elbows"],llist=llist)

                                elif self.left_elbow and 0 <= self.left_elbow <= 120:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["take proper support of elbows"],llist=llist)

                                else:
                                    #CHECK FACE IN left OR NOT
                                    if self.head_position != 'Left':

                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["touch your head on ground and raise up see left side"],llist=llist)

                                    else:
                                        #check  WRIST IS TOUCH TO TOE OR NOT
                                        r_touch_toe = abs(int(self.all_methods.l_toe_x - self.all_methods.r_wrist_x))
                                        if r_touch_toe >= 50 and self.all_methods.r_hip_y < self.all_methods.r_wrist_y:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["hold your left toe with left hand"],llist=llist)

                                        l_toe_touch = abs(int(self.all_methods.r_toe_x - self.all_methods.l_wrist_x))

                                        if l_toe_touch >= 50 and self.all_methods.r_hip_y < self.all_methods.l_wrist_y:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["hold your right toe with right hand"],llist=llist)

                                        else:

                                            l_hip_differ = abs(int(self.all_methods.l_hip_y - self.all_methods.l_knee_y))

                                            if l_hip_differ >= 70 :
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["sleep on head and hip and knee only"],llist=llist)

                                            else:
                                                return True
                                        

    def left_reverse_position(self,frames,llist):

        count = 0

        if self.right_shoulder_hip and 11 <= self.right_shoulder_hip <= 30:

            if count == 0:
                self.all_methods.reset_voice()
                self.all_methods.play_voice(["stay in same position and wait for other instruction"],llist=llist)
                count += 1

            elif count == 1:
                self.all_methods.reset_voice()
                self.all_methods.play_voice(["stay in same position and wait for other instruction"],llist=llist)
            
        elif self.right_shoulder_hip and 0 <= self.right_shoulder_hip <= 15:

            self.all_methods.reset_voice()
            self.all_methods.play_voice(["good job you complete perfect , get relax and out off that pose"],llist=llist)
            return True
        pass

    def right_reverse_position(self,frames,llist):

        count = 0

        if self.left_shoulder_hip and 11 <= self.left_shoulder_hip <= 30:

            if count == 0:
                self.all_methods.reset_voice()
                self.all_methods.play_voice(["stay in same position and wait for other instruction"],llist=llist)
                count += 1

            elif count == 1:
                self.all_methods.reset_voice()
                self.all_methods.play_voice(["stay in same position and wait for other instruction"],llist=llist)
            
        elif self.left_shoulder_hip and 0 <= self.left_shoulder_hip <= 15:

            self.all_methods.reset_voice()
            self.all_methods.play_voice(["good job you complete perfect , get relax and out off that pose"],llist=llist)
            return True
        pass

    def left_matyasana_name(self,frames):

        # correct = (
        #     self.right_shoulder_hip and 11 <= self.right_shoulder_hip <= 30 and
        #     self.head_position and self.head_position == "Right" and
        #     self.right_elbow and 10 <= self.right_elbow <= 40 and
        #     self.left_elbow1 and 10 <= self.left_elbow1 <= 40 and
        #     self.right_hip and 80 <= self.right_hip <= 120 and
        #     self.right_hip_knee and 30 <= self.right_hip_knee <= 50

        # )
        # if correct :
        #     return True
        pass
    def right_matyasana_name(self,frames):

        # correct = (
        #     self.left_shoulder_hip and 11 <= self.left_shoulder_hip <= 30 and
        #     self.head_position and self.head_position == "Left" and
        #     self.left_elbow and 10 <= self.left_elbow <= 40 and
        #     self.left_elbow1 and 10 <= self.left_elbow1 <= 40 and
        #     self.left_hip and 80 <= self.left_hip <= 120 and
        #     self.left_hip_knee and 30 <= self.left_hip_knee <= 50

        # )

        # if correct :
        #     return True
        pass

def main():

    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    global  llist
    detect = matyasana()
    all_methods = allmethods()
    ref_video = cv.VideoCapture("second videos/matyasana.mp4")
    flag = False
    ready_for_exercise = False
    reverse_yoga = False
    checking_wrong = False
    checkTime_for_holdingBody = 0
    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break

        # ret_ref, ref_frame = ref_video.read()
        # if not ret_ref:
        #     ref_video.set(cv.CAP_PROP_POS_FRAMES, 0)  # loop video
        #     ret_ref, ref_frame = ref_video.read()

        # # Resize reference video frame and place it at top-left
        # if ret_ref:
        #     ref_frame = cv.resize(ref_frame, (400, 300))
        #     h, w, _ = ref_frame.shape
        #     frames[0:h, 0:w] = ref_frame  # Overlay in corner
        img = cv.imread("second_images/matyasana.jpg")
        img1 = cv.resize(img, None, fx=3.0, fy=3.0, interpolation=cv.INTER_LINEAR)
        
        if not flag:

            detect.pose_positions(img1,draw=False)
            llist = detect.pose_landmarks(img1,False)
            nose_x = llist[0][1]
            side_view = allmethods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION='head',head=nose_x)

            if side_view == "left":
                detect.left_matyasana(frames=img1,llist=llist,elbow=(11,13,15), hip=(11,23,25), knee=(23,25,27), shoulder=(13,11,23),right_knee1=(24,26,28),right_elbow=(12,14,16),draw=True)
                # wrong_left = detect.wrong_matyasana_left(frames=frames,llist=llist,height=height,width=width)
                if not checking_wrong:
                    wrong_left = detect.wrong_left(frames=img1,llist=llist,height=height,width=width)
                    if wrong_left:
                        checking_wrong = True
                
                if checking_wrong:
                    # correct = detect.left_matyasana_name(frames=frames)
                    if not ready_for_exercise and not flag:
                        correct = detect.left_matyasana_name(frames=img1)
                        if correct:
                            ready_for_exercise = True
                            reverse_yoga = True

                if reverse_yoga  and not flag:
                    left_reverse = detect.left_reverse_position(frames=img1,llist=llist)
                    # ready_for_exercise = False
                    if left_reverse:
                        flag = True
        

            #Right Side
            elif side_view == "right":
                detect.right_matyasana(frames=img1,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee1=(23,25,27),left_elbow=(11,13,15),draw=True)                        
                
                # correct = detect.right_matyasana_name(frames=frames)
                if not checking_wrong:
                    wrong_right = detect.wrong_right(frames=img1,llist=llist,height=height,width=width)
                    if wrong_right:
                        checking_wrong = True

                if checking_wrong:
                    if not ready_for_exercise and not flag:
                        correct = detect.right_matyasana_name(frames=img1)
                        if correct:
                            ready_for_exercise = True
                            reverse_yoga = True

                    if reverse_yoga and not flag:
                        right_reverse = detect.right_reverse_position(frames=img1,llist=llist)
                        if right_reverse:
                            flag = True
                        
        cv.imshow("video",img1)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main() 