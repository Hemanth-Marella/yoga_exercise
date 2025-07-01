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
from abstract_class import yoga_exercise

class usthrasana(yoga_exercise):
     
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
        self.l_count = 0
        self.r_count = 0
        self.l_r_count = 0
        self.r_r_count = 0

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
                px,py ,pz= (poselms.x * self.w) , (poselms.y * self.h),(poselms.z)

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

        if draw:

            cv.putText(frames,f'l_elbow{str(int(self.left_elbow))}',(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_hip{str(int(self.left_hip))}',(10,80),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_knee{str(int(self.left_knee))}',(10,120),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_shoulder{str(int(self.left_shoulder))}',(10,160),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_knee{str(int(self.right_knee1))}',(10,200),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_elbow{str(int(self.right_elbow1))}',(10,240),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)

       
        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.right_knee1,self.right_elbow1,self.head_position #self.left_knee_y,self.left_wrist_y,self.ground_left,self.ground_left_min

    def right_usthrasana(self,frames,llist,elbow, hip, knee, shoulder,left_knee1,left_elbow,draw=True):

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
            cv.putText(frames,f'l_knee{str(self.left_knee1)}',(10,200),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_elbow{str(self.left_elbow1)}',(10,240),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)

        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,left_knee1,self.head_position,self.left_elbow1#self.right_knee_y,self.right_wrist_y,self.ground_right,self.ground_right_min
    

    # def slope(self,frames,llist,height,width):

    #     if llist is None:
    #         return None

    #     self.under_leg_slope = self.all_methods.slope(frames=frames,lmlist=llist,point1=25,point2=27,height=height,width=width,draw=True)
        
    #     return self.under_leg_slope
    

    def wrong_left(self,frames,llist):

        count = 0

        self.all_methods.all_x_values(frames=frames,llist=llist)

        stand_on_knee = (self.left_knee and 80 <= self.left_knee <= 100 and
                         self.right_knee1 and 80 <= self.right_knee1 <=100 and
                         self.left_hip and 160 <= self.left_hip <= 180)
        
        hands_on_hip = (self.left_knee and 80 <= self.left_knee <= 100 and
                         self.right_knee1 and 80 <= self.right_knee1 <=100 and
                         self.left_elbow and 80 <= self.left_elbow <= 100 and
                         self.right_elbow1 and 80 <= self.right_elbow1 <= 100)
        
        self.head_value_x= llist[0][1]
        self.toe_value_x,self.toe_value_y,self.toe_value_z = llist[29][1:]
        self.r_toe_x,self.r_toe_y,self.r_toe_z = llist[30][1:]
        self.l_hand_y = llist[15][2]
        self.r_hand_y = llist[16][2]
        self.l_shoulder_value_x = llist[11][1]

        #check legs is in forward side
        check_left_ankle = (self.all_methods.l_hip_x < self.all_methods.l_ankle_x)
        check_right_ankle = (self.all_methods.l_hip_x < self.all_methods.r_ankle_x)

        #check hands is in forward side
        check_left_hand = (self.all_methods.l_hip_x < self.all_methods.l_elbow_x and 
                      self.all_methods.l_hip_x < self.all_methods.l_wrist_x)
        check_right_hand = (self.all_methods.l_hip_x < self.all_methods.r_elbow_x and 
                      self.all_methods.l_hip_x < self.all_methods.r_wrist_x)

        # foot_y = max(toe_value_y,r_toe_y)

        if not llist:
            return None
        
        if (not self.check_initial_position and stand_on_knee):
            
            self.check_initial_position = True

        elif not stand_on_knee:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["you are not in initial position please stand on your knees"],llist=llist)

        if self.check_initial_position:
            self.l_count = 0

            voice_list = ["you are in initial position start ushtrasana","bend your hip back and take hands support"]

            if stand_on_knee:

                if self.l_count < len(voice_list):

                    self.all_methods.reset_after_40_sec()
                    trigger = self.all_methods.play_after_40_sec([voice_list[self.l_count]],llist=llist)
                    if trigger: 
                        self.l_count += 1
                else:
                    self.l_count = 1

            else:

                #CHECK KNEE 

                if (self.left_knee and 0 <= self.left_knee <= 69 and check_left_ankle):

                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please keep your left leg in 90 degrees"],llist=llist)

                elif not check_left_ankle:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["stand on left knee and keep your foot back side"],llist=llist) 

                elif (self.left_knee and 101 <= self.left_knee <= 180 and check_left_ankle):

                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["keep your left leg in 90 degrees"],llist=llist)

                else:
                    #CHECK RIGHT KNEE

                    if (self.right_knee1 and 0 <= self.right_knee1 <= 69 and check_right_ankle):

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["please keep your right leg 90 degress"],llist=llist)

                    elif not check_right_ankle:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["stand on right knee and keep your foot back side"],llist=llist)

                    elif self.right_knee1 and 101 <= self.right_knee1 <= 180:

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["keep your right leg 90 degrees"],llist=llist)

                    else:

                        if self.left_hip:
                            #CHECK HIP IS IN STRAIGHT 
                            if self.left_hip and 0 <= self.left_hip <= 119:

                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["you are bending hip too much please raise up slight"],llist=llist)

                            elif self.left_hip and 151 <= self.left_hip <= 180:

                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["your hip is in straight position please bend back and touch your palms to foot"],llist=llist)

                            else:
                                #CHECK LEFT ELBOW
                                if (self.left_elbow and 0 <=self.left_elbow <= 149 and check_left_hand):
                                    self.all_methods.reset_after_40_sec()  
                                    self.all_methods.play_after_40_sec(["please keep your left hand straight and touch your foot"],llist=llist)

                                elif not check_left_hand:

                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["your left hand in forward side please keep your left hand back"],llist=llist)

                                else:
                                    #CHECK RIGHT ELBOW
                                    if (self.right_elbow1 and 0 <= self.right_elbow1 <= 149 and check_right_hand):
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["keep your right hand straight and touch your foot"],llist=llist)

                                    elif not check_right_hand:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["your left hand in forward side please keep your left hand back"],llist=llist)

                                    else:
                                        #shoulder CHECK
                                        if self.left_shoulder:
                                            if self.left_shoulder and 0 <= self.left_shoulder <= 39 :
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please open your chest and gently move your hands outwards"],llist=llist)

                                            elif self.left_shoulder and 80 <= self.left_shoulder <= 180 :
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please open your chest and gently move your hands inwards"],llist=llist)

                                            else:

                                                #check head cross the head or not

                                                if self.head_value_x < self.l_shoulder_value_x:

                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["keep your head back"],llist=llist)

                                                else:

                                                    #left palms touch to hand or not
                                                    if self.toe_value_y and self.l_hand_y and 0 <= self.l_hand_y <= self.toe_value_y - 71:
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["your left palm must touch the foot"],llist=llist)

                                                    else:
                                                        #left palms touch to hand or not
                                                        if self.r_toe_y and self.r_hand_y and 0 <= self.r_hand_y <= self.r_toe_y - 71:
                                                            self.all_methods.reset_after_40_sec()
                                                            self.all_methods.play_after_40_sec(["your right palm must touch the foot"],llist=llist)

                                                        else:

                                                            if self.head_position != "Up":
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["please face your head up"],llist=llist)

                                                            else:

                                                                return True


    def wrong_right(self,frames,llist):
        count = 0

        self.all_methods.all_x_values(frames=frames,llist=llist)

        stand_on_knee = (self.left_knee1 and 80 <= self.left_knee1 <= 100 and
                         self.right_knee and 80 <= self.right_knee <=100 and
                          self.right_hip and 160 <= self.right_hip <= 180)
        
        hands_on_hip = (self.left_knee and 80 <= self.left_knee <= 100 and
                         self.right_knee1 and 80 <= self.right_knee1 <=100 and
                         self.left_elbow and 80 <= self.left_elbow <= 100 and
                         self.right_elbow1 and 80 <= self.right_elbow1 <= 100)
        
        self.head_value_x = llist[0][1]
        self.toe_value_x,self.toe_value_y,self.toe_value_z = llist[29][1:]
        self.r_toe_x,self.r_toe_y,self.r_toe_z = llist[30][1:]
        self.l_hand_y = llist[15][2]
        self.r_hand_y = llist[16][2]
        self.r_shoulder_value_x = llist[12][1]

        #check legs is in forward side
        check_left_ankle = (self.all_methods.l_hip_x > self.all_methods.l_ankle_x)
        check_right_ankle = (self.all_methods.l_hip_x > self.all_methods.r_ankle_x)

        #check hands is in forward side
        check_left_hand = (self.all_methods.l_hip_x > self.all_methods.l_elbow_x and 
                      self.all_methods.l_hip_x > self.all_methods.l_wrist_x)
        check_right_hand = (self.all_methods.l_hip_x > self.all_methods.r_elbow_x and 
                      self.all_methods.l_hip_x > self.all_methods.r_wrist_x)

        foot_y = max(self.toe_value_y,self.r_toe_y)

        if not llist:
            return None
        
        if (not self.check_initial_position and stand_on_knee):
            
            self.check_initial_position = True

        elif not stand_on_knee:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["you are not in initial position please stand on your knees"],llist=llist)

        if self.check_initial_position:
            self.r_count = 0
            voice_list = ["you are in initial positionstart ushtrasana"," please bend back your hip and take hands support"]

            if stand_on_knee:

                if self.r_count < len(voice_list):
                    self.all_methods.reset_after_40_sec()
                    trigger = self.all_methods.play_after_40_sec([voice_list[self.r_count]],llist=llist)
                    if trigger:
                        self.r_count += 1
                else:
                    self.r_count = 1

            else:
                #CHECK KNEE 

                if (self.left_knee1 and 0 <= self.left_knee1 <= 69 and check_left_ankle):

                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please keep your left leg in 90 degrees"],llist=llist)

                elif not check_left_ankle:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["stand on left knee and keep your foot back"],llist=llist)

                elif (self.left_knee1 and 101 <= self.left_knee1 <= 180 and check_left_ankle):

                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["keep your left leg in 90 degrees"],llist=llist)

                else:
                    #CHECK RIGHT KNEE

                    if (self.right_knee and 0 <= self.right_knee <= 69 and check_right_ankle):

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["please keep your right leg 90 degress"],llist=llist)

                    elif not check_right_ankle:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["stand on right knee and keep your foot back side"])

                    elif (self.right_knee and 101 <= self.right_knee <= 180 and check_right_ankle):

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["keep your right leg 90 degrees"],llist=llist)

                    else:

                        if self.right_hip:
                            #CHECK HIP IS IN STRAIGHT 
                            if self.right_hip and 0 <= self.right_hip <= 119:

                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["you are bending hip too much please raise up slight"],llist=llist)

                            elif self.right_hip and 151 <= self.right_hip <= 190:

                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["your hip is in straight position please bend back and touch your palms to foot"],llist=llist)

                            else:
                                #CHECK right ELBOW
                                if (self.right_elbow and 0 <=self.right_elbow <= 149 and check_right_hand):
                                    self.all_methods.reset_after_40_sec()  
                                    self.all_methods.play_after_40_sec(["please keep your left hand straight and touch your foot"],llist=llist)

                                elif not check_right_hand:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["your right hand in forward side please keep your right hand back"],llist=llist)

                                else:
                                    #CHECK  left ELBOW
                                    if (self.left_elbow1 and 0 <= self.left_elbow1 <= 149  and check_left_hand):
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["keep your right hand straight and touch your foot"],llist=llist)
                                    
                                    elif not check_left_hand:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["your left hand in forward side please keep your left hand back"],llist=llist)

                                    else:
                                        #shoulder CHECK
                                        if self.right_shoulder:
                                            if self.right_shoulder and 0 <= self.right_shoulder <= 39 :
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please open your chest and gently move your hands outwards"],llist=llist)

                                            elif self.right_shoulder and 80 <= self.right_shoulder <= 180 :
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please open your chest and gently move your hands inwards"],llist=llist)

                                            else:

                                                #check head cross the head or not

                                                if self.head_value_x > self.r_shoulder_value_x:

                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["keep your head back"],llist=llist)

                                                else:

                                                    #left palms touch to hand or not
                                                    if self.toe_value_y and self.l_hand_y and 0 <= self.l_hand_y <= self.toe_value_y - 71:
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["your left palm must touch the foot"],llist=llist)

                                                    else:
                                                        #right palms touch to hand or not
                                                        if self.r_toe_y and self.r_hand_y and 0 <= self.r_hand_y <= self.r_toe_y - 71:
                                                            self.all_methods.reset_after_40_sec()
                                                            self.all_methods.play_after_40_sec(["your right palm must touch the foot"],llist=llist)

                                                        else:
                                                            if self.head_position != "Up":
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["please face your head up"],llist=llist)

                                                            else:

                                                                return True
                                                            
                                                        

    def check_sitting(self,frames,llist,height,width):

        sitting_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if sitting_position == "sitting":
            
            return True 
        

        elif sitting_position != "sitting":
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be in sitting position","   ","this yoga may started in sitting position"],llist=llist)
            return False
        

    def check_side_view(self,frames,llist,height,width,left_knee_angle,right_knee_angle):

        self.left_knee_angle, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=left_knee_angle,lmList=llist)
        self.right_knee_angle, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=right_knee_angle,lmList=llist)

        side_view = self.all_methods.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)

        if side_view == "right":
            
            return "right"

        if side_view == "left":
           
            return "left"
        
        if side_view == "forward":
            self.all_methods.reset_after_40_sec() 
            self.all_methods.play_after_40_sec(["please turn total your body left, or , right"],llist=llist)
            return False  


    def left_reverse_to_strating_position(self,frames,llist):

        if not self.right_knee1 and not self.left_knee:
            return 
        check_last_before_position = (self.left_knee and 70 <= self.left_knee <= 100 and
                         self.right_knee1 and 70 <= self.right_knee1 <=100 and
                         self.left_elbow and 160 <= self.left_elbow <= 180 )
        
        check_last_position = (self.left_knee and 70 <= self.left_knee <= 100 and
                         self.right_knee1 and 70 <= self.right_knee1 <=100 and
                         self.left_hip and 160 <= self.left_hip <= 180)

        if check_last_before_position:
            self.l_r_count = 0

            voice_list = ["good,stay in same position , and wait for other instruction"," very good , back to stand on knee position"]

            if self.l_r_count < len(voice_list):

                self.all_methods.reset_after_40_sec()
                trigger = self.all_methods.play_after_40_sec([voice_list[self.l_r_count]],llist=llist)
                if trigger:
                    self.l_r_count += 1

            else:
                self.l_r_count = 1

        elif check_last_position:
            self.all_methods.reset_after_40_sec()
            final = self.all_methods.play_after_40_sec(["good job you complete yoga and back to relax"],llist=llist)
            if final:
                self.pose_completed = True
                return True
        
    def right_reverse_to_strating_position(self,frames,llist):

        if not self.right_knee and not self.left_knee1:
            return 

        check_last_before_position = (self.right_knee and 0 <= self.right_knee <= 30 and  
            self.left_knee1 and 0 <= self.left_knee1 <= 30 )
        
        check_last_position = (self.left_knee1 and 160 <= self.left_knee1 <= 180 and
                self.right_knee and 160 <= self.right_knee <= 180)
        
        if check_last_before_position:
            self.r_r_count = 0

            voice_list = ["good","stay in same and wait for instruction"," very good , back to stand on knee position"]

            if self.r_r_count < len(voice_list):
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec([voice_list[self.r_r_count]],llist=llist)
                self.r_r_count += 1

            else:
                self.r_r_count = 1
 
        elif check_last_position:
            self.all_methods.reset_after_40_sec()
            final = self.all_methods.play_after_40_sec(["good job you complete yoga and back to relax"],llist=llist)
            if final:
                self.pose_completed = True
                return True

        
    def left_usthrasana_name(self,frames): 

        correct =(
            self.left_shoulder and 40 <= self.left_shoulder <= 79 and
            self.left_elbow and 150 <= self.left_elbow <= 180 and
            self.right_elbow1 and 150 <= self.right_elbow1 <=180 and
            self.left_hip and 120 <= self.left_hip <= 150 and 
            self.left_knee and 70 <= self.left_knee <= 99 and
            self.right_knee1 and 70 <= self.right_knee1 <= 99 and 
            self.head_value_x and self.head_value_x > self.l_shoulder_value_x and
            self.toe_value_y and self.toe_value_y-70 <= self.l_hand_y <= self.toe_value_y and  
            self.head_position and self.head_position == "Up")
        
        if correct:
           
            cv.putText(frames,str("usthrasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                
            return True
        
        return False

    
    def right_usthrasana_name(self,frames):

        correct = (self.right_elbow and 150 <= self.right_elbow <= 180 and   
                   self.left_elbow1 and 150 <= self.left_elbow1 <= 180 and        
            self.right_hip and 120 <= self.right_hip <= 150 and            
            self.right_knee and 70 <= self.right_knee <= 99 and  
            self.left_knee1 and 70 <= self.left_knee1 <= 99 and                                    
            self.right_shoulder and 40 <= self.right_shoulder <= 79 and
            self.head_value_x and self.head_value_x < self.r_shoulder_value_x and
            self.toe_value_y and self.toe_value_y-70 <= self.l_hand_y <= self.toe_value_y and  
            self.r_toe_y and self.r_toe_y - 70 <= self.r_hand_y <= self.r_toe_y and
            self.head_position and self.head_position == "Up")
        if correct:

            cv.putText(frames,str("usthrasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

            return True
        
        return False
                                                                    
def main():

    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    global  llist
    detect = usthrasana()
    all_methods = allmethods()
    ref_video = cv.VideoCapture("second videos/usthrasana.mp4")
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
        img = cv.imread("second_images/usthrasana.jpg")
        img1 = cv.resize(img, None, fx=3.0, fy=3.0, interpolation=cv.INTER_LINEAR)
        
        if not flag:

            detect.pose_positions(frames,draw=False)
            llist = detect.pose_landmarks(frames,False)
            # usthrasana_slope = detect.slope(frames=frames,llist=llist,height=height,width=width)
            sitting_detect = detect.check_sitting(frames=frames,llist=llist,height=height,width=width)
            
            if sitting_detect:

                side_view = detect.check_side_view(frames=frames,llist=llist,height=height,width=width,left_knee_angle=(23,25,27),right_knee_angle=(24,26,28))

                if side_view == "left":
                    detect.left_usthrasana(frames=frames,llist=llist,elbow=(11,13,15), hip=(11,23,25), knee=(23,25,27), shoulder=(13,11,23),right_knee1=(24,26,28),right_elbow=(12,14,16),draw=False)
                    # wrong_left = detect.wrong_left(frames=frames,llist=llist,height=height,width=width)
                    if not checking_wrong:
                        wrong_left = detect.wrong_left(frames=frames,llist=llist)
                        if wrong_left:
                            checking_wrong = True
                    
                    if checking_wrong:
                        # correct = detect.left_usthrasana_name(frames=frames)
                        if not ready_for_exercise and not flag:
                            correct = detect.left_usthrasana_name(frames=frames)
                            if correct:
                                ready_for_exercise = True
                                reverse_yoga = True

                    if reverse_yoga  and not flag:
                        left_reverse = detect.left_reverse_to_strating_position(frames=frames,llist=llist)
                        # ready_for_exercise = False
                        if left_reverse:
                            flag = True
            

                #Right Side
                elif side_view == "right":
                    detect.right_usthrasana(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee1=(23,25,27),left_elbow=(11,13,15),draw=False)                        
                    
                    # correct = detect.right_usthrasana_name(frames=frames)
                    if not checking_wrong:
                        wrong_right = detect.wrong_right(frames=frames,llist=llist)
                        if wrong_right:
                            checking_wrong = True

                    if checking_wrong:
                        if not ready_for_exercise and not flag:
                            correct = detect.right_usthrasana_name(frames=frames)
                            if correct:
                                ready_for_exercise = True
                                reverse_yoga = True

                        if reverse_yoga and not flag:
                            right_reverse = detect.right_reverse_to_strating_position(frames=frames,llist=llist)
                            if right_reverse:
                                flag = True
                    
        # elif flag:
        #     cv.putText(frames, "Exercise Completed", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
                        
        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main() 