import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math
import os

from threading import Thread
from allMethods import allmethods
from voiceModule import VoicePlay
from face_detect import HeadPoseEstimator

class dhanurasana:
     
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
        self.angle =0
        self.voice_thread = None
        self.voice_detect = False
        self.all_methods = allmethods()
        self.head_pose = HeadPoseEstimator()
        self.voice = VoicePlay()

        self.check_sleep_position = False
        self.check_initial_position = False
        self.start_exercise = False
        self.reverse_sleep = False
        self.pose_completed = False

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        #shoulder hip
        self.MAX_BACK_LIMT = 45
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

        self.lpslist =[]

        if self.results.pose_landmarks:

            self.h,self.w,cl = frames.shape

            for id,poselms in enumerate(self.results.pose_landmarks.landmark):
                px,py,pz = (poselms.x * self.w) , (poselms.y * self.h),(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lpslist 
    

    def right_dhanurasana(self,frames,llist,elbow, hip, knee, shoulder,left_knee,left_elbow1, draw = True):


        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        self.ground_right = self.all_methods.ground_distance_right(frames=frames,lmlist=llist)
        self.min_ground_right = self.ground_right-30

        self.right_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
        self.right_hip, hip_coords = self.all_methods.calculate_angle(frames=frames, points=hip,lmList=llist)
        self.right_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)
        self.right_shoulder,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.left_knee1,left_knee1_coords = self.all_methods.calculate_angle(frames=frames,lmList=llist,points=left_knee)
        self.left_elbow1,left_elbow1_coords = self.all_methods.calculate_angle(frames=frames,lmList=llist,points=left_elbow1)

        if draw:
            if elbow_coords:
                cv.putText(frames, str(int(self.right_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.right_hip)), (hip_coords[2]-20, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.right_knee)), (knee_coords[2]-20, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if points_cor7:
                cv.putText(frames, str(int(self.right_shoulder)), (points_cor7[2]+10, points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            if left_knee1_coords:
                cv.putText(frames,str(int(self.left_knee1)),(left_knee1_coords[2]+10 , left_knee1_coords[3]+10),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            if left_elbow1_coords:
                cv.putText(frames,str(int(self.left_elbow1)),(left_elbow1_coords[2]+10 , left_elbow1_coords[3]+10),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)


        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,self.head_position,self.left_elbow1,self.left_knee1
       
    def left_dhanurasana(self,frames,llist,elbow, hip, knee,shoulder,right_knee,right_elbow1,draw = True):


        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        self.ground_left = self.all_methods.ground_distance_left(frames=frames,lmlist=llist)
        self.min_ground_left = self.ground_left-30

        self.left_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
        self.left_hip, hip_coords = self.all_methods.calculate_angle(frames=frames,points= hip,lmList=llist)
        self.left_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)       
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.right_knee1,right_knee1_coords = self.all_methods.calculate_angle(frames=frames,lmList=llist,points=right_knee)
        self.right_elbow1, right_elbow1_coords = self.all_methods.calculate_angle(frames=frames, points=right_elbow1,lmList=llist)


        if draw:
            if elbow_coords:
                cv.putText(frames, str(int(self.left_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.left_hip)), (hip_coords[2]+10, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.left_knee)), (knee_coords[2]+10, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if points_cor8:
                cv.putText(frames, str(int(self.left_shoulder)), (points_cor8[2]+10, points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            if right_knee1_coords:
                cv.putText(frames,str(int(self.right_knee1)),(right_knee1_coords[2]+10 , right_knee1_coords[3]+10),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            if right_elbow1_coords:
                cv.putText(frames,str(int(self.right_elbow1)),(right_elbow1_coords[2]+10 , right_elbow1_coords[3]+10),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)


        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.head_position,self.right_elbow1,self.right_knee1

    
    def wrong_left(self,frames,llist,height,width):

        count = 0

        left_shoulder_x,left_shoulder_y,left_shoulder_z = llist[11][1:]
        right_shoulder_x,right_shoulder_y,right_shoulder_z = llist[12][1:]
        tolerance = abs(left_shoulder_z - right_shoulder_z)


        #check legs is in forward side
        check_left_knee = (self.all_methods.l_hip_x < self.all_methods.l_knee_x)
        check_right_knee = (self.all_methods.l_hip_x < self.all_methods.r_knee_x)

        #check hands is in forward side
        check_left_hand = (self.all_methods.l_hip_x < self.all_methods.l_elbow_x and 
                      self.all_methods.l_hip_x < self.all_methods.l_wrist_x)
        check_right_hand = (self.all_methods.l_hip_x < self.all_methods.r_elbow_x and 
                      self.all_methods.l_hip_x < self.all_methods.r_wrist_x)

        #left slope condition
        self.left_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width,draw=False)
        self.left_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=False)


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

            if tolerance <= 40:

                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are in side sleep position","please sleep in flat position"],llist=llist)


            elif not self.start_exercise and right_shoulder_z > left_shoulder_z and tolerance > 40:
                
                self.start_exercise = True
                self.reverse_sleep = True

            elif not self.reverse_sleep and left_shoulder_z > right_shoulder_z and tolerance > 40:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are in flat position","Lie on your stomach with your forehead on the ground"],llist=llist)

            if self.start_exercise:
                # check sleep position
                if self.right_shoulder_hip and self.right_hip:

                    if ((0 <= self.right_shoulder_hip <= 15) and (0 <= self.right_hip_knee <= 15)):
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["you are in initial position, start dhanurasana, bend your hip back"],llist=llist)
                        return
                    
                    else:

                        # ========== LEFT HIP ==========
                        if self.left_hip is not None:
                            if 151 <= self.left_hip <= 180:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please bend your hip back"], llist=llist)
                                    return

                            elif 0 <= self.left_hip <= 109:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["your are bend too much your hip"], llist=llist)
                                    return
                            
                            else:

                                # ========== LEFT KNEE ==========
                                if self.left_knee is not None:
                                    if 101 <= self.left_knee <= 180:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please fold your left leg line in reference video"], llist=llist)
                                        return

                                    elif 0 <= self.left_knee <= 69:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["you are bending too much left leg please relax little"], llist=llist)
                                        return
                                    
                                    else:

                                        # ========== RIGHT KNEE ==========
                                        if self.right_knee1 is not None:
                                            if 141 <= self.right_knee1 <= 180:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["your right leg may be need to fold little bit"], llist=llist)
                                                return

                                            elif 0 <= self.right_knee1 <= 79:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["you are bending too much your right leg please relax little bit"], llist=llist)
                                                return
                                            
                                            else:

                                                # ========== LEFT ELBOW ==========
                                                if self.left_elbow is not None:
                                                    if 0 <= self.left_elbow <= 159:
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["please be straight your left hand back and hold your foot"], llist=llist)
                                                        return
                                                    else:
                                                    
                                                        # ========== RIGHT ELBOW ==========
                                                        if self.right_elbow1 is not None:
                                                            if 0 <= self.right_elbow1 <= 159:
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["please be straight your right hand back and hold your foot"], llist=llist)
                                                                return

                                                            else:
                                                                # ========== LEFT SHOULDER ==========
                                                                if self.left_shoulder is not None:
                                                                    if 0 <= self.left_shoulder <= 44:
                                                                        self.all_methods.reset_after_40_sec()
                                                                        self.all_methods.play_after_40_sec(["please raise your hands up little bit like in reference video"], llist=llist)
                                                                        return

                                                                    elif 71 <= self.left_shoulder <= 180:
                                                                        self.all_methods.reset_after_40_sec()
                                                                        self.all_methods.play_after_40_sec(["please move your hands inwards little bit like in reference video"], llist=llist)
                                                                        return
                                                                    
                                                                    else:
                                                                        # ========== HEAD POSITION ==========
                                                                        if self.head_position is not None:
                                                                            if self.head_position != "Left":
                                                                                
                                                                                self.all_methods.reset_after_40_sec()
                                                                                self.all_methods.play_after_40_sec(["please turn your head to left side"], llist=llist)
                                                                                return
                                                                            
                                                                            else:
                                                                                return True



    def wrong_right(self,frames,llist,height,width):

        left_shoulder_x,left_shoulder_y,left_shoulder_z = llist[11][1:]
        right_shoulder_x,right_shoulder_y,right_shoulder_z = llist[12][1:]

        tolerance = abs(left_shoulder_z - right_shoulder_z)

        if not self.right_hip and not self.right_elbow and not self.right_knee and not self.right_shoulder and not self.left_knee1 and not right_shoulder_z and not left_shoulder_z:
            return
        
        #check legs is in forward side
        check_left_knee = (self.all_methods.l_hip_x < self.all_methods.l_knee_x)
        check_right_knee = (self.all_methods.l_hip_x < self.all_methods.r_knee_x)

        #check hands is in forward side
        check_left_hand = (self.all_methods.l_hip_x < self.all_methods.l_elbow_x and 
                      self.all_methods.l_hip_x < self.all_methods.l_wrist_x)
        check_right_hand = (self.all_methods.l_hip_x < self.all_methods.r_elbow_x and 
                      self.all_methods.l_hip_x < self.all_methods.r_wrist_x)
        
        #right slope condition
        self.right_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=False)
        self.right_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=False)
        
        sleeping_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if not self.check_sleep_position and sleeping_position == "sleeping":
            
            self.all_methods.reset_voice()
            self.all_methods.play_voice(["you are in sleeping position and do dhanurasana"],llist=llist)
            self.check_sleep_position = True
            self.check_initial_position = True
            return 

        elif not self.check_initial_position and sleeping_position != "sleeping":
            self.all_methods.reset_voice()
            self.all_methods.play_voice(["please be in sleeping position","   ","this yoga may started in sleeping position"],llist=llist)
            return 
        
        if self.check_sleep_position:

            if tolerance <= 40:

                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are in side sleep position"],llist=llist)

            elif not self.start_exercise and left_shoulder_z > right_shoulder_z and tolerance > 40:
                self.start_exercise = True
                self.reverse_sleep = True

            elif not self.reverse_sleep and right_shoulder_z > left_shoulder_z and tolerance > 40:

                self.all_methods.reset_after_40_sec([])
                self.all_methods.play_after_40_sec(["you are in flat position","Lie on your stomach with your forehead on the ground"],llist=llist)

            if self.start_exercise:
                # check sleep position
                if self.right_shoulder_hip and self.right_hip:

                    if ((0 <= self.right_shoulder_hip <= 15) and (0 <= self.right_hip_knee <= 15)):
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["you are in initial position, start dhanurasana, bend your hip back"],llist=llist)
                        return
                    
                    else:

                        # ===== Right Hip =====
                        if self.right_hip is not None:
                            if 151 <= self.right_hip <= 180:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please bend your hip back"], llist=llist)
                                return
                            
                            elif (0 <= self.right_hip <= 109):
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["you are bending too much your hip"], llist=llist)
                                return
                            
                            else:
                                # ===== Right Knee =====
                                if self.right_knee is not None:
                                    if 101 <= self.right_knee <= 180:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please fold your right leg like in reference video"], llist=llist)
                                        return

                                    elif 0 <= self.right_knee <= 69:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["you are bending too much right leg, please relax it a little"], llist=llist)
                                        return
                                    
                                    else:

                                        # ===== Left Knee (right pose check) =====
                                        if self.left_knee1 is not None:
                                            if 141 <= self.left_knee1 <= 180:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["your left leg may need to fold a little bit"], llist=llist)
                                                return
                                            
                                            elif 0 <= self.left_knee1 <= 79:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["you are bending too much your left leg, please relax a little bit"], llist=llist)
                                                return
                                            else:
                                                
                                                # ===== Right Elbow =====
                                                if self.right_elbow is not None:
                                                    if 0 <= self.right_elbow <= 159:
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["please straight your right hand and hold your foot"], llist=llist)
                                                        return
                                                    
                                                    else:

                                                        #======Left Elbow =====
                                                        if self.left_elbow1 is not None:
                                                            if 0 <= self.left_elbow1 <= 159:
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["please straight your left hand and hold your foot"], llist=llist)
                                                                return
                                                            
                                                            else:

                                                                # ===== Right Shoulder =====
                                                                if self.right_shoulder is not None:
                                                                    if 0 <= self.right_shoulder <= 44:
                                                                        self.all_methods.reset_after_40_sec()
                                                                        self.all_methods.play_after_40_sec(["please straight your hands slightly like in the reference video"], llist=llist)
                                                                        return

                                                                    elif 71 <= self.right_shoulder <= 180:
                                                                        self.all_methods.reset_after_40_sec()
                                                                        self.all_methods.play_after_40_sec(["please staright your hands slightly like in the reference video"], llist=llist)
                                                                        return

                                                                    else: 

                                                                        # ===== Head Position =====
                                                                        if self.head_position is not None:
                                                                            if self.head_position != "Right":
                                                                                self.all_methods.reset_after_40_sec()
                                                                                self.all_methods.play_after_40_sec(["please turn your head to right side"], llist=llist)

                                                                                return
                                                                            
                                                                            else:
                                                                                return True
                    

    def check_side_view(self,frames,llist,height,width):

        if len(llist) == 0:
            return None

        nose_x = llist[0][1]

        side_view = self.all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION="head",head=nose_x)

        if side_view == "left":
            return "left"
        
        elif side_view == "right":
            return "right"
        
        else:
            self.all_methods.reset_voice()
            self.all_methods.play_voice(["please turn left side or right side"],llist=llist)
            return False
        
    def right_side_reverse_to_sleep(self,frames,llist,height,width):

        #right slope condition
        self.right_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=False)
        self.right_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=False)


        correct = (self.left_elbow1 and 160 <= self.left_elbow1 <= 180 and
            self.right_elbow  and 160 <= self.right_elbow <= 180 and 
            self.right_hip and 110 <= self.right_hip <= 150 and  
            self.right_knee and 70 <= self.right_knee <= 100 and
            self.left_knee1 and 80 <= self.left_knee <= 140 and
            self.right_shoulder and 45 <= self.right_shoulder <= 70 and
            self.head_position and self.head_position == "Right")
        
        if correct:

            self.all_methods.reset_voice()
            self.all_methods.play_voice(["you done your yoga pose perfect","and back to sleep Position"],llist=llist)
            return False

        elif  ((0 <= self.right_shoulder_hip <= 15) and (0 <= self.right_hip_knee <= 15)):

            self.all_methods.reset_voice()
            self.all_methods.play_voice(["you are in final position"],llist=llist)
            self.pose_completed = True
            return True
        

    def left_side_reverse_to_sleep(self,frames,llist,height,width):

        #right slope condition
        self.right_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=False)
        self.right_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=False)


        correct = (self.right_elbow1 and 160 <= self.right_elbow1 <= 180 and 
            self.left_elbow and 160 <= self.left_elbow <= 180 and
            self.left_hip and 110 <= self.left_hip <= 150 and
            self.left_knee and 70 <= self.left_knee <= 100 and 
            self.right_knee1 and 80 <= self.right_knee1 <= 140 and
            self.left_shoulder and 45 <= self.left_shoulder <= 70 and
            self.head_position and self.head_position == "Left")
        
        if correct:

            self.all_methods.reset_voice()
            self.all_methods.play_voice(["you done your yoga pose perfect","and back to sleep Position"],llist=llist)
            return False

        elif  ((0 <= self.right_shoulder_hip <= 15) and (0 <= self.right_hip_knee <= 15)):

            self.all_methods.reset_voice()
            self.all_methods.play_voice(["you are in final position"],llist=llist)
            self.pose_completed = True
            return True
        

    def right_dhanurasana_name(self,frames):  
       
        if (
            self.left_elbow1 and 160 <= self.left_elbow1 <= 180 and
            self.right_elbow  and 160 <= self.right_elbow <= 180 and 
            self.right_hip and 110 <= self.right_hip <= 150 and  
            self.right_knee and 70 <= self.right_knee <= 100 and
            self.left_knee1 and 80 <= self.left_knee <= 140 and
            self.right_shoulder and 45 <= self.right_shoulder <= 70 and
            self.head_position and self.head_position == "Right" ):
            
                # self.all_methods.play_voice("Forward Bend, is a yoga pose that involves folding forward from a hasta uttanasana position, with legs straight and hands on the ground or")
                cv.putText(frames,str("dhanurasana"),(80,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                self.all_methods.reset_voice()
                # self.all_methods.play_voice(["you done your yoga pose perfect"],llist=llist)
                
                return True
        
        return False  
    

    def left_dhanurasana_name(self,frames):


        if (
            self.right_elbow1 and 160 <= self.right_elbow1 <= 180 and 
            self.left_elbow and 160 <= self.left_elbow <= 180 and
            self.left_hip and 110 <= self.left_hip <= 150 and
            self.left_knee and 70 <= self.left_knee <= 100 and 
            self.right_knee1 and 80 <= self.right_knee1 <= 140 and
            self.left_shoulder and 45 <= self.left_shoulder <= 70 and
            self.head_position and self.head_position == "Left"
            ):

                cv.putText(frames,str("dhanurasana"),(80,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                self.all_methods.reset_voice()
                # self.all_methods.play_voice(["you done your yoga pose perfect"],llist=llist)
                
                return True
        
        return False  
    
    # def head_value(self,llist,frames):

    #     if len(llist) == 0:
    #         return None

    #     self.nose_x = llist[0][1]
    #     return self.nose_x
    
   
def main():
    global llist
    all_methods = allmethods()
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980)
    detect = dhanurasana()
    ready_for_exercise = False
    checking_wrong = False
    flag = False

    ref_video=cv.VideoCapture("second videos/dhanurasana.webm")
    

    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        # img = cv.imread("second_images/dhanurasana.jpeg")
        # img1 = cv.resize(img, None, fx=4.0, fy=4.0, interpolation=cv.INTER_LINEAR)

        if not flag:

            ret_ref, ref_frame = ref_video.read()
            if not ret_ref:
                ref_video.set(cv.CAP_PROP_POS_FRAMES, 0)  # loop video
                ret_ref, ref_frame = ref_video.read()

            # Resize reference video frame and place it at top-left
            if ret_ref:
                ref_frame = cv.resize(ref_frame, (400, 300))
                h, w, _ = ref_frame.shape
                frames[0:h, 0:w] = ref_frame  # Overlay in corner

            detect.pose_positions(frames,draw = False)
            llist = detect.pose_landmarks(frames,draw=False)
            # nose = detect.head_value(llist=llist,frames=frames)
            # sleeping_position =detect.check_sleeping(frames=frames,llist=llist,height=height,width=width)
            side_view = detect.check_side_view(frames=frames,llist=llist,height=height,width=width)

            if len(llist) is None:
                return None
            
            if side_view ==  "right":

                detect.right_dhanurasana(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),left_elbow1=(11,13,15),draw=False)
                wrong_right = detect.wrong_right(frames=frames,llist=llist,height=height,width=width)
                if not checking_wrong and wrong_right:
                    checking_wrong = True

                if checking_wrong:
                    correct = detect.right_dhanurasana_name(frames=frames)

                    if not ready_for_exercise and not flag:
                        if correct:
                            ready_for_exercise = True

                if ready_for_exercise and not flag:
                    reverse_correct = detect.right_side_reverse_to_sleep(frames=frames,llist=llist,height=height,width=width)
                    if reverse_correct:
                        flag = True

            elif side_view == "left":

                detect.left_dhanurasana(frames=frames,llist=llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),right_knee= (24,26,28),right_elbow1=(12,14,16),draw=False)
                wrong_left = detect.wrong_left(frames=frames,llist=llist,height=height,width=width)
                if not checking_wrong and wrong_left:
                    checking_wrong = True
                
                if checking_wrong:
                    correct = detect.left_dhanurasana_name(frames=frames)

                    if not ready_for_exercise and not flag:
                        if correct:
                            ready_for_exercise = True

                if ready_for_exercise and not flag:
                    reverse_correct = detect.left_side_reverse_to_sleep(frames=frames,llist=llist,height=height,width=width)
                    if reverse_correct:
                        flag = True


        cv.imshow("video",frames)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()

main()

