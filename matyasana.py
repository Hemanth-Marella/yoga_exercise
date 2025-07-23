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
# from body_position import bodyPosition

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
        self.l_check_leg_raise = False
        self.r_check_leg_raise = False
        self.check_sleep_position = False
        self.half_pose_detected = False
        self.seventy_pose_completed = False
        self.r_leg_up = False
        self.l_leg_up = False

        self.l_r_count = 0
        self.r_r_count = 0
        self.l_count = 0
        self.r_count = 0

        self.angle =0
        self.all_methods = allmethods()
        self.voice = VoicePlay()
        # self.body_position = bodyPosition()
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

    
    def matyasana(self, frames,llist, l_elbow, l_hip, l_knee,l_shoulder,r_elbow,r_hip,r_knee,r_shoulder,l_draw =True,r_draw = True,l_v_draw = True,r_v_draw = True):
        
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

    def wrong_left(self,frames,llist,height,width):

        if len(llist) == 0:
            return

        self.all_methods.all_x_values(frames=frames,llist=llist)
        self.all_methods.all_y_values(frames=frames,llist=llist)
        self.all_methods.all_z_values(frames=frames,llist=llist)

        # if (self.all_methods.l_shoulder_y > self.all_methods.nose_y):
        #     # print("true")

        left_shoulder_x,left_shoulder_y,left_shoulder_z = llist[11][1:]
        right_shoulder_x,right_shoulder_y,right_shoulder_z = llist[12][1:]
        tolerance = abs((left_shoulder_z - right_shoulder_z))
        cv.putText(frames,f"tolerance-->{str(tolerance)}",(10,100),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

        #right slope condition
        self.right_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=False)
        self.right_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=False)
        self.right_hip_ankle = self.all_methods.slope(frames=frames,lmlist=llist,point1=26,point2=28,height=height,width=width,draw=False)

        r_45 = ( 50 <= self.right_knee <= 80)
        l_45 = (50 <= self.left_knee <= 80)

        check_hip_knee_on_ground = abs(self.all_methods.l_knee_y - self.all_methods.l_hip_y)
        # cv.putText(frames,f"hip_knee_ground{check_hip_knee_on_ground}",(20,200),2,cv.FONT_HERSHEY_PLAIN,(0,255,0),2)

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
            # print("hello ")

            if not self.check_initial_position and sleeping_position != "sleeping":
                self.all_methods.reset_voice()
                self.all_methods.play_voice(["please be in sleeping position","   ","this yoga may started in sleeping position"],llist=llist)
             
            if sleeping_position == "sleeping":

                if tolerance <= 0.15:

                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are in side sleep position","please sleep in flat position"],llist=llist)

                elif  left_shoulder_z < right_shoulder_z and tolerance > 0.15:

                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["Lie on your back, with legs extended"],llist=llist)


                elif  left_shoulder_z > right_shoulder_z and tolerance > 0.15:
                    self.start_exercise = True

        #CHECK INITIAL POSITION
        if self.start_exercise:
            if left_shoulder_z > right_shoulder_z:

                if (self.left_knee and 160 <= self.left_knee <= 180 and 
                    self.right_knee and 160 <= self.right_knee <= 180):

                    initial_voices = ["you are in initial position and start yoga mathyasana","please start yoga and fold your left leg around 45 degress"]

                    if self.l_count < len(initial_voices):
                        self.all_methods.reset_after_40_sec()
                        voice = self.all_methods.play_after_40_sec([initial_voices[self.l_count]],llist=llist)
                        if voice:
                            self.l_count += 1

                    else:
                        self.l_count = 1

                else:
                    #check left leg at 45 or not
                    if not self.l_detect_45:

                        if not l_45 and self.left_knee and 0 <= self.left_knee <= 49:
                           
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your left leg around 45 degress outwards"],llist=llist)

                        elif not l_45 and self.left_knee and 81 <= self.left_knee <= 120:
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["good, fold your left leg ,inward some more"],llist=llist)

                        elif not l_45 and self.left_knee and 121 <= self.left_knee <= 180:
                            
                            self.all_methods.reset_after_40_sec()

                            self.all_methods.play_after_40_sec(["please fold your left leg, inwards, 45 degress"],llist=llist)

                        elif l_45:
                            self.all_methods.play_after_40_sec(["giod"],llist=llist)
                            self.l_detect_45 = True
                    
                    #check right leg at 45 or not
                    if self.l_detect_45 and not self.r_detect_45:
                        if not r_45 and self.right_knee and 0 <= self.right_knee <= 49:
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your right leg around 45 degress outwards"],llist=llist)

                        elif not r_45 and self.right_knee and 81 <= self.right_knee <= 120:
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["good  , fold your right leg inward , some more"],llist=llist)

                        elif not r_45 and self.right_knee and 121 <= self.right_knee <= 180:
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please fold your right leg, inwards, 45 degress"],llist=llist)

                        elif r_45:
                            
                            self.r_detect_45 = True

                    #check left leg is in thigh or not
                    if self.r_detect_45 and not self.l_check_leg_raise:

                        if not self.l_leg_up:
                            if self.all_methods.l_ankle_y < self.all_methods.r_knee_y:
                                self.l_leg_up = True
                            elif self.all_methods.l_ankle_y > self.all_methods.r_knee_y:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please , lift your left leg up"], llist=llist)

                        if self.l_leg_up:
                            if self.all_methods.l_ankle_y < self.all_methods.r_knee_y:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please , keep your left leg on right thigh"], llist=llist)

                            if (self.all_methods.l_ankle_y > self.all_methods.r_knee_y and 
                                self.all_methods.l_ankle_y < self.all_methods.r_hip_y):
                                self.l_check_leg_raise = True
                    
                    if self.l_check_leg_raise and not self.r_check_leg_raise:
                        
                        if not self.r_leg_up:

                            if self.all_methods.r_ankle_y > self.all_methods.l_knee_y:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please , lift your right leg up"],llist=llist)

                            elif self.all_methods.r_ankle_y < self.all_methods.l_knee_y:
                                self.r_leg_up = True

                        if self.r_leg_up:
                            if self.all_methods.r_ankle_y < self.all_methods.r_knee_y:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please keep your right leg on left thigh"],llist=llist)

                            if(self.all_methods.r_ankle_y > self.all_methods.l_knee_y and
                                self.all_methods.r_ankle_y < self.all_methods.r_hip_y):

                                self.r_check_leg_raise = True   

                    if self.l_check_leg_raise and not self.half_pose_detected:
                        if check_hip_knee_on_ground > 20:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please , lay your lower body on ground"],llist=llist)

                        elif self.right_elbow and 0 <= self.right_elbow <= 159:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please, keep your hands straight beside your left upper body"],llist=llist)

                        else:
                            self.all_methods.reset_after_40_sec()
                            half_pose_voice = self.all_methods.play_after_40_sec(["excellent , you completed half pose"],llist=llist)
                            if half_pose_voice:
                                self.half_pose_detected = True

                    #completed half pose
                    if self.half_pose_detected and not self.seventy_pose_completed:

                        hip_wrist_x = (self.all_methods.l_wrist_x - self.all_methods.l_hip_x)
                        hip_wrist_y = (self.all_methods.l_wrist_y - self.all_methods.l_hip_y)

                        if check_hip_knee_on_ground and check_hip_knee_on_ground > 20:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["your hip and knnes is on ground only"],llist=llist)

                        else:
                            if self.right_elbow and 160 <= self.right_elbow <= 180:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please , lay your elbow on ground"],llist=llist)

                            else:
                                if hip_wrist_x > 10 and hip_wrist_y > 20:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please , tour hip with your hand"],llist=llist)

                                else:
                                    if self.right_shoulder_hip and 0 <= self.right_shoulder_hip <= 15:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["take elbow, and head support , lift your upper body up"],llist=llist)
                                    
                                    elif self.right_shoulder_hip and 45 <= self.right_shoulder_hip <= 90:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please, down your upper body"],llist=llist)   

                                    else:
                                        if self.head_position != "Left":
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please , set your head in sorrect position"],llist=llist)

                                        else:
                                            if (self.right_elbow and 130 <= self.right_elbow <= 159 and
                                                check_hip_knee_on_ground and check_hip_knee_on_ground <= 20 and
                                                self.head_position and self.head_position == "Left" and
                                                self.left_shoulder_hip and 16 <= self.left_shoulder_hip <= 44):

                                                self.all_methods.reset_after_40_sec()
                                                seventy_pose = self.all_methods.play_after_40_sec(["awesome , you completed 70 percent pose"],llist=llist)
                                                if seventy_pose:
                                                    self.seventy_pose_completed = True

                    if self.seventy_pose_completed:
                        toe_touch_y = (self.all_methods.l_wrist_y - self.all_methods.l_toe_y)
                        toe_touch_x = (self.all_methods.l_wrist_x - self.all_methods.l_toe_x)

                        if self.right_elbow and 0 <= self.right_elbow <= 159:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please keep hands straight, and touch your toes"],llist=llist)

                        else:
                            if toe_touch_x > 30 and toe_touch_y < 15:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please touch your toe with your hands"],llist=llist)

                            else:
                                return True           


    def wrong_right(self,frames,llist,height,width):

        if len(llist) == 0:
            return

        self.all_methods.all_x_values(frames=frames,llist=llist)
        self.all_methods.all_y_values(frames=frames,llist=llist)
        self.all_methods.all_z_values(frames=frames,llist=llist)

        left_shoulder_x,left_shoulder_y,left_shoulder_z = llist[11][1:]
        right_shoulder_x,right_shoulder_y,right_shoulder_z = llist[12][1:]

        tolerance = abs(left_shoulder_z - right_shoulder_z)
        check_hip_knee_on_ground = abs(self.all_methods.r_knee_y - self.all_methods.r_hip_y)
        # cv.putText(frames,f"hip_knee_ground{check_hip_knee_on_ground}",(10,300),3,cv.FONT_HERSHEY_PLAIN,(255,255,0),2)

        #left slope condition
        self.left_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width,draw=False)
        self.left_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=False)
        self.left_hip_ankle = self.all_methods.slope(frames=frames,lmlist=llist,point1=25,point2=27,height=height,width=width,draw=False)

        r_45 = ( 50 <= self.right_knee <= 80)
        l_45 = (50 <= self.left_knee <= 80)
        
        #check sleep POSITION
        sleeping_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if not self.check_sleep_position and sleeping_position == "sleeping":
            
            self.check_sleep_position = True
            self.check_initial_position = True
             
        elif not self.check_initial_position and sleeping_position != "sleeping":
            self.all_methods.reset_voice()
            self.all_methods.play_voice(["please be in sleeping position","   ","this yoga may started in sleeping position"],llist=llist)

        #CHECK FLAT POSITION
        if self.check_sleep_position and not self.start_exercise:
           
            if sleeping_position != "sleeping":
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["please be in sleeping position , this yoga may started in sleeping position"])

            if sleeping_position == "sleeping":

                if  tolerance <= 0.15:

                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are in side sleep position","please sleep in flat position"],llist=llist)

                elif  left_shoulder_z > right_shoulder_z and tolerance > 0.15:

                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["Lie on your back, with legs extended"],llist=llist)

                elif  left_shoulder_z < right_shoulder_z and tolerance > 0.15:
                    
                    self.start_exercise = True

        #CHECK INITIAL POSITION
        if self.start_exercise:

            if left_shoulder_z < right_shoulder_z:

                if ((160 <= self.right_knee <= 180) and (160 <= self.left_knee <= 180)):
                    
                    voices = ["you are in initial position and start yoga mathyasana","please start yoga and fold your right leg around 45 degress"]
                    if self.r_count < len(voices):
                        self.all_methods.reset_after_40_sec()
                        voice = self.all_methods.play_after_40_sec([voices[self.r_count]],llist=llist)
                        if voice:
                            self.r_count += 1
                    else:
                        self.r_count = 1
            
                else:    
                    #check right leg at 45 or not
                    if not self.l_detect_45:

                        if not r_45 and self.right_knee and 0 <= self.right_knee <= 49:
                            # print("angle between 45")
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your right leg around 45 degress outwards"],llist=llist)

                        elif not r_45 and self.right_knee and 81 <= self.right_knee <= 120:
                            # print("angle between 80")
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["good , and , fold your right leg ,inward"],llist=llist)

                        elif not r_45 and self.right_knee and 121 <= self.right_knee <= 180:
                            # print("angle between 140")
                            self.all_methods.reset_after_40_sec()

                            self.all_methods.play_after_40_sec(["please fold your right leg, inwards, 45 degress"],llist=llist)

                        elif r_45:
                            # print("...................................")
                            # self.all_methods.play_after_40_sec(["good"],llist=llist)
                            self.r_detect_45 = True

                    #check left leg at 45 or not
                    if self.r_detect_45 and not self.l_detect_45:
                        if not l_45 and self.left_knee and 0 <= self.left_knee <= 49:
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your left leg around 45 degress outwards"],llist=llist)

                        elif not l_45 and self.left_knee and 81 <= self.left_knee <= 120:
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["good, fold your left leg inward little more"],llist=llist)

                        elif not l_45 and self.left_knee and 121 <= self.left_knee <= 180:
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please fold your left leg, inwards, 45 degress"],llist=llist)

                        elif l_45:
                            
                            self.l_detect_45 = True

                    #check right leg is in thigh
                    if self.l_detect_45 and not self.r_check_leg_raise:
                        
                        if not self.r_leg_up:

                            if self.all_methods.r_ankle_y < self.all_methods.l_knee_y:
                                self.r_leg_up = True

                            elif self.all_methods.r_ankle_y > self.all_methods.l_knee_y:
                                
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please , lift your right leg up"],llist=llist)

                        if self.r_leg_up:

                            if self.all_methods.r_ankle_y < self.all_methods.l_knee_y:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please , keep your right leg on left thigh"],llist=llist)

                            elif (self.all_methods.r_ankle_y > self.all_methods.l_knee_y and 
                                self.all_methods.r_ankle_y < self.all_methods.l_hip_y):
                                print(".........")
                                self.r_check_leg_raise = True

                    #check left leg is in thigh 
                    if self.r_check_leg_raise and not self.l_check_leg_raise:

                        if not self.l_leg_up:
                            if self.all_methods.l_ankle_y > self.all_methods.l_knee_y:
                                print("this is ok")
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please, lift your left leg up"],llist=llist)

                            if self.all_methods.l_ankle_y < self.all_methods.r_knee_y:
                                self.l_leg_up = True

                        if self.l_leg_up:
                            print("direct....................")
                            if self.all_methods.l_ankle_y < self.all_methods.r_knee_y:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please ,  keep your left leg on right thigh"],llist=llist)

                            elif (self.all_methods.l_ankle_y > self.all_methods.r_knee_y and 
                                self.all_methods.l_ankle_y < self.all_methods.l_hip_y ):
                                print("direct")
                                self.l_check_leg_raise = True

                    # check half pose detected
                    if self.l_check_leg_raise and not self.half_pose_detected:
                        if check_hip_knee_on_ground > 20:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please , lay your lower body on ground"],llist=llist)

                        elif self.left_elbow and 0 <= self.left_elbow <= 159:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please, keep your hands straight beside your left upper body"],llist=llist)

                        else:
                            self.all_methods.reset_after_40_sec()
                            half_pose_voice = self.all_methods.play_after_40_sec(["excellent , you completed half pose"],llist=llist)
                            if half_pose_voice:
                                self.half_pose_detected = True

                    if self.half_pose_detected and not self.seventy_pose_completed:

                        hip_wrist_x = (self.all_methods.l_wrist_x - self.all_methods.l_hip_x)
                        hip_wrist_y = (self.all_methods.l_wrist_y - self.all_methods.l_hip_y)

                        if check_hip_knee_on_ground and check_hip_knee_on_ground >= 40:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["keep your hip and knee on ground"],llist=llist)
                        
                        else:

                            if self.left_elbow and 160 <= self.left_elbow <= 180:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please , lay your elbow on ground"],llist=llist)

                            elif self.left_elbow and 0 <= self.left_elbow <= 129:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please bend your hand little more"],llist=llist)

                            else:
                                if hip_wrist_x > 10 and hip_wrist_y > 20:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please , touch your hip with your hand"],llist=llist)

                                else:
                                    if self.left_shoulder_hip and 0 <= self.left_shoulder_hip <= 15:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["take elbow, and head support , lift your upper body up"],llist=llist)
                                    
                                    elif self.left_shoulder_hip and 45 <= self.left_shoulder_hip <= 90:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please, down your upper body"],llist=llist)

                                    else:
                                        if self.head_position != "Right":
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please , set your head in sorrect position"],llist=llist)
                                        
                                        else:
                                            if (self.left_elbow and 130 <= self.left_elbow <= 159 and
                                                check_hip_knee_on_ground and check_hip_knee_on_ground <= 20 and
                                                self.head_position and self.head_position == "Right" and
                                                self.left_shoulder_hip and 16 <= self.left_shoulder_hip <= 44):

                                                self.all_methods.reset_after_40_sec()
                                                seventy_pose = self.all_methods.play_after_40_sec(["awesome , you completed 70 percent pose"],llist=llist)
                                                if seventy_pose:
                                                    self.seventy_pose_completed = True

                    if self.seventy_pose_completed:
                        toe_touch_y = (self.all_methods.r_wrist_y - self.all_methods.r_toe_y)
                        toe_touch_x = (self.all_methods.r_wrist_x - self.all_methods.r_toe_x)

                        if self.left_elbow and 0 <= self.left_elbow <= 159:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please keep hands straight, and touch your toes"],llist=llist)

                        else:
                            if toe_touch_x > 30 and toe_touch_y < 15:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please touch your toe with your hands"],llist=llist)

                            else:
                                return True

    def left_reverse_position(self,frames,llist):

        if self.right_shoulder_hip and 11 <= self.right_shoulder_hip <= 30:
            l_voice = ["good , stay in same position and wait for other instruction","perfect , back to sleep position"]
            if self.l_r_count < len(l_voice):
                self.all_methods.reset_after_40_sec()
                voice = self.all_methods.play_after_40_sec([l_voice[self.l_r_count]],llist=llist)
                if voice:
                    self.l_r_count += 1

            else:
                self.l_r_count = 1
            
        elif self.right_shoulder_hip and 0 <= self.right_shoulder_hip <= 15:

            self.all_methods.reset_voice()
            self.all_methods.play_voice(["good job you complete perfect , get relax and out off that pose"],llist=llist)
            return True

    def right_reverse_position(self,frames,llist):

        if self.left_shoulder_hip and 11 <= self.left_shoulder_hip <= 30:
            r_voice = ["good , stay in same position and wait for other instruction","perfect , back to sleep position"]
            if self.r_r_count < len(r_voice):
                self.all_methods.reset_after_40_sec()
                voice = self.all_methods.play_after_40_sec([r_voice[self.r_r_count]],llist=llist)
                if voice:
                    self.r_r_count += 1

            else:
                self.r_r_count = 1
            
        elif self.left_shoulder_hip and 0 <= self.left_shoulder_hip <= 15:

            self.all_methods.reset_voice()
            self.all_methods.play_voice(["good job you complete perfect , get relax and out off that pose"],llist=llist)
            return True

    def left_matyasana_name(self,frames):

        correct = (
            self.right_shoulder_hip and 11 <= self.right_shoulder_hip <= 44 and
            self.head_position and self.head_position == "Left" and
            self.right_elbow and 160 <= self.right_elbow <= 180 

        )
        if correct :
            return True
        # pass
    def right_matyasana_name(self,frames):

        correct = (
            self.left_shoulder_hip and 11 <= self.left_shoulder_hip <= 44 and
            self.head_position and self.head_position == "Right" and
            self.left_elbow and 160 <= self.left_elbow <= 180 
        )

        if correct :
            return True
        # pass

    def side_view_detect(self, frames, llist):
        result = None

        # Check if llist is valid and has enough data
        if not llist or len(llist[0]) < 2:
            # print("No human detected or invalid llist structure")
            return None

        side_view = self.all_methods.findSideView(
            frame=frames,
            FLAG_HEAD_OR_TAIL_POSITION="head",
            head=llist[0][1]
        )

        if side_view == "left":
            result = "left"
        elif side_view == "right":
            result = "right"
        else:
            result = None

        return result

def main():

    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    global  llist
    detect = matyasana()
    all_methods = allmethods()
    flag = False
    ready_for_exercise = False
    reverse_yoga = False
    checking_wrong = False
    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        
        if not flag:

            detect.pose_positions(frames,draw=False)
            llist = detect.pose_landmarks(frames,False)
            
            side_view = detect.side_view_detect(frames=frames,llist=llist)

            if side_view == "left":
                detect.matyasana(frames,llist,r_elbow=(12,14,16), r_hip=(12,24,26), r_knee=(24,26,28), r_shoulder=(14,12,24),l_elbow=(11,13,15),l_hip= (11,23,25),l_knee= (23,25,27), l_shoulder=(13,11,23),l_draw=False ,r_draw=True,l_v_draw=False,r_v_draw=False)
                # wrong_left = detect.wrong_matyasana_left(frames=frames,llist=llist,height=height,width=width)
                if not checking_wrong:
                    wrong_left = detect.wrong_left(frames=frames,llist=llist,height=height,width=width)
                    if wrong_left:
                        checking_wrong = True
                
                if checking_wrong:
                    # correct = detect.left_matyasana_name(frames=frames)
                    if not ready_for_exercise and not flag:
                        correct = detect.left_matyasana_name(frames=frames)
                        if correct:
                            ready_for_exercise = True
                            reverse_yoga = True

                if reverse_yoga  and not flag:
                    left_reverse = detect.left_reverse_position(frames=frames,llist=llist)
                    # ready_for_exercise = False
                    if left_reverse:
                        flag = True
        

            #Right Side
            elif side_view == "right":
                detect.matyasana(frames,llist,r_elbow=(12,14,16), r_hip=(12,24,26), r_knee=(24,26,28), r_shoulder=(14,12,24),l_elbow=(11,13,15),l_hip= (11,23,25),l_knee= (23,25,27), l_shoulder=(13,11,23),l_draw=True ,r_draw=False,l_v_draw=False,r_v_draw=False)
                
                # correct = detect.right_matyasana_name(frames=frames)
                if not checking_wrong:
                    wrong_right = detect.wrong_right(frames=frames,llist=llist,height=height,width=width)
                    if wrong_right:
                        checking_wrong = True

                if checking_wrong:
                    if not ready_for_exercise and not flag:
                        correct = detect.right_matyasana_name(frames=frames)
                        if correct:
                            ready_for_exercise = True
                            reverse_yoga = True

                    if reverse_yoga and not flag:
                        right_reverse = detect.right_reverse_position(frames=frames,llist=llist)
                        if right_reverse:
                            flag = True
                        
        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main() 