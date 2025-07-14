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

class sirasasana(yoga_exercise):
     
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

        self.check_sitting = False
        self.check_initial_position = False

        self.angle =0
        self.all_methods = allmethods()
        # self.stop = self.all_methods.stop_sometime()
        self.voice = VoicePlay()
        self.body_position = bodyPosition()
        self.head_pose_estimator = HeadPoseEstimator()
        # self.x_values = self.all_methods.all_x_values()

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

    def left_sirasasana(self, frames,llist, elbow, hip, knee,shoulder,right_knee1,right_elbow,draw =True):

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

    def right_sirasasana(self,frames,llist,elbow, hip, knee, shoulder,left_knee1,left_elbow,draw=True):

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


    def side_view_detect(self,frames,llist):
        
        self.all_methods.all_z_values(frames=frames,llist=llist)
        
        shoulder_diff_z = abs(int(self.all_methods.l_shoulder_z - self.all_methods.r_shoulder_z))
        hip_diff_z = abs(int(self.all_methods.l_hip_z - self.all_methods.r_hip_z))

        cv.putText(frames,f"{str(shoulder_diff_z)}" , (10,80),cv.FONT_HERSHEY_DUPLEX,2,(255,0,255),2)
        
        if shoulder_diff_z > 2 and hip_diff_z > 2:
            
            if self.all_methods.l_shoulder_z < self.all_methods.r_shoulder_z:
                if self.all_methods.l_hip_z < self.all_methods.r_hip_z:
                    return "left"
                
            elif self.all_methods.r_shoulder_z < self.all_methods.l_shoulder_z:
                if self.all_methods.r_hip_z < self.all_methods.l_hip_z:
                    return "right"
        else:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be in sitting position and turn any side you want"],llist=llist)
            return "forward" 

        
    
    def wrong_sirasasana_left(self,frames,llist,height,width): 
        self.all_methods.all_x_values(frames=frames,llist=llist)
        self.all_methods.all_y_values(frames=frames,llist=llist)
        self.all_methods.all_z_values(frames=frames,llist=llist)
        
        #initial position logic
        initial_position=[
                    self.left_shoulder and 27 <= self.left_shoulder <= 40 and
                    self.left_elbow and 150 <= self.left_elbow <= 180 and
                    self.right_elbow1 and 150 <= self.right_elbow1 and
                    self.left_hip and 85 <= self.left_hip <= 115 and 
                    self.left_knee and 0 <= self.left_knee <= 30 and
                    self.right_knee1 and 0 <= self.right_knee1 <= 30
                ]
        
        hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width)
        #check sitting
        if not self.check_sitting:
        
            if self.all_methods.is_person_standing_sitting == "sitting":
                self.check_sitting = True
                
            else:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["please be in sitting position"],llist=llist)
                
        elif self.check_sitting:

            if self.side_view_detect == "forward":
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["please turn your total body left , or , right "],llist=llist)
                
            else:
                #check initial position
                if not self.check_initial_position:
                    if initial_position:
                        self.check_initial_position = True
                        
                    elif not initial_position:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["you are not in initial position , please be in vajrasana pose"],llist=llist)
                
                #c CHECK sirasasana POSE 
                elif self.check_initial_position:
                    if initial_position:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["you are in initial position, and , come to donkey pose"],llist=llist)
                        
                    else:
                        # CHECK DONKEY POSE
                        if (self.left_knee and 50 <= self.left_knee <= 100 and
                            self.left_elbow and 160 <= self.left_elbow <= 180 and
                            self.left_hip and 50 <= self.left_hip <= 100 and
                            self.all_methods.nose_y < self.all_methods.l_elbow_y):
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please down your head , and , kepp touch your head to ground"],llist=llist)
                            
                        else:
                            # CHECK HEAD TOUCH TO GROUND OR NOT 
                            
                            hip_shoulder = self.all_methods.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width)
                            
                            if (hip_shoulder and 20 <= hip_shoulder <= 40 and
                                self.left_knee and 80 <= self.left_knee <= 110 and
                                self.left_hip and 60 <= self.left_hip <= 110 and
                                self.all_methods.l_wrist_y > self.all_methods.l_elbow_y):
                                
                                if self.all_methods.nose_y < self.all_methods.l_hip_y:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please touch your head to ground, take palms support ,and , maintain upper body slant"],llist=llist)
                                    
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["stretch your legs straight in same position"],llist=llist)
                                
                            else:
                                # CHECK LEGS STRAIGHT OR NOT
                                if (self.left_knee and 160 <= self.left_knee <= 180 and
                                    self.left_elbow and 65 <= self.left_elbow <= 110 and
                                    self.all_methods.nose_y > self.all_methods.l_knee_y and
                                    self.all_methods.l_wrist_y > self.all_methods.l_elbow_y):
                                    
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["keep head , foot , palms on ground , bend your hip closer"],llist=llist)
                                
                                else:
                                    # CHECK LEGS IS IN FOLD POSITION OR NOT
                                    if (self.left_knee and 110 <= self.left_knee <= 150 and
                                        self.right_knee1 and 110 <= self.right_knee1 <= 150 and
                                        self.left_hip and 10 <= self.left_hip <= 30 and
                                        self.all_methods.nose_y > self.all_methods.l_knee_y and
                                        self.all_methods.l_wrist_y > self.all_methods.l_elbow_y):
                                        
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["stand on head , take palms support , Move your knees a little inwards towards your abdomen , and lift your both legs"])
                                    
                                    else:
                                        # CHECK LEGS IS RAISE OR NOT
                                        if( self.left_knee and  50 <= self.left_knee <= 115 and
                                           self.right_knee1 and 50 <= self.right_knee1 <= 115 and
                                           hip_shoulder and 70 <= hip_shoulder <= 90 and
                                           self.left_elbow and 70 <= self.left_shoulder <= 100 ):
                                            
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["stand on head , take palms support, your upper leg is perpendicular to floor"],llist=llist)
                                            
                                        else:
                                            # CHECK LEGS IN PERPENDICULAR OR NOT
                                            if (self.left_knee and 20 <= self.left_knee <= 50 and
                                                self.right_knee1 and 15 <= self.right_knee1 <= 50 and
                                                self.left_hip and 70 <= self.left_hip <= 100 and
                                                hip_knee and 0 <= hip_knee <= 20):
                                            
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please raise your right leg"],llist=llist)
                    

                                            else:
                                                if (self.left_knee and 0 <= self.left_knee <= 159):
                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["keep your left leg straight"],llist=llist)
                                                    
                                                else:
                                                    if (self.right_knee1 and 0 <= self.right_knee1 <= 159):
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["keep your right leg straight"],llist=llist)
                                                        
                                                    else:
                                                        if(self.left_hip and 0 <= self.left_hip <= 159):
                                                            self.all_methods.reset_after_40_sec()
                                                            self.all_methods.play_after_40_sec(["keep your body straight"],llist=llist)
                                                            
                                                        else:
                                                            if(self.left_elbow  and 0 <= self.left_elbow <= 59):
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["please keep your elbow 90 degrees"],llist=llist)
                                                                
                                                            elif (self.left_elbow and 101 <= self.left_elbow <= 180):
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["keep your elbow in 90 degrees"],llist=llist)
                                                                
                                                            else:
                                                                if (self.left_shoulder and 0 <= self.left_shoulder <= 30):
                                                                    self.all_methods.reset_after_40_sec()
                                                                    self.all_methods.play_after_40_sec(["keep your hands 90 degrees"],llist=llist)
                                                                    
                                                                elif (self.left_shoulder and 101 <= self.left_shoulder <= 180):
                                                                    self.all_methods.reset_after_40_sec()
                                                                    self.all_methods.play_after_40_sec(["keep your hands 90 degrees"],llist=llist)
                                                                    
                                                                else:
                                                                    return True 
                                                    
    def wront_right_sirasasana(self,frames,llist,height,width):
        
        self.all_methods.all_x_values(frames=frames,llist=llist)
        self.all_methods.all_y_values(frames=frames,llist=llist)
        self.all_methods.all_z_values(frames=frames,llist=llist)

        #initial position logic
        initial_position=[
                    self.right_shoulder and 27 <= self.right_shoulder <= 40 and
                    self.right_elbow and 150 <= self.right_elbow <= 180 and
                    self.left_elbow1 and 150 <= self.left_elbow1 and
                    self.right_hip and 85 <= self.right_hip <= 115 and 
                    self.right_knee and 0 <= self.right_knee <= 30 and
                    self.left_knee1 and 0 <= self.left_knee1 <= 30
                ]
        
        hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width)
        
        #check sitting
        if not self.check_sitting:
        
            if self.all_methods.is_person_standing_sitting == "sitting":
                self.check_sitting = True
                
            else:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["please be in sitting position"],llist=llist)
                
        elif self.check_sitting:

            if self.side_view_detect == "forward":
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["please turn your total body right , or , right "],llist=llist)
                
            else:
                
                #check initial position
                if not self.check_initial_position:
                    if initial_position:
                        self.check_initial_position = True
                        
                    elif not initial_position:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["you are not in initial position , please be in sirasasana pose"],llist=llist)
                
                #c CHECK sirasasana POSE 
                elif self.check_initial_position:
                    if initial_position:
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["you are in initial position, and , come to donkey pose"],llist=llist)
                        
                    else:
                        # CHECK DONKEY POSE
                        if (self.right_knee and 50 <= self.right_knee <= 100 and
                            self.right_elbow and 160 <= self.right_elbow <= 180 and
                            self.right_hip and 50 <= self.right_hip <= 100 and
                            self.all_methods.nose_y < self.all_methods.r_elbow_y):
                            
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please down your head , and , carefully touch your head to ground"],llist=llist)
                            
                        else:
                            # CHECK HEAD TOUCH TO GROUND OR NOT 
                            
                            hip_shoulder = self.all_methods.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width)
                            
                            if (hip_shoulder and 20 <= hip_shoulder <= 40 and
                                self.right_knee and 80 <= self.right_knee <= 110 and
                                self.right_hip and 60 <= self.right_hip <= 110 and 
                                self.all_methods.r_wrist_y > self.all_methods.r_elbow_y):
                                
                                if self.all_methods.nose_y < self.all_methods.r_hip_y:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please carefully touch your head to ground, take palms support ,and , maintain upper body slant"],llist=llist)
                                    
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["stretch your legs straight in same position"],llist=llist)
                                
                            else:
                                # CHECK LEGS STRAIGHT OR NOT
                                if (self.right_knee and 160 <= self.right_knee <= 180 and
                                    self.right_elbow and 65 <= self.right_elbow <= 110 and
                                    self.all_methods.nose_y > self.all_methods.r_knee_y and
                                    self.all_methods.r_wrist_y > self.all_methods.r_elbow_y):
                                    
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["keep head , toe , palms on ground ,carefully bend your hip , keep closer knees and upper body"],llist=llist)
                                
                                else:
                                    # CHECK LEGS IS IN FOLD POSITION OR NOT
                                    if (self.right_knee and 110 <= self.right_knee <= 150 and
                                        self.left_knee1 and 110 <= self.left_knee1 <= 150 and
                                        self.right_hip and 10 <= self.right_hip <= 30 and
                                        self.all_methods.nose_y > self.all_methods.r_knee_y and
                                        self.all_methods.r_wrist_y > self.all_methods.r_elbow_y):
                                        
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["slowly, and, carefully stand on head ,please take palms support , fold your legs closer ,slowly and lift your both legs , at medium height"])
                                    
                                    else:
                                        # CHECK LEGS IS RAISE OR NOT
                                        if( self.right_knee and  50 <= self.right_knee <= 115 and
                                           self.left_knee1 and 50 <= self.left_knee1 <= 120 and
                                           hip_shoulder and 70 <= hip_shoulder <= 90 and
                                           self.right_elbow and 70 <= self.right_shoulder <= 100 ):
                                            
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["slowly take palms support , please raise your both legs straight"],llist=llist)
                    
                                        else:
                                            # CHECK LEGS IN PERPENDICULAR OR NOT
                                            if (self.left_knee and 20 <= self.left_knee <= 50 and
                                                self.right_knee1 and 15 <= self.right_knee1 <= 50 and
                                                self.left_hip and 70 <= self.left_hip <= 100 and
                                                hip_knee and 0 <= hip_knee <= 20):
                                            
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please raise your right leg"],llist=llist)
                    

                                            else:
                                                if (self.right_knee and 0 <= self.right_knee <= 159):
                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["keep your right leg straight"],llist=llist)
                                                    
                                                else:
                                                    if (self.left_knee1 and 0 <= self.left_knee1 <= 159):
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["keep your left leg straight"],llist=llist)
                                                        
                                                    else:
                                                        if(self.right_hip and 0 <= self.right_hip <= 159):
                                                            self.all_methods.reset_after_40_sec()
                                                            self.all_methods.play_after_40_sec(["keep your body straight"],llist=llist)
                                                            
                                                        else:
                                                            if(self.right_elbow  and 0 <= self.right_elbow <= 59):
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["please keep your elbow 90 degrees"],llist=llist)
                                                                
                                                            elif (self.right_elbow and 101 <= self.right_elbow <= 180):
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["keep your elbow in 90 degrees"],llist=llist)
                                                                
                                                            else:
                                                                if (self.right_shoulder and 0 <= self.right_shoulder <= 30):
                                                                    self.all_methods.reset_after_40_sec()
                                                                    self.all_methods.play_after_40_sec(["keep your hands 90 degrees"],llist=llist)
                                                                    
                                                                elif (self.right_shoulder and 101 <= self.right_shoulder <= 180):
                                                                    self.all_methods.reset_after_40_sec()
                                                                    self.all_methods.play_after_40_sec(["keep your hands 90 degrees"],llist=llist)
                                                                    
                                                                else:
                                                                    return True
        
    def right_sirasasana_name(self,frames):
        
        correct = (self.right_knee and 160 <= self.right_knee <= 180 and
                   self.left_knee1 and 160 <= self.left_knee1 <= 180 and
                   self.right_hip and 160 <= self.right_hip <= 180 and
                   self.right_elbow and 70 <= self.right_elbow <= 110 and
                   self.right_shoulder and 70 <= self.right_shoulder <= 110 and
                   self.all_methods.r_hip_y > self.all_methods.r_ankle_y)
        
        if correct :
            return True
        
    def left_sirasasana_name(self,frames):
        
        correct = (self.left_knee and 160 <= self.left_knee <= 180 and
                   self.right_knee1 and 160 <= self.right_knee1 <= 180 and
                   self.left_hip and 160 <= self.left_hip <= 180 and
                   self.left_elbow and 70 <= self.left_elbow <= 100 and
                   self.left_shoulder and 70 <= self.left_shoulder <= 100 and
                   self.all_methods.l_hip_y > self.all_methods.l_ankle_y)
        
        if correct :
            return True
        
    def left_reverse(self,frames):
        
        if (self.left_knee and 160 <= self.left_knee <= 180):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["good stay in same same position"],llist=llist)
            
        elif(self.left_knee and 160 <= self.left_knee <= 180):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["very good get relax and fold your legs"],llist=llist)
       
    
    def right_reverse(self,frames):
        
        if (self.right_knee and 160 <= self.right_knee <= 180):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["good stay in same same position"],llist=llist)
            
        elif(self.right_knee and 160 <= self.right_knee <= 180):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["very good get relax and fold your legs"],llist=llist)
            
                     
def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    global  llist
    detect = sirasasana()
    all_methods = allmethods()
    # ref_video = cv.VideoCapture("second videos/sirasasana.mp4")
    flag = False
    ready_for_exercise = False
    reverse_yoga = False
    checking_wrong = False
    
    while True:

        isTrue,frames = video_capture.read()
        # print(frames)
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        
        
        if not flag:

            detect.pose_positions(frames,draw=False)
            llist = detect.pose_landmarks(frames,False)

            if len(llist) is None:
                return
            
            side_view = detect.side_view_detect(frames=frames,llist=llist)
            if side_view == "left":
                detect.left_sirasasana(frames=frames,llist=llist,elbow=(11,13,15), hip=(11,23,25), knee=(23,25,27), shoulder=(13,11,23),right_knee1=(24,26,28),right_elbow=(12,14,16),draw=True)
                if not checking_wrong:
                    wrong_left = detect.wrong_sirasasana_left(frames=frames,llist=llist,height=height,width=width)
                    if wrong_left:
                        checking_wrong = True
                
                if checking_wrong:
                    if not ready_for_exercise and not flag:
                        correct = detect.left_sirasasana_name(frames=frames)
                        if correct:
                            ready_for_exercise = True
                            reverse_yoga = True

                if reverse_yoga  and not flag:
                    left_reverse = detect.left_reverse(frames=frames,llist=llist)
                    if left_reverse:
                        flag = True
    

            #Right Side
            elif side_view == "right":
                detect.right_sirasasana(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee1=(23,25,27),left_elbow=(11,13,15),draw=True)                        

                if not checking_wrong:
                    wrong_right = detect.wront_right_sirasasana(frames=frames,llist=llist,height=height,width=width)
                    if wrong_right:
                        checking_wrong = True

                if checking_wrong:
                    if not ready_for_exercise and not flag:
                        correct = detect.right_sirasasana_name(frames=frames)
                        if correct:
                            ready_for_exercise = True
                            reverse_yoga = True

                    if reverse_yoga and not flag:
                        right_reverse = detect.right_reverse(frames=frames,llist=llist)
                        if right_reverse:
                            flag = True

        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()