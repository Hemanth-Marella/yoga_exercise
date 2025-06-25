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

class vajrasana:
     
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

        self.right_count = False
        self.left_count = False
        self.pose_completed  = False
        self.initial_position = False

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

    def left_vajrasana(self, frames,llist, elbow, hip, knee,shoulder,right_knee1,right_elbow,draw =True):

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
            if elbow_coords:
                cv.putText(frames, str(int(self.left_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.left_hip)), (hip_coords[2]+10, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.left_knee)), (knee_coords[2]+10, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if points_cor8:
                cv.putText(frames, str(int(self.left_shoulder)), (points_cor8[2]+10, points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if right_knee1_coords:
                cv.putText(frames, str(int(self.right_knee1)), (right_knee1_coords[2]-20, right_knee1_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.right_knee1,self.head_position #self.left_knee_y,self.left_wrist_y,self.ground_left,self.ground_left_min

    def right_vajrasana(self,frames,llist,elbow, hip, knee, shoulder,left_knee1,left_elbow,draw=True):

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
            if elbow_coords:
                cv.putText(frames, str(int(self.right_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.right_hip)), (hip_coords[2]-20, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.right_knee)), (knee_coords[2]-20, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if points_cor7:
                cv.putText(frames, str(int(self.right_shoulder)), (points_cor7[2]+10, points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            if left_knee1_coords:
                cv.putText(frames, str(int(self.left_knee1)), (left_knee1_coords[2]-20, left_knee1_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,left_knee1,self.head_position#self.right_knee_y,self.right_wrist_y,self.ground_right,self.ground_right_min
    

    def slope(self,frames,llist,height,width):

        if llist is None:
            return None

        self.under_leg_slope = self.all_methods.slope(frames=frames,lmlist=llist,point1=25,point2=27,height=height,width=width,draw=True)
        
        return self.under_leg_slope

    
    def wrong_vajrasana_left(self,frames,llist,height,width): 

        if not self.left_hip and not self.left_elbow and not self.left_knee and not self.left_shoulder and not self.right_knee1 and not self.left_shoulder and not self.under_leg_slope:
            return
            
        check_initial_position = (self.left_knee and 160 <= self.left_knee <= 180 and
            self.right_knee1 and 160 <= self.right_knee1 <= 180)
        
        if not self.left_count and check_initial_position:

            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["you are in initial position and start yoga ,and keep your left leg back"], llist=llist)
            self.all_methods.reset_after_40_sec()
            self.left_count = True
            self.initial_position = True
            return 

        
        elif not self.initial_position and not check_initial_position:

            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["you are not in initial position , keep straight your legs which side you are in"],llist=llist)
            # self.initial_position 
            return
        

        if self.left_count:
                    
                if (self.left_knee and 160 <= self.left_knee <= 180 and
                    self.right_knee1 and 160 <= self.right_knee1 <= 180):
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are in initial position and start vajrasana ,and keep your left leg back"],llist=llist)

                else:
                    # STEP 2: Check left knee position
                    if self.left_knee:
                        if  (80 <= self.left_knee <= 159):
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please fold your left leg back and sit on that leg,, like in reference video"], llist=llist)
                            return
                        elif 31 <= self.left_knee <= 79:
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["fold your left leg back and sit on that leg,, like in reference video"], llist=llist)
                            return

                        else:
                            # STEP 3: Check right knee position
                            if self.right_knee1:
                                if  (80 <= self.right_knee1 <= 180):
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please fold your right leg back and sit on that leg,, like in reference video"], llist=llist)
                                    return
                                if 31 <= self.right_knee1 <= 79:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["fold your right leg back and sit on that leg,, like in reference video"], llist=llist)
                                    return
                                
                                else:

                                    # STEP 4: Elbow check
                                    if self.left_elbow and 0 <= self.left_elbow <= 149:
                                        
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please keep your elbows straight, touch your hands to knees"], llist=llist)
                                        return
                                    
                                    else:

                                        #STEP 4.1:RIGHT ELBOW CHECK

                                        if self.right_elbow1 and 0 <= self.right_elbow1 <= 149:
                                        
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please keep your elbows straight, touch your hands to knees"], llist=llist)
                                            return
                                    
                                        else:

                                            # STEP 5: Shoulder position
                                            if self.left_shoulder:
                                                if 0 <= self.left_shoulder <= 26:
                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["move your hands little up forward and keep straight"], llist=llist)
                                                    return
                                                elif 41 <= self.left_shoulder <= 180:
                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["move your hands forward inwards slight"], llist=llist)
                                                    return
                                    
                                                else:

                                                    # STEP 6: Hip angle
                                                    if self.left_hip:
                                                        if 0 <= self.left_hip <= 79 or 116 <= self.left_hip <= 180:
                                                            self.all_methods.reset_after_40_sec()
                                                            self.all_methods.play_after_40_sec(["please keep your upper body straight"], llist=llist)
                                                            return
                                                        
                                                        else:

                                                            # STEP 7: Head position
                                                            if self.head and self.head_position != "Left":
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["please turn your head left side"], llist=llist)
                                                                return
                                                            
                                                            else:
                                                                return True


    def wront_right_vajrasana(self,frames,llist,height,width):

        
        if not self.right_hip and not self.right_elbow and not self.right_knee and not self.right_shoulder and not self.left_knee1 and not self.right_shoulder and not self.under_leg_slope:
            return None

        check_initial_position = (self.left_knee1 and 160 <= self.left_knee1 <= 180 and
            self.right_knee and 160 <= self.right_knee <= 180)
        
        if not self.right_count and check_initial_position:

            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["you are in initial position and start yoga ,and keep your left leg back"], llist=llist)
            self.right_count = True
            self.initial_position = True

            return 
        
        elif not self.initial_position and not check_initial_position:

            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["you are not in initial position , keep straight your legs which side you are in"],llist=llist)
            return

        if self.right_count:

                # self.all_methods.reset_after_40_sec()
                # self.all_methods.play_after_40_sec(["please keep your legs straight on right side"],llist=llist)

                if (self.left_knee1 and 160 <= self.left_knee1 <= 180 and
                        self.right_knee and 160 <= self.right_knee <= 180):
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are in initial position and start vajrasana ,and keep your right leg back"], llist=llist)
                    
                else:
                        # STEP 2: Check left knee position
                        if self.right_knee:
                            if  (80 <= self.right_knee <= 159):
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please fold your right leg back and sit on that leg,, like in reference video"], llist=llist)
                                return
                            elif 31 <= self.right_knee <= 79:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["fold your right leg back and sit on that leg,, like in reference video"], llist=llist)
                                return

                            else:
                                # STEP 3: Check right knee position
                                if self.left_knee1:
                                    if  (80 <= self.left_knee1 <= 180):
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please fold your left leg back and sit on that leg,, like in reference video"], llist=llist)
                                        return
                                    elif 31 <= self.left_knee1 <= 79:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["fold your left leg back and sit on that leg,, like in reference video"], llist=llist)
                                        return
                                
                                    else:

                                        # STEP 4: Elbow check
                                        if self.right_elbow and 0 <= self.right_elbow <= 149:
                                            
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please keep your elbows straight, touch your hands to knees"], llist=llist)
                                            return
                                        else:

                                            #STEP 4.1:LEFT ELBOW CHECK

                                            if self.left_elbow1 and 0 <= self.left_elbow1 <= 149:
                                            
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please keep your elbows straight, touch your hands to knees"], llist=llist)
                                                return
                                        
                                            else:

                                                # STEP 5: Shoulder position
                                                if self.right_shoulder:
                                                    if 0 <= self.right_shoulder <= 26:
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["move your hands little up forward and keep straight"], llist=llist)
                                                        return
                                                    elif 41 <= self.right_shoulder <= 180:
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["move your hands forward inwards slight"], llist=llist)
                                                        return
                                        
                                                    else:

                                                        # STEP 6: Hip angle
                                                        if self.right_hip:
                                                            if 0 <= self.right_hip <= 79 or 116 <= self.right_hip <= 180:
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["please keep your upper body straight"], llist=llist)
                                                                return
                                                            
                                                            else:

                                                                # STEP 7: Head position
                                                                if self.head and self.head_position != "Right":
                                                                    self.all_methods.reset_after_40_sec()
                                                                    self.all_methods.play_after_40_sec(["please turn your head left side"], llist=llist)
                                                                    return

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
            self.all_methods.play_after_40_sec(["please turn total your body left, or , right, and stretch your legs straight"],llist=llist)
            return False  


    def left_reverse_to_strating_position(self,frames,llist):

        count = 0

        if not self.right_knee1 and not self.left_knee:
            return 

        # if self.check_sitting == "sitting":

        check_last_before_position = (self.right_knee1 and 0 <= self.right_knee1 <= 30 and  
            self.left_knee and 0 <= self.left_knee <= 30 )
        
        check_last_position = (self.left_knee and 160 <= self.left_knee <= 180 and
                self.right_knee1 and 160 <= self.right_knee1 <= 180)
        
        if check_last_before_position:
            if count == 0:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["good job you complete perfectly, wait for one more instruction"],llist=llist)
                count += 1

            elif count == 1:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec([" , get relax , keep your legs straight on which side your are in"],llist=llist)
                count = 1


        elif ((31 <= self.right_knee1 <= 159) and (31<= self.left_knee <= 159)):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["keep your legs straight on left side"],llist=llist)
            

        elif (31 <= self.right_knee1 <= 159):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["keep your right leg straight on left side"],llist=llist)

        elif (31 <= self.left_knee <= 159):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["keep your left leg straight on left side"],llist=llist)

        
        elif check_last_position:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["you are in final position and please be in standing position"],llist=llist)
            self.pose_completed = True
            

            return True
        
    
    def right_reverse_to_strating_position(self,frames,llist):
        count = 0

        if not self.right_knee and not self.left_knee1:
            return 

        # if self.check_sitting == "sitting":

        check_last_before_position = (self.right_knee and 0 <= self.right_knee <= 30 and  
            self.left_knee1 and 0 <= self.left_knee1 <= 30 )
        
        check_last_position = (self.left_knee1 and 160 <= self.left_knee1 <= 180 and
                self.right_knee and 160 <= self.right_knee <= 180)
        
        if check_last_before_position:
            if count == 0:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["good job you complete perfectly, wait for one more instruction"],llist=llist)
                count += 1

            elif count == 1:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec([" , get relax , keep your legs straight on which side your are in"],llist=llist)
                count = 1
                

        elif ((31 <= self.right_knee <= 159) and (31<= self.left_knee1 <= 159)):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["keep your legs straight on right side"],llist=llist)
            # return False

        elif (31 <= self.right_knee <= 159):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["keep your right leg straight on right side"],llist=llist)

        elif (31 <= self.left_knee1 <= 159):
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["keep your left leg straight on right side"],llist=llist)

 
        elif check_last_position:
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["you are in final position and please be in standing position"],llist=llist)
            self.pose_completed = True
            # return False

            return True

        
    def left_vajrasana_name(self,frames): 

        correct =(
            self.left_shoulder and 27 <= self.left_shoulder <= 40 and
            self.left_elbow and 150 <= self.left_elbow <= 180 and
            self.right_elbow1 and 150 <= self.right_elbow1 and
            self.left_hip and 85 <= self.left_hip <= 115 and 
            self.left_knee and 0 <= self.left_knee <= 30 and
            self.right_knee1 and 0 <= self.right_knee1 <= 30 and 
            self.under_leg_slope and 0 <= self.under_leg_slope <= 25 and
            # self.ground_left and self.ground_left_min <= self.left_knee_y <= self.ground_left  and
            # self.left_knee_y and self.left_knee_min_y < self.left_wrist_y < self.left_knee_y and
            self.head_position and self.head_position == "Left")
        
        if correct:
           
            cv.putText(frames,str("vajrasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                
            return True
        
        return False

    
    def right_vajrasana_name(self,frames):

        correct = (self.right_elbow and 140 <= self.right_elbow <= 180 and   
                   self.left_elbow1 and 150 <= self.left_elbow1 and        
            self.right_hip and 85 <= self.right_hip <= 115 and            
            self.right_knee and 0 <= self.right_knee <= 30 and  
            self.left_knee1 and 0 <= self.left_knee1 <= 30 and                                    
            self.right_shoulder and 27 <= self.right_shoulder <= 40 and
            self.under_leg_slope and 0 <= self.under_leg_slope <= 25 and
            # self.ground_right and self.ground_right_min <= self.right_knee_y <= self.ground_right  and
            # self.right_knee_y and self.right_knee_min_y < self.right_wrist_y < self.right_knee_y and
            self.head_position and self.head_position == "Right")
        if correct:

            cv.putText(frames,str("vajrasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

            return True
        
        return False
    
    
    def draw_countdown_circle_on_image(self, img, countdown_time, center = (550, 55), radius = 45):
        circle_color = (50, 50, 50)  # dark gray color
        thickness = 3
        font = cv.FONT_HERSHEY_COMPLEX
        font_thickness = 3
        font_color = (255, 255, 0)  # yellow

        # Draw the watch face circle
        cv.circle(img, center, radius, circle_color, thickness)

        text = str(countdown_time)

        # Max width allowed for text = 80% of diameter
        max_text_width = int(radius * 2 * 0.8)

        # Start with a large font scale and decrease until text fits inside circle
        font_scale = 2
        while font_scale > 0.1:
            text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
            if text_size[0] <= max_text_width:
                break
            font_scale -= 0.1

        # Center the text inside the circle
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2

        cv.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

        return img

def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    global  llist
    detect = vajrasana()
    all_methods = allmethods()
    ref_video = cv.VideoCapture("second videos/vajrasana.mp4")
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

        ret_ref, ref_frame = ref_video.read()
        if not ret_ref:
            ref_video.set(cv.CAP_PROP_POS_FRAMES, 0)  # loop video
            ret_ref, ref_frame = ref_video.read()

        # Resize reference video frame and place it at top-left
        if ret_ref:
            ref_frame = cv.resize(ref_frame, (400, 300))
            h, w, _ = ref_frame.shape
            frames[0:h, 0:w] = ref_frame  # Overlay in corner
        # img = cv.imread("images/image1.webp")
        # resized_img = cv.resize(img, None, fx=3.0, fy=3.0, interpolation=cv.INTER_LINEAR)
        
        if not flag:

            detect.pose_positions(frames,draw=False)
            llist = detect.pose_landmarks(frames,False)
            vajrasana_slope = detect.slope(frames=frames,llist=llist,height=height,width=width)
            sitting_detect = detect.check_sitting(frames=frames,llist=llist,height=height,width=width)
            side_view = detect.check_side_view(frames=frames,llist=llist,height=height,width=width,left_knee_angle=(23,25,27),right_knee_angle=(24,26,28))

            if sitting_detect:

                if side_view == "left":
                    detect.left_vajrasana(frames=frames,llist=llist,elbow=(11,13,15), hip=(11,23,25), knee=(23,25,27), shoulder=(13,11,23),right_knee1=(24,26,28),right_elbow=(12,14,16),draw=False)
                    # wrong_left = detect.wrong_vajrasana_left(frames=frames,llist=llist,height=height,width=width)
                    if not checking_wrong:
                        wrong_left = detect.wrong_vajrasana_left(frames=frames,llist=llist,height=height,width=width)
                        if wrong_left:
                            checking_wrong = True
                    
                    if checking_wrong:
                        # correct = detect.left_vajrasana_name(frames=frames)
                        if not ready_for_exercise and not flag:
                            correct = detect.left_vajrasana_name(frames=frames)
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
                    detect.right_vajrasana(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee1=(23,25,27),left_elbow=(11,13,15),draw=False)                        
                    
                    # correct = detect.right_vajrasana_name(frames=frames)
                    if not checking_wrong:
                        wrong_right = detect.wront_right_vajrasana(frames=frames,llist=llist,height=height,width=width)
                        if wrong_right:
                            checking_wrong = True

                    if checking_wrong:
                        if not ready_for_exercise and not flag:
                            correct = detect.right_vajrasana_name(frames=frames)
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