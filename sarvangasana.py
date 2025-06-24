import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math

from threading import Thread
from allMethods import allmethods
from voiceModule import VoicePlay
from face_detect import HeadPoseEstimator

class sarvangasana:
     
    def __init__(self,mode = False,mindetectconf=0.5,mintrcackconf=0.5):
        self.timeTaking_for_same_current_position = 0
        self.oneTimeRun_flag_for_always_sleeping = 0

        self.voice_stop_timer = 0

        self.voice_cooldown_active = False
        self.voice_cooldown_counter = 0


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
        self.calculate_angle = self.all_methods.calculate_angle
        self.head_detect = self.all_methods.head_detect
        self.left_heel_detect = self.all_methods.left_heel_detect
        self.right_heel_detect = self.all_methods.right_heel_detect
        self.head_pose = HeadPoseEstimator()
        self.voice = VoicePlay()

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        self.round = True

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
                px,py,pz = int(poselms.x * self.w) , int(poselms.y * self.h),(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lpslist 
    

    def right_sarvangasana(self,frames,llist,elbow, hip, knee, shoulder,left_knee,right_shoulder_ear, draw = True):

        # self.all_methods.play_voice("move to next exercise")

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
        self.right_shoulder_ear,shoulder_ear_coords = self.all_methods.calculate_angle(frames=frames,lmList=llist,points=right_shoulder_ear)

        # if draw:
        #     if elbow_coords:
        #         cv.putText(frames, str(int(self.right_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        #     if hip_coords:
        #         cv.putText(frames, str(int(self.right_hip)), (hip_coords[2]-20, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        #     if knee_coords:
        #         cv.putText(frames, str(int(self.right_knee)), (knee_coords[2]-20, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        #     if points_cor7:
        #         cv.putText(frames, str(int(self.right_shoulder)), (points_cor7[2]+10, points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        #     if left_knee1_coords:
        #         cv.putText(frames,str(int(self.left_knee1)),(left_knee1_coords[2]+10 , left_knee1_coords[3]+10),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            # if shoulder_ear_coords:
            #     cv.putText(frames,str(int(self.right_shoulder_ear)),(shoulder_ear_coords[2]+10 , shoulder_ear_coords[3]+10),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)


        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,self.head_position
       
    def left_sarvangasana(self,frames,llist,elbow, hip, knee,shoulder,right_knee,left_shoulder_ear,draw = True):


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
        # self.right_knee,right_hip_coords = self.all_methods.calculate_angle(frames=frames,lmList=llist,points=right_hip,draw=True)
        self.left_shoulder_ear,shoulder_ear_coords = self.all_methods.calculate_angle(frames=frames,lmList=llist,points=left_shoulder_ear)
        
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
            # if shoulder_ear_coords:
            #     cv.putText(frames,str(int(self.left_shoulder_ear)),(shoulder_ear_coords[3]+10 , shoulder_ear_coords[3]+10),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)


        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.head_position


    def head_heel_points(self,frames,llist,face_points,left_heel_point,right_heel_point,draw = True):

        self.head_points = self.head_detect(frames=frames,lmlist=llist,points=face_points) 

        self.left_heel_points = self.left_heel_detect(frames=frames,lmlist=llist,point=left_heel_point)

        self.right_heel_points = self.right_heel_detect(frames=frames,lmlist=llist,point=right_heel_point)

        if None in (self.head_points,self.left_heel_points,self.right_heel_points):
            return

        if draw:
            if self.head_points:
                self.head_x = self.head_points[0]
                self.head_y = self.head_points[1]
                self.head_z = self.head_points[2]

            if self.left_heel_points:
                self.left_heel_x = self.left_heel_points[0]
                self.left_heel_y = self.left_heel_points[1]
                self.left_heel_z = self.left_heel_points[2]

            if self.right_heel_points:
                self.right_heel_x = self.right_heel_points[0]
                self.right_heel_y = self.right_heel_points[1]
                self.right_heel_z = self.right_heel_points[2]


        return self.head_x,self.head_y,self.left_heel_x,self.left_heel_y
    

    def wrong_left(self,frames):


        #this one is for left knee
        left_knee_correct = (self.left_knee)
        if left_knee_correct:
            left_knee = (self.left_knee and 0 <= self.left_knee <= 159)
            if left_knee:
                self.all_methods.play_voice(["lift your legs up"])
                # self.all_methods.voice("your left leg is not in correct position please be straight")

        right_knee1_correct = (self.right_knee1)
        if right_knee1_correct:
            right_knee1_ = (self.right_knee1 and 0 <= self.right_knee1 <= 159)
            if right_knee1_:
                self.all_methods.play_voice(["lift your legs up"])
                # self.all_methods.voice("you legs are in straight position please bend your legs front side")


        #this one is for hip back side
        left_hip = (self.left_hip)
        if left_hip:
            # left_hip_bend = (self.left_hip and 161 <= self.left_hip <= 180)
            # if left_hip_bend:
            #     self.all_methods.play_voice(["just bend down your hip"])
                # self.all_methods.voice("your hip not bend pefect move your hip to front and bend back side")

            #this one is for hip front side
            left_hip_up = (self.left_hip and 0 <= self.left_hip <= 129)
            if left_hip_up:
                self.all_methods.play_voice(["lift your hip up"])
                # self.all_methods.voice("your hip is bend in fron position so you are in bad position so bend your body back side")

        
        #left elbow
        left_elbow_correct = (self.left_elbow)
        if left_elbow_correct:
            left_elbow_open = (self.left_elbow and 0 <= self.left_elbow <= 60)
            if left_elbow_open:
                self.all_methods.play_voice(["open your hands and touch your palm to hip"])

            left_elbow_close = (self.left_elbow and 101 <= self.left_elbow <= 180)
            if left_elbow_close:
                self.all_methods.play_voice(["close your hands and touch your plam to hip"])

        #this one is for left shoulder
        left_shoulder_correct = (self.left_shoulder)
        if left_shoulder_correct:
            left_shoulder_open = (self.left_shoulder and 0<= self.left_shoulder <= 60)
            if left_shoulder_open:
                self.all_methods.play_voice(["open your shoulders"])

            left_shoulder_close = (self.left_shoulder and 101 <= self.left_shoulder <= 180)
            if left_shoulder_close:
                self.all_methods.play_voice(["close your shoulders"])


        #this one is for head
        head_correct = (self.head_y and self.left_heel_y)
        if head_correct:
            left = (self.head_y < self.left_heel_y)
            if left:
                self.all_methods.play_voice(["legs is in upside and head is touch to ground"])
                # self.all_methods.voice("your body is in left position but your head is not cross the your heel so do the head to cross your heel back side")

        else:
            return 
    

    def wrong_right(self,frames):

        #this one is for right knee
        right_knee_correct = (self.right_knee)
        if right_knee_correct:
            right_knee_straight = (self.right_knee and 0 <= self.right_knee <= 159)
            if right_knee_straight:
                self.all_methods.play_voice(["lift your legs up your legs must be straight"])
                

        left_knee1_correct = (self.left_knee1)    
        if left_knee1_correct:
            left_knee1 =(self.left_knee1 and 0 <= self.left_knee1 <= 159)
            if left_knee1:
                self.all_methods.play_voice(["lift your legs up your legs must be straight"])


        hip_leg_ = (self.right_hip and self.right_knee)
        if hip_leg_:
            hip_leg_correct = ((130 <= self.right_hip <= 160 ) and (0 <= self.right_knee <=  159))
            if hip_leg_correct:
                self.all_methods.play_voice(["your hip is is correct position please be staright your legs up"])

            leg_hip_correct = ((0 <= self.right_hip <= 129 ) and (160 <= self.right_knee <=  180))
            if leg_hip_correct:
                self.all_methods.play_voice(["your legs is in correct position please be staright your hip"])
                

        #this one is for hip back side
        right_hip = (self.right_hip)
        if right_hip:
            # right_hip_bend = (self.right_hip and 161 <= self.right_hip <= 180)
            # if right_hip_bend:
            #     self.all_methods.play_voice(["just bend down your hip"])
                # self.all_methods.voice("your hip not bend pefect move your hip to front and bend back side")

            #this one is for hip front side
            right_hip_up = (self.right_hip and 0 <= self.right_hip <= 129)
            if right_hip_up:
                self.all_methods.play_voice(["lift your hip up"])
                # self.all_methods.voice("your hip is bend in bad position so you are in bad position so bend your body back side")


        #left elbow
        right_elbow_correct = (self.right_elbow)
        if right_elbow_correct:
        #right elbow
            right_elbow_open = (self.right_elbow and 0 <= self.right_elbow <= 60)
            if right_elbow_open:
                self.all_methods.play_voice(["open your hands and touch your palm to your hip"])
                
            right_elbow_close = (self.right_elbow and 101 <= self.right_elbow <= 180)
            if right_elbow_close:
                self.all_methods.play_voice(["close your hands and touch your palm to your hip"])


        #this one is for right shoulder
        right_shoulder_correct = (self.right_shoulder)
        if right_shoulder_correct:
            right_shoulder_open = (self.right_shoulder and 0<= self.right_shoulder <= 60)
            if right_shoulder_open:
                self.all_methods.play_voice(["open your shoulders"])

            right_shoulder_close = (self.right_shoulder and 101 <= self.right_shoulder <= 180)
            if right_shoulder_close:
                self.all_methods.play_voice(["close your shoulders"])
            
        
        #this one is for head
        head_correct = (self.head_y and self.right_heel_y)
        if head_correct:
            right = (self.head_y < self.right_heel_y)
            if right:
                self.all_methods.play_voice(["legs is in upside and head is touch to ground"])

        else:
            return

    def right_sarvangasana_name(self,frames):  

        if not self.head_x or not self.right_elbow or not self.right_hip or not  self.right_knee or not self.right_shoulder:
            return
       
        if (
            self.head_y > self.right_heel_y and 
            self.right_elbow  and 61 <= self.right_elbow <= 100 and 
            self.right_hip and 130 <= self.right_hip <= 160 and  
            self.right_knee and 160 <= self.right_knee <= 180 and
            self.left_knee1 and 160 <= self.left_knee1 <= 180 and
            self.right_shoulder and 61 <= self.right_shoulder <= 100):# and
            
                cv.putText(frames,str("sarvangasana"),(80,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                self.all_methods.play_voice(["you done your yoga pose perfect"])
                
                return True
        
        return False      

    def left_sarvangasana_name(self,frames):

        if not self.head_x or not self.left_elbow or not self.left_hip or not  self.left_knee or not self.left_shoulder:
            return

        if (
            self.head_y > self.left_heel_y and
            self.left_elbow and 61 <= self.left_elbow <= 100 and
            self.left_hip and 130 <= self.left_hip <= 160 and
            self.left_knee and 160 <= self.left_knee <= 180 and 
            self.right_knee1 and 160 <= self.right_knee1 <= 180 and
            self.left_shoulder and 61 <= self.left_shoulder <= 100):# and
            # self.left_shoulder_ear and 65 <= self.left_shoulder_ear <= 130):

                cv.putText(frames,str("sarvangasana"),(80,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                self.all_methods.play_voice(["you done your yoga pose perfect"])
                
                return True
        
        return False  
    



    # def resetOneTimeRunFlag(self, ignoringFlag):
    #     attr_map = {
    #         "ALWAYS_SLEEPING": "oneTimeRun_flag_for_always_sleeping"
            
    #     }

    #     all_attrs = attr_map.values()
    #     ignore_attr = attr_map.get(ignoringFlag)

    #     for attr in all_attrs:
    #         if attr != ignore_attr:
    #             setattr(self, attr, 0)



    # def reset_voiceHoldingTime_AND_oneTimeRunFlag(self, oneTimeRunFlag, ignoringFlag):
    #     if getattr(self, oneTimeRunFlag) < 1:
    #         setattr(self, oneTimeRunFlag, 1)
    #         self.timeTaking_for_same_current_position=0
    #         self.resetOneTimeRunFlag(ignoringFlag=ignoringFlag)
        
    #     if self.voice.isVoicePlaying:
    #         self.timeTaking_for_same_current_position = 0
    #     else:
    #         self.timeTaking_for_same_current_position+=1
   
def main():
    global llist
    all_methods = allmethods()
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980)
    detect = sarvangasana()
    voice = VoicePlay()
    all_methods = allmethods()
    intial_position = False
    ready_for_exercise = False
    sleep_voice_given = False  

    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        cv.putText(frames,  f'TIME {voice.isVoicePlaying}', (30, 60), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # detect.voice_cooldown_counter

        cv.putText(frames,  f'TIME {detect.voice_cooldown_counter}', (30, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        img = cv.imread("images/download.jpeg")
        img1 = cv.resize(img, None, fx=4.0, fy=4.0, interpolation=cv.INTER_LINEAR)
        detect.pose_positions(frames,draw = False)
        llist = detect.pose_landmarks(frames,draw=False)
        detect.head_heel_points(frames=frames,llist=llist,face_points=(0,2,5),left_heel_point=29,right_heel_point=30)
        side_view = all_methods.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)
        sleeping_position = all_methods.is_person_standing_sitting(frames=frames,llist=llist,hip_points=(11,23,25),leg_points=(23,25,27),elbow_points=(11,13,15),height=height,width=width)


        if not intial_position and not ready_for_exercise and sleeping_position == "sleeping":   
            
            all_methods.play_voice(["you are in sleep position and do sarvangasana", "lift your legs up"])
            
            ready_for_exercise = True
            intial_position = True
        
        if ready_for_exercise and sleeping_position == "reverse":

            if sleeping_position == "reverse":

                all_methods.play_voice(["you are in sleep position and do sarvangasana", "lift your legs up"])
            
                if side_view ==  "right":

                    detect.right_sarvangasana(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),right_shoulder_ear=(23,11,7),draw=True)
                    wrong_right = detect.wrong_right(frames=frames)
                    correct = detect.right_sarvangasana_name(frames)

                elif side_view == "left":

                    detect.left_sarvangasana(frames=frames,llist=llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),right_knee= (24,26,28),left_shoulder_ear=(24,12,8),draw=True)
                    wrong_left = detect.wrong_left(frames=frames)
                    correct = detect.left_sarvangasana_name(frames)

            elif sleeping_position != "reverse":
                sleep_voice_given = False
                ready_for_exercise = False
                intial_position = False

        elif sleeping_position != "sleeping" and sleeping_position != "reverse":
            
            all_methods.play_voice(["you must be in sleep position"])
            sleep_voice_given = False
            ready_for_exercise = False
            intial_position = False

        cv.imshow("video",frames)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()

main()



