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

class plankpose:
     
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
        self.lmslist = []
        self.angle =0
        self.all_methods = allmethods()
        self.calculate_angle = self.all_methods.calculate_angle
        self.module_change = True
        self.head_point = self.all_methods.head_detect
        self.voice = VoicePlay()

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        self.head_pose = HeadPoseEstimator()

        # self.corrections = "legs_correct"   
        #BODY BACK CONDITIONS
        self.MAX_BACK_LIMT = 40
        self.MIN_BACK_LIMT = 15
        
        #THIGH CONDITIONS
        self.MAX_THIGH_LIMT = 50
        self.MIN_THIGH_LIMT = 20
        
        #LOWER LEG
        self.MAX_LOWER_LEG_LIMIT = 35
        self.MIN_LOWER_LEG_LIMIT = 10
        
        #FULL_HAND CONDITIONS
        self.MAX_FULL_HAND_LIMT = 95
        self.MIN_FULL_HAND_LIMT = 85

    def pose_positions(self,frames,draw = True):

        imgRB = cv.cvtColor(frames,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frames,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

    def pose_landmarks(self,frames,draw = True):

        self.lpslist =[]

        if self.results.pose_landmarks:

            h,w,cl = frames.shape

            for id,poselms in enumerate(self.results.pose_landmarks.landmark):
                px,py ,pz= int(poselms.x * w) , int(poselms.y * h),(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lpslist
    
    def slope_landmarks(self,frames,draw = True):

        self.lmslist =[]

        if self.results.pose_landmarks:

            self.h,self.w,cl = frames.shape

            for id,poselms in enumerate(self.results.pose_landmarks.landmark):
                px,py ,pz= (poselms.x * self.w) , (poselms.y * self.h),(poselms.z)

                self.lmslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lmslist
    
 
    def left_plank(self, frames,llist, elbow, hip, knee,shoulder,right_knee,draw =True):

        self.ground_left = self.all_methods.ground_distance_left(frames=frames,lmlist=llist)
        # cv.putText(frames,str(self.ground_left),(10,520),cv.FONT_HERSHEY_DUPLEX,2,(255,0,255),2)
        self.min_ground_left = self.ground_left-30

        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        self.left_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
        self.left_hip, hip_coords = self.all_methods.calculate_angle(frames=frames,points= hip,lmList=llist)
        self.left_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)       
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.right_knee1,points_cor10 = self.all_methods.calculate_angle(frames=frames,points=right_knee, lmList=llist)
        
        if draw:
            if elbow_coords:
                cv.putText(frames, str(int(self.left_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.left_hip)), (hip_coords[2]+10, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.left_knee)), (knee_coords[2]+10, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if points_cor8:
                cv.putText(frames, str(int(self.left_shoulder)), (points_cor8[2]+10, points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            # if points_cor10:
            #     cv.putText(frames, str(int(self.right_knee1)), (points_cor10[2]+10, points_cor10[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.head_position,self.right_knee1

    def right_plank(self,frames,llist,elbow, hip, knee, shoulder,left_knee,draw=True):

        self.ground_right = self.all_methods.ground_distance_right(frames=frames,lmlist=llist)
        # cv.putText(frames,str(self.ground_right),(10,520),cv.FONT_HERSHEY_DUPLEX,2,(255,0,255),2)
        self.min_ground_right = self.ground_right-30

        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        self.right_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
        self.right_hip, hip_coords = self.all_methods.calculate_angle(frames=frames, points=hip,lmList=llist)
        self.right_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)
        self.right_shoulder,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.left_knee1,points_cor9 = self.all_methods.calculate_angle(frames=frames,points=left_knee, lmList=llist)

        if draw:
            if elbow_coords:
                cv.putText(frames, str(int(self.right_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.right_hip)), (hip_coords[2]-20, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.right_knee)), (knee_coords[2]-20, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if points_cor7:
                cv.putText(frames, str(int(self.right_shoulder)), (points_cor7[2]+10, points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            # if points_cor9:
            #     cv.putText(frames, str(int(self.left_knee1)), (points_cor9[2]+10, points_cor9[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)


        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,self.left_knee1
    
    def hands_legs_correct_position(self,frames,llist,points):

        if len(llist) != 0:

            self.right_finger_x,self.right_finger_y,self.right_finger_z = llist[points[0]][1:]
            # print("right_finger",self.right_finger_y)
            self.left_finger_x,self.left_finger_y,self.left_finger_z= llist[points[1]][1:]
            # cv.putText(frames,str(self.left_finger_y),(180,250),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
            self.right_foot_x,self.right_foot_y,self.right_foot_z = llist[points[2]][1:]
            self.left_foot_x , self.left_foot_y,self.left_foot_z = llist[points[3]][1:]
            
            return self.right_finger_x,self.right_finger_y,self.left_finger_x,self.left_finger_y,self.left_foot_x,self.left_foot_y,self.right_foot_x,self.right_foot_y,self.right_finger_z,self.left_finger_z,self.right_foot_z,self.left_foot_z
        
        return False

    def left_slope_condition(self,frames,llist,height,width):

        self.left_full_hand = self.all_methods.slope(frames=frames,lmlist=llist,point1=15,point2=11,height=height,width=width,draw=False)
        self.left_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=11,height=height,width=width,draw=False)
        self.left_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=25,point2=23,height=height,width=width,draw=False)
        self.left_knee_ankle = self.all_methods.slope(frames=frames,lmlist=llist,point1=27,point2=25,height=height,width=width,draw=False)

        # cv.putText(frames,f'full_hand=>{str(self.left_full_hand)}',(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        # cv.putText(frames,f'shoulder_hip=>{str(self.left_shoulder_hip)}',(40,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        # cv.putText(frames,f'hip_knee=>{str(self.left_hip_knee)}',(40,130),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        # cv.putText(frames,f'knee_ankle=>{str(self.left_knee_ankle)}',(40,180),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

        return self.left_full_hand,self.left_shoulder_hip,self.left_hip_knee,self.left_knee_ankle
    
    def right_slope_condition(self,frames,llist,height,width):

        self.right_full_hand = self.all_methods.slope(frames=frames,lmlist=llist,point1=16,point2=12,height=height,width=width,draw=False)
        self.right_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=False)
        self.right_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=False)
        self.right_knee_ankle = self.all_methods.slope(frames=frames,lmlist=llist,point1=28,point2=26,height=height,width=width,draw=False)

        # cv.putText(frames,f'full_hand=>{str(self.right_full_hand)}',(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        # cv.putText(frames,f'shoulder_hip=>{str(self.right_shoulder_hip)}',(40,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        # cv.putText(frames,f'hip_knee=>{str(self.right_hip_knee)}',(40,130),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

        return self.right_full_hand,self.right_shoulder_hip,self.right_hip_knee,self.right_knee_ankle

    def start_left_exercise(self,frames):

        self.ashwa_left = (
            self.min_ground_left-40 <= self.left_finger_y <= self.ground_left and
            ((self.left_finger_z < self.right_foot_z) or (self.left_finger_z < self.right_foot_z < self.right_finger_z)) and
            self.left_elbow and 140 <= self.left_elbow <= 180 and
            self.left_hip and 130 <= self.left_hip <= 180 and
            self.left_knee and 120 <= self.left_knee <= 180 and
            self.left_shoulder and 50 <= self.left_shoulder <= 105  and 
            self.head_position and self.head_position == "Left" #and
            # self.right_knee1 and 50 <= self.right_knee1 <= 100
            # self.right_leg and 45 <= self.right_leg <= 60
            )
        
        if self.ashwa_left:
            self.voice.playAudio(["move to next pose  plank","extend right leg back"],play=True)
            return True
        else:
            return False
    
    def start_right_exercise(self,frames):

        self.ashwa_right = (
            ((self.right_finger_z < self.left_foot_z) or (self.right_finger_z < self.left_foot_z < self.left_finger_z)) and
            self.min_ground_right-40 <= self.right_finger_y <= self.ground_right and
            self.right_elbow and 140 <= self.right_elbow <= 180 and  
            self.right_hip and 130 <= self.right_hip <= 180 and
            self.right_knee and  120 <= self.right_knee <= 180 and
            self.right_shoulder and 50 <= self.right_shoulder <= 105 and
            self.head_position and self.head_position == "Right" #and
            # self.left_knee1 and 50 <= self.left_knee1 <= 100
            # self.left_leg and 45 <= self.left_leg <= 60
            )
        
        if self.ashwa_right:
            self.voice.playAudio(["move to next pose plank ","extend right leg back"],play=True)
            return True
        else:
            return False


    def wrong_left_plank(self,frames):
            
            right_knee1 = (self.right_knee1)
            if right_knee1:
                right_knee1_correct = (self.right_knee1 and 0 <= self.right_knee1 <= 159)
                if right_knee1_correct:
                    self.voice.playAudio(["please keep your right leg back straight"],callBack=None,play=True)
            else:
                return 
            

            knee_correct = (self.left_knee )
            if knee_correct:
                left_knee_correct = (self.left_knee and 0 <= self.left_knee <= 159)
                if left_knee_correct:
                    self.voice.playAudio(["please keep your left leg back straight "],callBack=None,play=True)
                    # self.all_methods.playAudio("please keep your left leg straight and keep your legs together")
                      

            #this one for both sides of hips
            hip_correct = (self.left_hip)
            if hip_correct:
                left_hip_correct = (self.left_hip and 0 <= self.left_hip <= 149)
                if left_hip_correct:
                    self.voice.playAudio(["please keep your left hip straight "],callBack=None,play=True)
                    # self.all_methods.playAudio("please keep your left hip straight ")

            #this one is for shoulder
            shoulder_correct =  self.left_shoulder
            if shoulder_correct:
                left_shoulder_correct_up = (self.left_shoulder and 0 <= self.left_shoulder <= 59)
                if left_shoulder_correct_up:
                    self.voice.playAudio(["please open your shoulders"],callBack=None,play = True)
                    # self.all_methods.playAudio("your left shoulder is in up position please keep your shoulder slight down your shoulders")

                
                left_shoulder_correct_down = (self.left_shoulder and 76 <= self.left_shoulder <= 180)
                if left_shoulder_correct_down:
                    self.voice.playAudio(["please close your shoulders"],callBack=None,play = True)
                    # self.all_methods.playAudio("your left shoulder is in up position please keep your shoulder slight down your shoulders")
                

            #this one is for elbow
            elbow_correct = (self.left_elbow)
            if elbow_correct:
                left_elbow_correct_down = (self.left_elbow and 0 <= self.left_elbow <= 159)
                if left_elbow_correct_down:
                    self.voice.playAudio(["please keep your elbow in straight"])


            head_correct = (self.head)
            if head_correct:
                up_correct =  (self.head_position and self.head_position != "Down")
                correct_up = False
                if up_correct:
                    self.voice.playAudio(["you should to face your head to left down"],callBack=None,play=True)
                    # self.all_methods.playAudio("you must be face camera  please keep your face in forward position")
                
            else:
                return  

    def wront_right_plank(self,frames):

        knee_correct = ( self.right_knee)   
        if knee_correct:
            right_knee_correct = (self.right_knee and 0 <= self.right_knee <= 159)
            if right_knee_correct:
    
                self.voice.playAudio(["please keep your legs straight"],callBack=None,play=True)
                # self.all_methods.playAudio("please keep your right leg straight and keep your legs together")

        left_knee1 = (self.left_knee1)
        if left_knee1:
            left_knee1_correct = (self.left_knee1 and 0 <= self.left_knee1 <= 159)
            if left_knee1_correct:
                self.voice.playAudio(["please keep your left leg straight"],callBack=None,play=True)
        else:
            return
        

        hip_correct = ( self.right_hip)
        if hip_correct:
            right_hip_correct = (self.right_hip and 0 <= self.right_hip <= 149)
            if right_hip_correct:
                self.voice.playAudio(["please keep your right hip straight "],callBack=None,play=True)
                # self.all_methods.playAudio("please keep your right hip straight ")


        elbow_correct = (self.right_elbow)
        if elbow_correct:
                right_elbow_correct_down = (self.right_elbow and 0 <= self.right_elbow <= 159)
                if right_elbow_correct_down:
                    self.voice.playAudio("please keep your elbow staright")


        shoulder_correct = (self.right_shoulder)    
        if shoulder_correct:
            right_shoulder_correct_up = (self.right_shoulder and 0 <= self.right_shoulder <= 59)
            if right_shoulder_correct_up:
                self.voice.playAudio(["please open your shoulders"],callBack=None,play = True)

            right_shoulder_correct_down = (self.right_shoulder and 76 <= self.right_shoulder <= 180)
            if right_shoulder_correct_down:
                self.voice.playAudio(["please close your shoulders"],callBack=None,play = True)

        head_correct = (self.head)
        if head_correct:
            up_correct =  (self.head_position and self.head_position != "Down")
            if up_correct:
                self.voice.playAudio(["you should to face your head to down "],callBack=None,play=True)
                # self.all_methods.playAudio("you must be face camera  please keep your face in forward position")

        else:
            return 


    def left_plank_name(self,frames): 

        if not self.left_knee or not self.left_hip or not self.left_elbow or not self.left_shoulder:
            return True
        
        correct =(
            self.MIN_BACK_LIMT <= self.left_shoulder_hip <= self.MAX_BACK_LIMT and
            self.MIN_FULL_HAND_LIMT <= self.left_full_hand <= self.MAX_FULL_HAND_LIMT and
            self.MIN_THIGH_LIMT <= self.left_hip_knee <= self.MAX_THIGH_LIMT and
            self.MIN_LOWER_LEG_LIMIT <= self.left_knee_ankle <= self.MAX_LOWER_LEG_LIMIT and
            self.left_shoulder and 60 <= self.left_shoulder <= 75 and
            self.left_elbow and 160 <= self.left_elbow <= 180 and
            self.left_hip and 150 <= self.left_hip <= 180 and 
            self.left_knee and 160 <= self.left_knee <= 180 and
            self.right_knee1 and 160 <= self.right_knee1 <= 180 and
            self.head_position and self.head_position == "Down")
        
        if correct:
                cv.putText(frames,str("Plank Pose"),(10,20),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                return True
        
        return False
    
    def right_plank_name(self,frames):

        if not self.right_knee or not self.right_hip or not self.right_elbow or not self.right_shoulder:
            return True

        # self.voice_ = None
        correct = (
            self.MIN_BACK_LIMT <= self.right_shoulder_hip <= self.MAX_BACK_LIMT and
            self.MIN_FULL_HAND_LIMT <= self.right_full_hand <= self.MAX_FULL_HAND_LIMT and
            self.MIN_THIGH_LIMT <= self.right_hip_knee <= self.MAX_THIGH_LIMT and
            self.MIN_LOWER_LEG_LIMIT <= self.right_knee_ankle <= self.MAX_LOWER_LEG_LIMIT and
            self.right_elbow and 160 <= self.right_elbow <= 180 and           
            self.right_hip and 150 <= self.right_hip <= 180 and            
            self.right_knee and 160 <= self.right_knee <= 180 and  
            self.left_knee1 and 160 <= self.left_knee1 <= 180 and                                  
            self.right_shoulder and 60 <= self.right_shoulder <= 75 and
            self.head_position and self.head_position == "Down")
        if correct:
                cv.putText(frames,str("Plank Pose"),(10,20),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                return True
        
        return False
    
    
    def head_distance(self,frames,point):

        if len(llist) != 0:

            self.head = self.head_point(frames=frames,lmlist=llist,points=point)

            self.head_x = self.head[0]
            self.head_y = self.head[1]

        return self.head_x
    

def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    global  llist
    global slope_llist
    detect = plankpose()
    all_methods = allmethods()
    voice= VoicePlay()
    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        img = cv.imread("images/image1.webp")
        resized_img = cv.resize(img, None, fx=3.0, fy=3.0, interpolation=cv.INTER_LINEAR)
        detect.pose_positions(frames,draw=False)
        llist = detect.pose_landmarks(frames,False)
        slope_llist = detect.slope_landmarks(frames=frames,draw=False)
        detect.hands_legs_correct_position(frames=frames,llist=llist,points=(16,15,32,31))
        detect.head_distance(frames=frames,point=(0,2,5))
        side_view = all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=detect.HEAD_POSITION,head=detect.head_x)
        standing_side_detect = all_methods.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)

        if len(llist) != 0:

            if standing_side_detect == "forward":
                voice.playAudio(["if you not understood taking by reference video"],play=True)

                if side_view == "left":
                    
                    plank_angles = detect.left_plank(frames=frames,llist=llist,elbow=(11,13,15), hip=(11,23,25), knee=(23,25,27), shoulder=(13,11,23),right_knee=(24,26,28),draw=True)
                    detect.left_slope_condition(frames=frames,llist=slope_llist,height=detect.h,width=detect.w)
                    wrong_left = detect.wrong_left_plank(frames=frames)
                    detect.left_plank_name(frames=frames)

                elif side_view == "right":
                    
                    plank_angles = detect.right_plank(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26), knee=(24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),draw=True)
                    detect.right_slope_condition(frames=frames,llist=slope_llist,height=detect.h,width=detect.w)
                    wrong_right = detect.wront_right_plank(frames=frames)
                    detect.right_plank_name(frames=frames)
              
        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()
