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
from side import side_potion
from body_position import bodyPosition

class pranamasana:
     
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
        self.all_methods = allmethods()
        self.calculate_angle = self.all_methods.calculate_angle
        self.module_change = True
        self.head_point = self.all_methods.head_detect
        self.left_heel_detect = self.all_methods.left_heel_detect
        self.right_heel_detect = self.all_methods.right_heel_detect
        self.voice = VoicePlay()
        self.side = side_potion()
        self.body_position = bodyPosition()
        self.round = True

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        self.head_pose = HeadPoseEstimator()    

        # self.corrections = "legs_correct"    

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

    def left_pranamasana(self, frames,llist, elbow, hip, knee,shoulder,draw =True):

        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        self.left_elbow, elbow_coords = self.all_methods.calculate_angle(frames=frames, points=elbow,lmList=llist)
        self.left_hip, hip_coords = self.all_methods.calculate_angle(frames=frames,points= hip,lmList=llist)
        self.left_knee, knee_coords = self.all_methods.calculate_angle(frames=frames,points= knee,lmList=llist)       
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        
        if draw:
            if elbow_coords:
                cv.putText(frames, str(int(self.left_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.left_hip)), (hip_coords[2]+10, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.left_knee)), (knee_coords[2]+10, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if points_cor8:
                cv.putText(frames, str(int(self.left_shoulder)), (points_cor8[2]+10, points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.head_position

    def right_pranamasana(self,frames,llist,elbow, hip, knee, shoulder,draw=True):

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

        if draw:
            if elbow_coords:
                cv.putText(frames, str(int(self.right_elbow)), (elbow_coords[2]+10, elbow_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if hip_coords:
                cv.putText(frames, str(int(self.right_hip)), (hip_coords[2]-20, hip_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if knee_coords:
                cv.putText(frames, str(int(self.right_knee)), (knee_coords[2]-20, knee_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if points_cor7:
                cv.putText(frames, str(int(self.right_shoulder)), (points_cor7[2]+10, points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder
    

    def head_heel_points(self,frames,llist,face_points,left_heel_point,right_heel_point,draw = True):

        self.head_points = self.head_point(frames=frames,lmlist=llist,points=face_points) 

        self.left_heel_points = self.left_heel_detect(frames=frames,lmlist=llist,point=left_heel_point)

        self.right_heel_points = self.right_heel_detect(frames=frames,lmlist=llist,point=right_heel_point)

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
        
        if not self.head_points or not self.left_heel_points or not self.right_heel_points:

            return None

        
    def check_stand(self,frames,llist,height,width):

        
        self.view_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,hip_points=(11,23,25),leg_points=(23,25,27),elbow_points=(11,13,15),height=height,width=width)

        # return self.view_position

        if self.view_position == "standing":

            # self.voice.playAudio(["you can start your yoga","stand in a left side or right side position and do pranamasana"],play=True)
            return True
        
        elif self.view_position != "standing":
            self.voice.playAudio(["you must be in standing position"],play=True)

    # def start_exercise(self,frames):

    #     self.start_ = (
    #         self.head_position and self.head_position != "Right" or
    #                   self.right_knee and 0 <= self.right_knee <= 180 and
    #                   self.right_elbow and 0 <= self.right_elbow <= 180 and
    #                   self.right_shoulder and 0 <= self.right_shoulder <= 10)
        
    #     if self.start_:

    #         if self.start_:
    #             self.voice.playAudio(["you can start your yoga","stand in a left side or right side position and do pranamasana"],play=True)

    #     else:
    #         return False

    
        
    def start_left_exercise(self,frames):

        self.start = (
                    self.head_position and self.head_position != "Left" or
                    self.left_knee and 0 <= self.left_knee <= 180 and
                    self.left_elbow and 0 <= self.left_elbow <= 180 and
                    self.left_shoulder and 0 <= self.left_shoulder <= 10)
        
        self.hasta_left = (
            self.head_x < self.left_heel_x and
            self.left_elbow and 120 <= self.left_elbow <= 180 and
            self.left_hip and 100 <= self.left_hip <= 160 and
            self.left_knee and 100 <= self.left_knee <= 180 and 
            self.left_shoulder and 120 <= self.left_shoulder <= 170)
        
        if self.round:

            if self.start:
                self.voice.playAudio(["you can start your yoga","stand in a left side or right side position and do pranamasana"],play=True)

                return True
            
            else:
                self.round = False
                return False
        
        elif not self.round:
            if self.hasta_left:
                self.voice.playAudio(["move to next pose stand in a left side position and do pranamasana"],play=True)
                return True
            else:
                # self.round = True
                return False
    
    def start_right_exercise(self,frames):

        self.start = (
            self.head_position and self.head_position != "Right" or
                      self.right_knee and 0 <= self.right_knee <= 180 and
                      self.right_elbow and 0 <= self.right_elbow <= 180 and
                      self.right_shoulder and 0 <= self.right_shoulder <= 10)
        
        self.hasta_right=(
            self.head_x > self.right_heel_x and 
            self.right_elbow  and 100 <= self.right_elbow <= 180 and 
            self.right_hip and 100 <= self.right_hip <= 180 and  
            self.right_knee and 100 <= self.right_knee <= 180 and
            self.right_shoulder and 100 <= self.right_shoulder <= 180)
            
        if self.round:
            if self.start:
                self.voice.playAudio(["you can start your yoga","stand in a right or left side position and do pranamasana"],play=True)

                return True
            else:
                self.round = False
                return False
            
        elif not self.round:
            if self.hasta_right:
                self.voice.playAudio(["move to next pose stand in a right side position and do pranamasana"],play=True)
                return True
            else:
                self.round = True
                return False
    
    def wrong_pranamasana_left(self,frames): 

            head_correct = (self.head)
            if head_correct:
                up_correct =  (self.head_position and self.head_position != "Left")
                correct_up = False
                if up_correct:
                    self.voice.playAudio(["you should to face your head to left side"],callBack=None,play=True)
                    # self.all_methods.playAudio("you must be face camera  please keep your face in forward position")

            knee_correct = (self.left_knee )
            if knee_correct:
                left_knee_correct = (self.left_knee and 0 <= self.left_knee <= 160)
                if left_knee_correct:
                    self.voice.playAudio(["please keep your left leg straight"],callBack=None,play=True)
                    # self.all_methods.playAudio("please keep your left leg straight and keep your legs together")
                

            #this one for both sides of hips
            hip_correct = (self.left_hip)
            if hip_correct:
                left_hip_correct = (self.left_hip and 0 <= self.left_hip <= 159)
                if left_hip_correct:
                    self.voice.playAudio(["please keep your left hip straight "],callBack=None,play=True)
                    # self.all_methods.playAudio("please keep your left hip straight ")

            #this one is for shoulder
            shoulder_correct =  self.left_shoulder
            if shoulder_correct:
                left_shoulder_correct_up = (self.left_shoulder and 16 <= self.left_shoulder <= 180)
                if left_shoulder_correct_up:
                    self.voice.playAudio(["please fold your shoulders"],callBack=None,play = True)
                    # self.all_methods.playAudio("your left shoulder is in up position please keep your shoulder slight down your shoulders")
                

            #this one is for elbow
            elbow_correct = (self.left_elbow)
            if elbow_correct:
                left_elbow_correct_down = (self.left_elbow and 0 <= self.left_elbow <= 49)
                if left_elbow_correct_down:
                    self.voice.playAudio(["please keep your elbow slightly open your elbows"],callBack=None,play=True)
                    # self.voice.playAudio("your left elbows is in down position please keep your elbow slightly up your elbows")

                left_elbow_correct_up = (self.left_elbow and 71 <= self.left_elbow <= 180)
                if left_elbow_correct_up:
                    self.voice.playAudio(["please keep your elbow slight close your elbows"],callBack=None,play=True)
                    # self.voice.playAudio("your left elbow is in up position please keep your elbow slight down your elbows")

                
            else:
                return  

    def wront_right_pranamasana(self,frames):

        head_correct = (self.head)
        if head_correct:
            up_correct =  (self.head_position and self.head_position != "Right")
            if up_correct:
                self.voice.playAudio(["you should to face your head to right side"],callBack=None,play=True)
                # self.all_methods.playAudio("you must be face camera  please keep your face in forward position")


        knee_correct = ( self.right_knee)   
        if knee_correct:
            right_knee_correct = (self.right_knee and 0 <= self.right_knee <= 160)
            if right_knee_correct:
                self.voice.playAudio(["please keep your right leg straight"],callBack=None,play=True)
                # self.all_methods.playAudio("please keep your right leg straight and keep your legs together")

        hip_correct = ( self.right_hip)
        if hip_correct:
            right_hip_correct = (self.right_hip and 0 <= self.right_hip <= 159)
            if right_hip_correct:
                self.voice.playAudio(["please keep your right hip straight "],callBack=None,play=True)
                # self.all_methods.playAudio("please keep your right hip straight ")

        elbow_correct = (self.right_elbow)
        if elbow_correct:
                right_elbow_correct_down = (self.right_elbow and 0 <= self.right_elbow <= 49)
                if right_elbow_correct_down:
                    self.voice.playAudio([" please keep your elbow slightly open your elbows"],callBack=None,play=True)
                    # self.voice.playAudio("your right elbows is in down position please keep your elbow slightly up your elbows")
                
                right_elbow_correct_up = (self.right_elbow and 71 <= self.right_elbow <= 180)
                if right_elbow_correct_up:
                    self.voice.playAudio(["please keep your elbow slight close your elbows"],callBack=None,play=True)
                    # self.voice.playAudio("your right elbow is in up position please keep your elbow slight down your elbows")


        shoulder_correct = (self.right_shoulder)
        if shoulder_correct:
            right_shoulder_correct_up = (self.right_shoulder and 16 <= self.right_shoulder <= 180)
            if right_shoulder_correct_up:
                self.voice.playAudio(["fold your shoulders "],callBack=None,play=True)
                # self.all_methods.playAudio("your right shoulder is in up position please keep your shoulder slight down your shoulders")
      

    def left_pranamasana_name(self,frames): 


        correct =(
            self.left_shoulder and 0 <= self.left_shoulder <= 15 and
            self.left_elbow and 50 <= self.left_elbow <= 70 and
            self.left_hip and 160 <= self.left_hip <= 180 and 
            self.left_knee and 170 <= self.left_knee <= 180 and
            self.head_position and self.head_position == "Left")
        
        if correct:
                cv.putText(frames,str("Pranamasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                self.voice.playAudio(["you done your yoga pose perfect"],play=True)
                return True
        
        return False
    
    def right_pranamasana_name(self,frames):

        # self.voice_ = None
        correct = (self.right_elbow and 50 <= self.right_elbow <= 70 and           
            self.right_hip and 160 <= self.right_hip <= 180 and            
            self.right_knee and 170 <= self.right_knee <= 180 and                                      
            self.right_shoulder and 0 <= self.right_shoulder <= 15 and
            self.head_position and self.head_position == "Right")
        if correct:
                cv.putText(frames,str("Pranamasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                self.voice.playAudio(["you done your yoga pose perfect"],play=True)
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
    detect = pranamasana()
    all_methods = allmethods()
    voice = VoicePlay()
    pose_detected = None
    voice_detect = False
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
        detect.head_heel_points(frames=frames,llist=llist,face_points=(0,2,5),left_heel_point=29,right_heel_point=30,draw=False)
        side_view = all_methods.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)

        if not llist or llist == 0:
            pass

        elif len(llist) != 0:

            #Left side
            if side_view == "left":
                detect.left_pranamasana(frames=frames,llist=llist,elbow=(11,13,15), hip=(11,23,25), knee=(23,25,27), shoulder=(13,11,23),draw=False)
                head = detect.head_distance(frames,(0,2,5))
                stand_position = detect.check_stand(frames=frames,llist=llist,height=detect.h,width=detect.w)
                if stand_position:
                    # start_left = detect.start_left_exercise(frames=frames)
                    # if not start_left:
                    wrong_left = detect.wrong_pranamasana_left(frames=frames)
                pranam_name = detect.left_pranamasana_name(frames=frames)

            #Right Side
            elif side_view == "right":
                pranamasana_angles = detect.right_pranamasana(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),draw=False)
                head = detect.head_distance(frames,(0,2,5))
                stand_position = detect.check_stand(frames=frames,llist=llist,height=detect.h,width=detect.w)
                if stand_position:
                    # start_right = detect.start_right_exercise(frames=frames)
                    # if not start_right:
                    wrong_right = detect.wront_right_pranamasana(frames=frames)
                pranam_name = detect.right_pranamasana_name(frames=frames)

            elif side_view == "left side cross position":
                voice.playAudio(["turn total left side"],play=True)

            elif side_view == "right side cross position":
                voice.playAudio(["turn total right side"],play=True)

            else:
                voice.playAudio(["turn left side or right side and do pranamasana"],play=True)

        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()
