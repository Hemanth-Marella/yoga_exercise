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
# from hastauttanasana import hastauttanasana

class bhujangasana:
     
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
        self.voice_thread = None
        self.voice_detect = False
        self.all_methods = allmethods()
        self.calculate_angle = self.all_methods.calculate_angle
        self.palm_distance = 0
        self.head_point = self.all_methods.head_detect
        self.voice = VoicePlay()

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        self.head_pose = HeadPoseEstimator()  

        #BODY BACK CONDITIONS
        self.MAX_BACK_LIMT = 75
        self.MIN_BACK_LIMT = 65
        
        #THIGH CONDITIONS
        self.MAX_THIGH_LIMT = 30
        self.MIN_THIGH_LIMT = 15

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
                px,py,pz = int(poselms.x * w) , int(poselms.y * h),(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py,pz),5,(255,255,0),2)

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

    def bhujangasana_right(self,frames,llist,elbow,hip,knee,shoulder,draw = True):

        # face detect 
        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        # print(self.head_position)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 

        # create angles 
        self.right_elbow,points_cor1 = self.all_methods.calculate_angle(frames=frames,points= elbow, lmList=llist)
        self.right_hip ,points_cor2= self.all_methods.calculate_angle(frames=frames,points=hip, lmList=llist)
        self.right_knee,points_cor3 = self.all_methods.calculate_angle(frames=frames,points=knee, lmList=llist)
        self.right_shoulder,points_cor4 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        
        if draw:

            if points_cor1:
                cv.putText(frames, str(int(self.right_elbow)), (points_cor1[2]+10, points_cor1[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor2:
                cv.putText(frames, str(int(self.right_hip)), (points_cor2[2]-20, points_cor2[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor3:
                cv.putText(frames, str(int(self.right_knee)), (points_cor3[2]-20, points_cor3[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor4:
                cv.putText(frames, str(int(self.right_shoulder)), (points_cor4[2]+10, points_cor4[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            
        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder
    
    def bhujangasana_left(self,frames,llist,elbow,hip,knee,shoulder,draw = True):
        
        # face detect 
        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 

        # create angles 
        self.left_elbow,self.points_cor5 = self.all_methods.calculate_angle(frames=frames,points=elbow, lmList=llist)
        self.left_hip,self.points_cor6 = self.all_methods.calculate_angle(frames=frames,points=hip, lmList=llist)
        self.left_knee,self.points_cor7 = self.all_methods.calculate_angle(frames=frames,points=knee, lmList=llist)
        self.left_shoulder,self.points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)

        if draw:
            if self.points_cor5:
                cv.putText(frames, str(int(self.left_elbow)), (self.points_cor5[2]+10, self.points_cor5[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if self.points_cor6:
                cv.putText(frames, str(int(self.left_hip)), (self.points_cor6[2]+10, self.points_cor6[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if self.points_cor7:
                cv.putText(frames, str(int(self.left_knee)), (self.points_cor7[2]+10, self.points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if self.points_cor8:
                cv.putText(frames, str(int(self.left_shoulder)), (self.points_cor8[2]+10, self.points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder
    
    def left_slope_condition(self,frames,llist,height,width):

        # self.left_full_hand = self.all_methods.slope(frames=frames,lmlist=llist,point1=15,point2=11,height=height,width=width)
        self.left_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width,draw=True)
        self.left_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=True)

        # cv.putText(frames,str(self.left_full_hand),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        cv.putText(frames,str(self.left_shoulder_hip),(40,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        cv.putText(frames,str(self.left_hip_knee),(40,130),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

        return self.left_shoulder_hip,self.left_hip_knee
    
    def right_slope_condition(self,frames,llist,height,width):

        # self.right_full_hand = self.all_methods.slope(frames=frames,lmlist=llist,point1=16,point2=12,height=height,width=width)
        self.right_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=True)
        self.right_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=True)

        return self.right_shoulder_hip,self.right_hip_knee

    def start_left_exercise(self,frames):

        #this one check ashtanga namaskar and do bhujangasana
        start_ashtanga_left = (
                    self.left_elbow and 25 <= self.left_elbow <= 85 and 
                    self.left_hip and 110 <= self.left_hip <= 160 and 
                    self.left_shoulder and 0 <= self.left_shoulder <= 35 and
                    self.left_knee and 110 <= self.left_knee <= 160)
        if start_ashtanga_left:
            self.voice.playAudio(["move to next pose bhujangasana","please lay down hip,knee on the ground by taking image"],play=True)
            return True
        else:
            return False
    
    def start_right_exercise(self,frames):

        #this one check ashtanga namaskar and do bhujangasana
        start_ashtanga_right = (
                    self.right_elbow and 25 <= self.right_elbow <= 85 and
                    self.right_hip and 110 <= self.right_hip <= 160 and 
                    self.right_knee and 110 <= self.right_knee <= 160 and
                    self.right_shoulder and 0 <= self.right_shoulder <= 35)
        if start_ashtanga_right:
            self.voice.playAudio(["move to next pose bhujangasana","please lay down hip,knee on the ground by taking image"],play=True)
            return True
        else:
            return False

    def wrong_left(self,frames):

        #check hip is in flat position or not
        left_hip = (self.left_hip)
        if left_hip:
            # left_hip_flat = (self.left_hip and 171 <= self.left_hip <= 180)
            # if left_hip_flat:
            #     self.voice.playAudio(["you are in flat position bend your hip back side"],play=True)
            #     # self.all_methods.voice("you are in flat position bend your hip back side")

            #check hip is in too much bend position
            left_hip_bending = (self.left_hip and 0 <= self.left_hip <= 129)
            if left_hip_bending:
                self.voice.playAudio(["please raise your hip front side"],play=True)
                # self.all_methods.voice("you are in too bending position raise your hip front side")


        #check head position is in upside side or not
        head = (self.head_position)
        if head:
            head_position = (self.head_position and self.head_position != "Up")
            if head_position:
                self.voice.playAudio(["you must be look at up "],play=True)
                # self.all_methods.voice("you must be look at up side dont turn another side")


        #elbow is in staright position or not
        left_elbow_correct = (self.left_elbow)
        if left_elbow_correct:
            left_elbow = (self.left_elbow and 0 <= self.left_elbow <= 139)
            if left_elbow:
                self.voice.playAudio(["your hands must be in straight position"],play=True)
                # self.all_methods.voice("your hands must be in straight position")


        #check knee is in straight position or not
        left_knee_correct = (self.left_knee)
        if left_knee_correct:
            left_knee = (self.left_knee and 0 <= self.left_knee <= 149)
            if left_knee:
                self.voice.playAudio(["your legs also must be in straight position"],play=True)
                # self.all_methods.voice("your legs also must be in straight position")


        #shoulder is in close position so to raise the shoulder
        left_shoulder = (self.left_shoulder)
        if left_shoulder:
            left_shoulder_open = (self.left_shoulder and 0 <= self.left_shoulder <= 24)
            if left_shoulder_open:
                self.voice.playAudio(["open the shoulders "],play=True)
                # self.all_methods.voice("open the shoulders because you are closing too much")

            #shoulder is in open position so close shoulder 
            left_shoulder_up = (self.left_shoulder and 51 <= self.left_shoulder <= 180)
            if left_shoulder_up:
                self.voice.playAudio(["close the shoulders "],play=True)
                # self.all_methods.voice("close the shoulders because you are open too much")


        else:
            return
        
    def wrong_right(self,frames):

        #check hip is in flat position or not
        right_hip = (self.right_hip)
        if right_hip:
        #     right_hip_flat = (self.right_hip and 171 <= self.right_hip <= 180)
        #     if right_hip_flat:
        #         self.voice.playAudio(["you are in flat position bend your hip back side"],play=True)
                # self.all_methods.voice("you are in flat position bend your hip back side")

            #check hip is in too much bend position
            right_hip_bending = (self.right_hip and 0 <= self.right_hip <= 129)
            if right_hip_bending:
                self.voice.playAudio(["please raise your hip front side"],play=True)
                # self.all_methods.voice("you are in too bending position raise your hip front side")


        #check head position is in upside side or not
        head = (self.head_position)
        if head:
            head_position = (self.head_position and self.head_position != "Up")
            if head_position:
                self.voice.playAudio(["you must be look at up"],play=True)
                # self.all_methods.voice("you must be look at up side dont turn another side")


        #elbow is in staright position or not
        right_elbow_correct = (self.right_elbow)
        if right_elbow_correct:
            right_elbow = (self.right_elbow and 0 <= self.right_elbow <= 139)
            if right_elbow:
                self.voice.playAudio(["your hands must be in straight position"],play=True)
                # self.all_methods.voice("your hands must be in straight position")


        #check knee is in straight position or not
        right_knee_correct = (self.right_knee)
        if right_knee_correct:
            right_knee = (self.right_knee and 0 <= self.right_knee <= 149)
            if right_knee:
                self.voice.playAudio(["your legs also must be in straight position"],play=True)
                # self.all_methods.voice("your legs also must be in straight position")


        #shoulder is in close position so to raise the shoulder
        right_shoulder = (self.right_shoulder)
        if right_shoulder:
            right_shoulder_open = (self.right_shoulder and 0 <= self.right_shoulder <= 24)
            if right_shoulder_open:
                self.voice.playAudio(["open the shoulders"],play=True)
                # self.all_methods.voice("open the shoulders because you are closing too much")

            #shoulder is in open position so close shoulder 
            right_shoulder_up = (self.right_shoulder and 51 <= self.right_shoulder <= 180)
            if right_shoulder_up:
                self.voice.playAudio(["close the shoulders"],play=True)
                # self.all_methods.voice("close the shoulders because you are open too much")


        else:
            return
    
    def bhujangasana_left_name(self,frames):
        
        if (
            self.MIN_BACK_LIMT <= self.left_shoulder_hip <= self.MAX_BACK_LIMT and
            self.MIN_THIGH_LIMT <= self.left_hip_knee <= self.MAX_THIGH_LIMT and
            self.left_elbow and 140 <= self.left_elbow <= 180 and 
            self.left_knee and 150 <= self.left_knee <= 180 and
            self.left_hip and 130 <= self.left_hip <= 180 and
            self.left_shoulder and 25 <= self.left_shoulder <= 50  and
            self.head_position and self.head_position == "Up"):

            cv.putText(frames,str("Bhujangasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

            return True
        return False

    def bhujangasana_right_name(self,frames):

        if (
            self.MIN_BACK_LIMT <= self.right_shoulder_hip <= self.MAX_BACK_LIMT and
            self.MIN_THIGH_LIMT <= self.right_hip_knee <= self.MAX_THIGH_LIMT and
            self.right_elbow and 140 <= self.right_elbow <= 180 and 
            self.right_knee and 150 <= self.right_knee <= 180 and
            self.right_hip and 130 <= self.right_hip <= 180 and
            self.right_shoulder and 25 <= self.right_shoulder <= 50 and
            self.head_position and self.head_position == "Up"):

            cv.putText(frames,str("bhujangasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

            return True
        return False
    
    # to create a head_point for side view
        
    def head_value(self,frames,point,llist):

        if len(llist) != 0:
            self.head_val = self.head_point(frames=frames,lmlist=llist,points=point)

            if self.head_val :
                self.head_x = self.head_val[0]
                self.head_y = self.head_val[1]

                cv.circle(frames,(self.head_x,self.head_y),5,(255,0,0),2)

                return self.head_x
            

            return False
        return False
    
    # def check_left_right(self,frames):

    #     self.check= ""

    #     if len(llist) != 0:

    #         self.left_index_finger = llist[19][2]
    #         self.right_index_finger = llist[20][2]

    #         if self.left_index_finger < self.right_index_finger:
    #             self.check += "left"

    #         else :
    #             self.check += "right"

    #     return self.check 

def main():
    global llist
    video_capture = cv.VideoCapture(0)
    detect = bhujangasana()
    all_methods = allmethods()
    voice = VoicePlay()
    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        img = cv.imread("images/image7.jpg")
        detect.pose_positions(img,draw = False)
        llist = detect.pose_landmarks(img,draw=False)

        head_point = detect.head_value(img,llist=llist,point=(0,2,5))
        side_view = all_methods.findSideView(frame=img,FLAG_HEAD_OR_TAIL_POSITION=detect.HEAD_POSITION,head=head_point)
        #ear_hip_point = all_methods.ear_hip_side_view(img,lmlist=llist)

        standing_side_detect = all_methods.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)

        if len(llist) != 0:

            if standing_side_detect == "forward":
                voice.playAudio(["if you not understood taking by reference video"],play=True)

                if side_view =="right":
                    right_bhujanga = detect.bhujangasana_right(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),draw=False)
                    right_slope = detect.right_slope_condition(frames=frames,llist=llist,height=detect.h,width=detect.w)
                    start_right = detect.start_right_exercise(frames=frames)
                    if not start_right:
                        wrong_right = detect.wrong_right(frames=frames)
                    bhujanga_name = detect.bhujangasana_right_name(frames=frames)

                elif side_view == "left":
                    left_bhujanga = detect.bhujangasana_left(frames=frames,llist=llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),draw=False)
                    left_slope = detect.left_slope_condition(frames=frames,llist=llist,height=detect.h,width=detect.w)
                    start_left = detect.start_left_exercise(frames=frames)
                    if not start_left:
                        wrong_left = detect.wrong_left(frames=frames)
                    bhujanga_name = detect.bhujangasana_left_name(frames=frames)

        cv.imshow("video",img)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()