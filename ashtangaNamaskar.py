import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math

from threading import Thread
from allMethods import allmethods
from voiceModule import VoicePlay
from face_detect import HeadPoseEstimator

class ashtanga:
     
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
        self.head_pose = HeadPoseEstimator()    

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        #shoulder hip
        self.MAX_BACK_LIMT = 40
        self.MIN_BACK_LIMT = 0
        
        #hip knee
        self.MAX_THIGH_LIMT = 40
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

            h,w,cl = frames.shape

            for id,poselms in enumerate(self.results.pose_landmarks.landmark):
                px,py,pz = int(poselms.x * w) , int(poselms.y * h),(poselms.z)

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
    
    #right side angles
    def right_side_ashtanga(self,frames,llist,elbow,hip,knee,shoulder,draw = True):

        self.right_elbow,points_cor1 = self.all_methods.calculate_angle(frames=frames,points= elbow, lmList=llist)
        self.right_hip ,points_cor2= self.all_methods.calculate_angle(frames=frames,points=hip, lmList=llist)
        self.right_knee,points_cor3 = self.all_methods.calculate_angle(frames=frames,points=knee, lmList=llist)
        self.right_shoulder,points_cor4 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)

        if draw:

            # if points_cor2 and points_cor6:
                # cv.line(frames,(points_cor2[2],points_cor2[3]),(points_cor6[2],points_cor6[3]),(255,0,0),2)
            if points_cor1:
                cv.putText(frames, str(int(self.right_elbow)), (points_cor1[2]+10, points_cor1[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor2:
                cv.putText(frames, str(int(self.right_hip)), (points_cor2[2]-20, points_cor2[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor3:
                cv.putText(frames, str(int(self.right_knee)), (points_cor3[2]-20, points_cor3[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor4:
                cv.putText(frames, str(int(self.right_shoulder)), (points_cor4[2]+10, points_cor4[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder  
    #left side angles
    def left_side_ashtanga(self,frames,llist,elbow, hip,knee,shoulder,draw = True):
        self.left_elbow,points_cor5 = self.all_methods.calculate_angle(frames=frames,points=elbow,lmList=llist)
        self.left_hip,points_cor6 = self.all_methods.calculate_angle(frames=frames,points=hip,lmList=llist)
        self.left_knee,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=knee,lmList=llist)
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder,lmList=llist)

        if draw:
            if points_cor5:
                cv.putText(frames, str(int(self.left_elbow)), (points_cor5[2]+10, points_cor5[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor6:
                cv.putText(frames, str(int(self.left_hip)), (points_cor6[2]+10, points_cor6[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor7:
                cv.putText(frames, str(int(self.left_knee)), (points_cor7[2]+10, points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor8:
                cv.putText(frames, str(int(self.left_shoulder)), (points_cor8[2]+10, points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder  


    # finding co ordinate points for head and fingers
    def horizontal_placed(self,frames,llist,face_point,points):

        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 

        if len(llist) != 0:

            self.head_points = self.head_point(frames=frames,lmlist=llist,points=face_point)
            self.head_x = self.head_points[0]
            self.head_y = self.head_points[1]
            self.head_z = self.head_points[2]

            self.right_finger_x,self.right_finger_y,self.right_finger_z = llist[points[0]][1:]
            self.left_finger_x,self.left_finger_y,self.left_finger_z= llist[points[1]][1:]
            cv.putText(frames,str(self.left_finger_z),(180,250),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
            self.right_foot_x,self.right_foot_y,self.right_foot_z = llist[points[2]][1:]
            self.left_foot_x , self.left_foot_y,self.left_foot_z = llist[points[3]][1:]
             

        return self.head_x,self.head_y,self.head_z

    def left_slope_condition(self,frames,llist,height,width):

        # self.left_full_hand = self.all_methods.slope(frames=frames,lmlist=llist,point1=15,point2=11,height=height,width=width)
        self.left_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width,draw=True)
        self.left_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=True)

        # cv.putText(frames,str(self.left_full_hand),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        # cv.putText(frames,str(self.left_shoulder_hip),(40,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        # cv.putText(frames,str(self.left_hip_knee),(40,130),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

        return self.left_shoulder_hip,self.left_hip_knee
    
    def right_slope_condition(self,frames,llist,height,width):

        # self.right_full_hand = self.all_methods.slope(frames=frames,lmlist=llist,point1=16,point2=12,height=height,width=width)
        self.right_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=True)
        self.right_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=True)

        return self.right_shoulder_hip,self.right_hip_knee
    

    def start_left_exercise(self,frames):

        self.plank_left = (
            
            self.left_shoulder and 50 <= self.left_shoulder <= 95 and
            self.left_elbow and 140 <= self.left_elbow <= 180 and
            self.left_hip and 140 <= self.left_hip <= 180 and 
            self.left_knee and 140 <= self.left_knee <= 180 and
            self.head_position and self.head_position == "Down")
        
        if self.plank_left:
            self.voice.playAudio(["move to next pose ashtanga namaskar","please lay down on the ground by taking image"],play=True)
            return True
        else:
            return False

    
    def start_right_exercise(self,frames):

        self.plank_right = (
            
            self.right_elbow and 140 <= self.right_elbow <= 180 and           
            self.right_hip and 140 <= self.right_hip <= 180 and            
            self.right_knee and 140 <= self.right_knee <= 180 and                                      
            self.right_shoulder and 50 <= self.right_shoulder <= 95 and
            self.head_position and self.head_position == "Down")
        
        if self.plank_right:
            self.voice.playAudio(["move to next pose ashtanga namaskar","please lay down on the ground by taking image"],play=True)
            return True
        else:
            return False
    
    def wrong_left(self,frames):

        #hip is in down position please raise up
        left_hip = (self.left_hip)
        if left_hip:
            left_hip_up = (self.left_hip and 0 <= self.left_hip <= 119)
            if left_hip_up:
                self.voice.playAudio([" please raise down your hip "],play=True)
                # self.all_methods.voice("your hip is in down position please raise up your hip till u reach the correct position")

            #hip is in up position please down
            left_hip_down = (self.left_hip and 151 <= self.left_hip <= 180)
            if left_hip_down:
                self.voice.playAudio(["please raise up your hip "],play=True)
                # self.all_methods.voice("your hip is in up position please raise down your hip till u reach the correct position")


        #check legs will open or not
        left_knee = (self.left_knee)
        if left_knee:
            left_knee_open = (self.left_knee and 0 <= self.left_knee <= 119)
            if left_knee_open:
                self.voice.playAudio(["please open the legs"],play=True)
                # self.all_methods.voice("please open the legs till reach the correct position")

            #check lega will close or not
            left_knee_close = (self.left_knee and 141 <= self.left_knee <= 180)
            if left_knee_close:
                self.voice.playAudio(["please close the legs or move back your hip "],play=True)
                # self.all_methods.voice("please close the legs or move back your hip ")


        #to check shoulder
        left_shoulder_correct = (self.left_shoulder)
        if left_shoulder_correct:
            left_shoulder = (self.left_shoulder and 26 <= self.left_shoulder <= 180)  
            if left_shoulder:
                self.voice.playAudio(["please fold your shoulders "],play=True)
                # self.all_methods.voice("please fold your shoulders like in image")


        #elbow is in down position please raise up
        left_elbow = (self.left_elbow)
        if left_elbow:
            left_elbow_up = (self.left_elbow and 0 <= self.left_elbow <= 24)
            if left_elbow_up:
                self.voice.playAudio(["your elbow is to much bend postion please raise up"],play=True)
                # self.all_methods.voice("your elbow is to much bend postion please raise up")

            #elbow is in down position please raise down
            left_elbow_down = (self.left_elbow and 86 <= self.left_elbow <= 180)
            if left_elbow_down:
                self.voice.playAudio(["your elbow is in up position please bend down "],play=True)
                # self.all_methods.voice("your elbow is in up position please bend down till u reach the correct position")


        #check head will cross the hands 
        head = (self.head)
        if head:
            head_left = (self.head_x > self.left_finger_x)
            if head_left:
                self.voice.playAudio(["your head must cross shoulder to left side which is shown in image"],play=True)
                # self.all_methods.voice("your head must cross shoulder to left side which is shown in image")


        else:
            return 
        
    def wrong_right(self,frames):

         #hip is in down position please raise up
        right_hip = (self.right_hip)
        if right_hip:
            right_hip_up = (self.right_hip and 0 <= self.right_hip <= 119)
            if right_hip_up:
                self.voice.playAudio(["please raise down your hip "],play=True)
                # self.all_methods.voice("your hip is in down position please raise up your hip ")

            #hip is in up position please down
            right_hip_down = (self.right_hip and 151 <= self.right_hip <= 180)
            if right_hip_down:
                self.voice.playAudio(["please raise up your hip "],play=True)
                # self.all_methods.voice("your hip is in up position please raise down your hip ")


        #check legs will open or not
        right_knee = (self.right_knee)
        if right_knee:
            right_knee_open = (self.right_knee and 0 <= self.right_knee <= 119)
            if right_knee_open:
                self.voice.playAudio(["please open the legs "],play=True)
                # self.all_methods.voice("please open the legs till reach the correct position")

            #check legs will close or not
            right_knee_close = (self.right_knee and 141 <= self.right_knee <= 180)
            if right_knee_close:
                self.voice.playAudio(["please close the legs or move back your hip "],play=True)
                # self.all_methods.voice("please close the legs or move back your hip till you reach the correct position")


        #to check shoulder
        right_shoulder_correct = (self.right_shoulder)
        if right_shoulder_correct:
            right_shoulder = (self.right_shoulder and 26 <= self.right_shoulder <= 180)  
            if right_shoulder:
                self.voice.playAudio(["please fold your shoulders"],play=True)
                # self.all_methods.voice("please fold your shoulders like in image")


        #elbow is in down position please raise up
        right_elbow = (self.right_elbow)
        if right_elbow:
            right_elbow_up = (self.right_elbow and 0 <= self.right_elbow <= 24)
            if right_elbow_up:
                self.voice.playAudio(["your elbow is to much bend postion please raise up"],play=True)
                # self.all_methods.voice("your elbow is to much bend postion please raise up")

            #elbow is in down position please raise down
            right_elbow_down = (self.right_elbow and 86 <= self.right_elbow <= 180)
            if right_elbow_down:
                self.voice.playAudio(["your elbow is in up position please bend down "],play=True)
                # self.all_methods.voice("your elbow is in up position please bend down ")


        #check head will be cross or not
        head = (self.head)
        if head:
            head_right = (self.head_x < self.right_finger_x)
            if head_right:
                self.voice.playAudio(["your head must cross shoulder to right side which is shown in image"],play=True)
                # self.all_methods.voice("your head must cross shoulder to right side which is shown in image")

        else:
            return 


    #left side view ashtanga name 
    def left_ashtanga_name(self,frames):
        
        if( 
        self.MIN_BACK_LIMT <=self.left_shoulder_hip <= self.MAX_BACK_LIMT and
        self.MIN_THIGH_LIMT <= self.left_hip_knee <= self.MAX_THIGH_LIMT and
        self.head_x < self.left_finger_x and
        self.left_elbow and 25 <= self.left_elbow <= 85 and 
        self.left_hip and 120 <= self.left_hip <= 150 and 
        self.left_shoulder and 0 <= self.left_shoulder <= 25 and
        self.left_knee and 120 <= self.left_knee <= 150):
            
            cv.putText(frames,str("Left_Ashtanga"),(40,80),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            return True
        return False

    #right side view ashtanga name
    def right_ashtanga_name(self,frames):

        if(
        self.MIN_BACK_LIMT <=self.right_shoulder_hip <= self.MAX_BACK_LIMT and
        self.MIN_THIGH_LIMT <= self.right_hip_knee <= self.MAX_THIGH_LIMT and
        self.head_x > self.right_finger_x and 
        self.right_elbow and 25 <= self.right_elbow <= 85 and
        self.right_hip and 120 <= self.right_hip <= 150 and 
        self.right_knee and 120 <= self.right_knee <= 150 and
        self.right_shoulder and 0 <= self.right_shoulder <= 25 ):
            
            cv.putText(frames,str("Right_Ashtanga"),(40,80),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            return True
        return False
    
def main():
    video_capture = cv.VideoCapture(0)
    global llist
    global slope_llist
    detect = ashtanga()
    all_methods = allmethods()
    voice = VoicePlay()
    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        img = cv.imread("images/image6.jpg")
        detect.pose_positions(frames,draw=False)
        llist = detect.pose_landmarks(frames,False)
        slope_llist = detect.slope_landmarks(frames=frames,draw=False)

        detect.horizontal_placed(frames,llist=llist,face_point=(0,2,5),points=(16,15,32,31))
        side_view = all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=detect.HEAD_POSITION,head=detect.head_x)
        
        standing_side_detect = all_methods.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)

        if len(llist) != 0:

            if standing_side_detect == "forward":
                voice.playAudio(["if you not understood taking by reference video"],play=True)
            
                if side_view == "right":
                    cv.putText(frames,str("right_side"),(40,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)                
                    detect.right_side_ashtanga(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26), knee=(24,26,28), shoulder=(14,12,24))
                    detect.right_slope_condition(frames=frames,llist=slope_llist,height=detect.h,width=detect.w)
                    detect.right_ashtanga_name(frames)  
        
                elif side_view == "left":
                    cv.putText(frames,str("left_side"),(40,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
                    detect.left_side_ashtanga(frames=frames,llist=llist,elbow=(11,13,15), hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23))
                    detect.left_slope_condition(frames=frames,llist=slope_llist,height=detect.h,width=detect.w)
                    detect.left_ashtanga_name(frames)                 

        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()