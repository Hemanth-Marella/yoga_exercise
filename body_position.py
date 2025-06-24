
import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math

from threading import Thread
from allMethods import allmethods
from face_detect import HeadPoseEstimator
from voiceModule import VoicePlay

class bodyPosition:
     
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
        self.voice = VoicePlay()
        self.position = None

        #shoulder hip
        self.MAX_BACK_LIMT = 50
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
                px,py ,pz= (poselms.x * self.w) , (poselms.y *self.h),(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lpslist

    def is_person_standing_sitting(self,frames,llist,hip_points,leg_points,elbow_points,height,width):

        #left slope condition
        self.left_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width,draw=False)
        cv.putText(frames,str(self.left_shoulder_hip),(10,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,0),2)
        self.left_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=False)

        # right slope condition
        self.right_shoulder_hip = self.all_methods.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=False)
        self.right_hip_knee = self.all_methods.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=False)

        #ground_left
        ground_left = self.all_methods.ground_distance_left(frames=frames,lmlist=llist)
        #ground_right
        ground_right = self.all_methods.ground_distance_right(frames=frames,lmlist=llist)

        hip_points,points_hip = self.calculate_angle(frames=frames,lmList=llist,points=hip_points,draw=True)
        leg_points,points_leg = self.calculate_angle(frames=frames,lmList=llist,points=leg_points,draw=True)
        elbow_points,point_elbow = self.calculate_angle(frames=frames,lmList=llist,points=elbow_points,draw=True)
        # bend_hip_point = hip_points // 2

        tolerance=0.05
        hip_y = points_hip[3]
        leg_y = points_leg [3]
        shoulder_y = point_elbow[1]

        if hip_points and leg_points:

            if( 
            ((ground_left and ground_left-100 <= leg_y <= ground_left) or (ground_right <= leg_y <= ground_right)) and
            ((ground_left and ground_left-150 <= hip_y <= ground_left) or (ground_right <= hip_y <= ground_right)) and
            ((self.MIN_BACK_LIMT <= self.left_shoulder_hip <= self.MAX_BACK_LIMT) or (self.MIN_BACK_LIMT <= self.right_shoulder_hip <= self.MAX_BACK_LIMT)) and
            ((self.MIN_THIGH_LIMT <=self.left_hip_knee <= self.MAX_THIGH_LIMT) or (self.MIN_THIGH_LIMT <=self.left_hip_knee <= self.MAX_THIGH_LIMT))):
                
                self.position = "sleeping"

                # self.voice.playAudio(["you are in sleeping position"],play=True)

            elif(
                shoulder_y < hip_y < leg_y and
                hip_points and 160 <= hip_points <= 180 and
                leg_points and 160 <= leg_points <= 180):
                
                self.position = "standing"
                
                # self.voice.playAudio(["you are in standing Position"],play=True)

            elif(
                hip_points and 70 <= hip_points <= 159 and
                leg_points and 120 <= leg_points <= 180
                ):

                self.position = "bending"

                # self.voice.playAudio(["you are in bending position"],play=True)
    

            else:

                self.position = "sitting"
                # self.voice.playAudio(["you are in sitting position"],play=True)

        # else:
        #     self.voice.playAudio(["move little back"],play=True)
        
        return self.position
 


def main():

    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980)  #height
    detect = bodyPosition()
    global llist

    while True:

        isTrue,frames = video_capture.read()

        detect.pose_positions(frames=frames,draw=False)
        llist = detect.pose_landmarks(frames=frames,draw=False)

        if llist:
            detect.is_person_standing_sitting(frames=frames,llist=llist,hip_points=(11,23,25),leg_points=(23,25,27),elbow_points=(11,13,15),height=detect.h,width=detect.w)

        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
# main()