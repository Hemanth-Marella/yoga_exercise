import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math

import pyttsx3 as pyt
from threading import Thread
import threading
from voiceModule import VoicePlay
from allMethods import allmethods
from face_detect import HeadPoseEstimator
# from body_position import bodyPosition
from abstract_class import yoga_exercise


class gomukhasana(yoga_exercise):
    
    def __init__(self,mode = False,mindetectconf=0.5,mintrcackconf=0.5):
        # c += 1
        # print(c)
        # print('again......................................................................')
        self.results = None
        self.mode = mode
        self.mindetectconf = mindetectconf
        self.mintrcackconf = mintrcackconf

        
        self.initial_position = False
        self.check_body_turn = False
        self.pose_completed = False
        self.half_pose = False
        self.count = 0

        self.mpPose = mp.solutions.pose
        self.pose= self.mpPose.Pose(
            static_image_mode = self.mode,
            min_detection_confidence = self.mindetectconf,
            min_tracking_confidence = self.mintrcackconf
        )
        self.mpDraw= mp.solutions.drawing_utils

        self.lmlist = []
        self.m_lmlist = []
        self.angle =0
        self.voice_thread = None
        self.voice_detect = False
        self.lock = threading.Lock()
        self.voice = VoicePlay()
        self.all_methods = allmethods()
        self.head_pose = HeadPoseEstimator()
        # self.stop_ = False

        
        self.forward_count = 0

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

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

        self.lmlist =[]

        if self.results.pose_landmarks:

            self.h,self.w,cl = frames.shape

            for id,poselms in enumerate(self.results.pose_landmarks.landmark):
                px,py ,pz= (poselms.x * self.w) , (poselms.y * self.h),(poselms.z)

                self.lmlist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lmlist
    
    def gomukhasana(self, frames,llist, left_elbow, left_hip, left_knee,left_shoulder,right_elbow, right_hip, right_knee, right_shoulder,draw =True):

        if not llist and len(llist) is None:
            return None
        
        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.head_position_detect(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        self.left_elbow, left_elbow_coords = self.all_methods.calculate_angle(frames=frames, points=left_elbow,lmList=llist)
        self.left_hip, left_hip_coords = self.all_methods.calculate_angle(frames=frames,points= left_hip,lmList=llist)
        self.left_knee, left_knee_coords = self.all_methods.calculate_angle(frames=frames,points= left_knee,lmList=llist)       
        self.left_shoulder,left_points_cor8 = self.all_methods.calculate_angle(frames=frames,points=left_shoulder, lmList=llist)

        self.right_elbow, right_elbow_coords = self.all_methods.calculate_angle(frames=frames, points=right_elbow,lmList=llist)
        self.right_hip, right_hip_coords = self.all_methods.calculate_angle(frames=frames, points=right_hip,lmList=llist)
        self.right_knee, right_knee_coords = self.all_methods.calculate_angle(frames=frames,points= right_knee,lmList=llist)
        self.right_shoulder,right_points_cor7 = self.all_methods.calculate_angle(frames=frames,points=right_shoulder, lmList=llist)

        if draw:

            cv.putText(frames,f'l_elbow{str(self.left_elbow)}',(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_hip{str(self.left_hip)}',(10,80),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_knee{str(self.left_knee)}',(10,120),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'l_shoulder{str(self.left_shoulder)}',(10,160),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)

            cv.putText(frames,f'r_elbow{str(self.right_elbow)}',(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_hip{str(self.right_hip)}',(10,80),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_knee{str(self.right_knee)}',(10,120),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            cv.putText(frames,f'r_shoulder{str(self.right_shoulder)}',(10,160),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),2)

        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder,self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,self.head_position

    def check_slopes(self,frames,lmlist,height,width):

        self.left_shoulder_elbow_slope = self.all_methods.slope(frames=frames,lmlist=lmlist,point1=11,point2=13,height=height,width=width,draw=True)
        self.right_shoulder_elbow_slope = self.all_methods.slope(frames=frames,lmlist=lmlist,point1=12,point2=14,height=height,width=width,draw=True)

        self.left_knee_ankle_slope = self.all_methods.slope(frames=frames,lmlist=lmlist,point1=25,point2=27,height=height,width=width,draw=True)
        self.right_knee_ankle_slope = self.all_methods.slope(frames=frames,lmlist=lmlist,point1=26,point2=28,height=height,width=width,draw=True)

        self.left_shoulder_hip_slope = self.all_methods.slope(frames=frames,lmlist=lmlist,point1=11,point2=23,height=height,width=width,draw=True)
        self.right_shoulder_hip_slope = self.all_methods.slope(frames=frames,lmlist=lmlist,point1=12,point2=24,height=height,width=width,draw=True)

        return self.left_shoulder_elbow_slope, self.right_shoulder_elbow_slope,self.left_knee_ankle_slope,self.right_knee_ankle_slope,self.left_shoulder_hip_slope,self.right_shoulder_hip_slope
    

    def wrong_gomukhasana(self,frames,llist,height,width): 

        self.all_methods.all_x_values(frames=frames,llist=llist)
        self.all_methods.all_y_values(frames=frames,llist=llist)
        self.all_methods.all_z_values(frames=frames,llist=llist)

        if not llist and len(llist):
            return None

        self.side_view = self.check_turn(frames=frames,llist=llist,height=height,width=width) 

        if not self.check_body_turn:

            if self.side_view == "side":
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are in side view,please turn forward","keep your legs straight"],llist=llist)
                
            elif self.side_view == "forward":
                self.check_body_turn = True

        if self.check_body_turn:

            if not self.initial_position and 160 <= self.left_knee <= 180 and 160 <= self.right_knee <= 180 and 160 <= self.left_hip <= 180:
                self.initial_position = True

            elif not self.initial_position and 0 <= self.left_knee <= 159 and 0 <= self.right_knee <= 159 and 0 <= self.left_hip <= 159:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you are not in initial position","keep your legs straight"], llist=llist)

            if self.initial_position and not self.half_pose:
                
                if  160 <= self.left_knee <= 180 and 160 <= self.right_knee <= 180 and 170 <= self.left_hip <= 180:

                    voice_list = ["you are in initial position , start gomukhasana","please, keep your left foot under right thigh"]

                    if self.count < len(voice_list):

                        self.all_methods.reset_after_40_sec()
                        trigger = self.all_methods.play_after_40_sec([voice_list[self.count]], llist=llist)
                        if trigger:
                            self.count += 1

                    else:
                        self.count = 1

                else:
                    
                    # LEFT KNEE
                    if (self.all_methods.l_toe_x > self.all_methods.r_hip_x ):

                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["Bend your left knee , and place your left foot, under your right thigh"],llist=llist)
                            
                    else:

                        # RIGHT KNEE
                        if (self.all_methods.r_toe_x < self.all_methods.l_hip_x ):

                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please bend your right knee , place on top of your left knee"],llist=llist)

                        else:
                            x_values_knees_diff = abs(int(self.all_methods.l_knee_x - self.all_methods.r_knee_x))
                            y_knees_diff = abs(int(self.all_methods.l_knee_y - self.all_methods.r_knee_y))
                            # cv.putText(frames,f"x_{x_values_knees_diff}",(30,60),2,cv.FONT_HERSHEY_PLAIN,(255,0,255),2)
                            # cv.putText(frames,f"y_{y_knees_diff}",(30,100),2,cv.FONT_HERSHEY_PLAIN,(255,0,255),2)

                            if x_values_knees_diff > 60 or y_knees_diff > 40:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please ,place your both knees in correct position"],llist=llist)
                            else:

                                if (self.all_methods.l_toe_x < self.all_methods.r_hip_x and
                                    self.all_methods.r_toe_x > self.all_methods.l_hip_x and
                                    x_values_knees_diff < 60 and y_knees_diff < 40):

                                    self.all_methods.reset_after_40_sec()
                                    half_voice = self.all_methods.play_after_40_sec(["excellent , you completed half pose"],llist=llist)
                                    if half_voice:
                                        self.half_pose = True
                        
            if self.half_pose:

                if (self.right_shoulder and 0 <= self.right_shoulder <= 140):
                    
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["please raise your right hand up"],llist=llist)

                else:
                    
                    if (self.all_methods.r_wrist_y > self.all_methods.r_elbow_y or 
                        self.right_elbow and 160 <= self.right_elbow <= 180):
                        
                        self.all_methods.reset_after_40_sec()
                        self.all_methods.play_after_40_sec(["good , bend your right hand"],llist=llist)

                    else:

                        if (self.all_methods.r_wrist_z < self.all_methods.l_shoulder_z):

                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["please keep your right hand back your head"],llist=llist)

                        else:

                            if (self.all_methods.l_elbow_y < self.all_methods.l_shoulder_y ):
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["please keep your hand down of your shoulders"],llist=llist)

                            else:
                                if (self.all_methods.l_wrist_z < self.all_methods.l_shoulder_z):
                                                    
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please keep your left hand ,back of your upper body, and, bend, hold another hand"],llist=llist)
                                
                                else:
                                    # HEAD POSITION
                                    if self.head_position and self.head_position != "Forward":
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["you should face your head to camera side"], llist=llist)

                                    else:
                                        return True

    def check_sitting(self,frames,llist,height,width):

        if len(llist) == 0:
            return None

        sitting_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if sitting_position == "sitting":
            return True

        elif sitting_position != "sitting":

            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be in sitting position, this yoga may started in sitting position"],llist=llist)
            # return False

    def check_turn(self,frames,llist,height,width):

        result = None

        self.all_methods.all_x_values(frames=frames,llist=llist)
        self.all_methods.all_y_values(frames=frames,llist=llist)
        self.all_methods.all_z_values(frames=frames,llist=llist)

        shoulder_z = abs(self.all_methods.l_shoulder_z - self.all_methods.r_shoulder_z)
        hip_z = abs(self.all_methods.l_hip_z - self.all_methods.r_hip_z)

        if (shoulder_z > 0.15 and hip_z > 0.15):
            result = "side"

        elif (shoulder_z < 0.15 and hip_z < 0.15):
            result = "forward"

        return result

    def gomukhasana_name(self,frames):

            correct = (
                    self.all_methods.r_wrist_z > self.all_methods.l_shoulder_z and
                    self.all_methods.l_wrist_z > self.all_methods.l_shoulder_z and
                    self.head_position and self.head_position == "Forward"
                   )
            
            if correct:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you completed your yoga pose corrected"],llist=llist)
                return True

    def reverse_gomukhasana(self,frames,llist):

        reverse_correct = (
                    self.all_methods.r_wrist_z > self.all_methods.l_shoulder_z and
                    self.all_methods.l_wrist_z > self.all_methods.l_shoulder_z and
                    self.head_position and self.head_position == "Forward"
                   )
        
        if reverse_correct:

            voice_list = ["good , stay in same position , wait for other instruction","very good keep your legs straight"]

            if self.forward_count < len(voice_list):
                self.all_methods.reset_after_40_sec()
                trigger = self.all_methods.play_after_40_sec([],llist=llist)
                if trigger:
                    self.forward_count += 1

            else:
                self.forward_count = 1

        elif not self.pose_completed and 0 <= self.left_knee <= 159 and 0<= self.right_knee <= 159:

            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please keep your legs straight"],llist=llist)

        elif not self.pose_completed and 160 <= self.left_knee <= 180 and 160 <= self.right_knee <= 180:
            self.all_methods.reset_after_40_sec()
            final = self.all_methods.play_after_40_sec(["good job you completed your yoga and back to relax"],llist=llist)
            if final:
                self.pose_completed = True
                return True

def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    global  llist
    detect = gomukhasana()
    is_person_sitting = False
    check_wrong = False
    check_position = False
    flag = False
    ref_video = cv.VideoCapture("second videos/gomukhasana.mp4")

    # if flag:
    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        # img = cv.imread("images/image1.webp")

        if not flag:

            detect.pose_positions(frames,draw=False)
            llist = detect.pose_landmarks(frames,False)
            sitting_detect = detect.check_sitting(frames=frames,llist=llist,height=height,width=width)
            detect.check_turn(frames=frames,llist=llist,height=height,width=width)

            if len(llist) is None:
                return None
            
            if not is_person_sitting:
                if sitting_detect :
                    is_person_sitting = True

            if is_person_sitting:

                detect.gomukhasana(frames=frames,llist=llist,left_elbow=(11,13,15), left_hip=(11,23,25), left_knee=(23,25,27), left_shoulder=(13,11,23),right_elbow=(12,14,16), right_hip=(12,24,26),right_knee= (24,26,28), right_shoulder=(14,12,24),draw=False)
                detect.check_slopes(frames=frames,lmlist=llist,height=height,width=width)

                if not check_wrong:
                
                    wrong = detect.wrong_gomukhasana(frames=frames,llist=llist,height=height,width=width)
                    if wrong:
                        check_wrong = True

                if check_wrong and not check_position:
                    correct = detect.gomukhasana_name(frames=frames)
                    if correct:
                        check_position = True

                if check_position:
                    
                    reverse = detect.reverse_gomukhasana(frames=frames,llist=llist)
                    if reverse:
                        return True
 
                    
        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()

