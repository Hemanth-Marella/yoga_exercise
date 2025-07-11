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

        self.check_sitting = False
        self.start_exercise = False
        self.initial_position = False
        self.check_body_turn = False
        self.pose_completed = False
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

        if not llist and len(llist):
            return None

        if not self.left_knee and not self.right_knee and not  self.left_shoulder and not self.right_shoulder and not self.left_shoulder_elbow_slope and not self.right_shoulder_elbow_slope and  not self.left_shoulder_hip_slope and not self.right_shoulder_hip_slope and not self.left_knee_ankle_slope and not self.right_knee_ankle_slope and not self.head_position :
            
            return None

        sitting_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,leg_points=(23,25,27),hip_points=(11,23,25),elbow_points=(11,13,15),height=height,width=width)

        if not self.start_exercise and sitting_position == "sitting":
            self.start_exercise = True
            self.check_sitting= True
            # return True

        elif not self.check_sitting and sitting_position != "sitting":
            self.all_methods.reset_after_40_sec()
            self.all_methods.play_after_40_sec(["please be in sitting position","   ","this yoga may started in sitting position"],llist=llist)
            return False


        #check person is in forward position or not
        if self.start_exercise:

            if not self.check_body_turn:

                side_view = self.all_methods.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)
                voice_list = ["you are in left side please turn forward","keep your legs straight"]
                if side_view == "right":
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are in right side please turn forward","keep your legs straight"],llist=llist)
                    return False

                elif side_view == "left":
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are in left side please turn forward","keep your legs straight"],llist=llist)

                elif side_view == "forward":
                    self.check_body_turn = True

            #check person is in initial position or not

            if self.check_body_turn:
            
                if not self.initial_position and 160 <= self.left_knee <= 180 and 160 <= self.right_knee <= 180:

                    self.initial_position = True

                elif not self.initial_position and 0 <= self.left_knee <= 159 and 0 <= self.right_knee <= 159:
                    self.all_methods.reset_after_40_sec()
                    self.all_methods.play_after_40_sec(["you are not in initial position","keep your legs straight"], llist=llist)

                if self.initial_position:
                    
                    if 160 <= self.left_knee <= 180 and 160 <= self.right_knee <= 180:

                        voice_list = ["you are in initial position , start gomukhasana","please keep your left foot under right thigh"]

                        if self.count < len(voice_list):

                            self.all_methods.reset_after_40_sec()
                            trigger = self.all_methods.play_after_40_sec([voice_list[self.count]], llist=llist)
                            if trigger:
                                self.count += 1

                        else:
                            self.count = 1

                    else:
                        
                        # LEFT KNEE
                        if (self.left_knee and (self.left_knee and 0 <= self.left_knee <= 49)):
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["Gently move your left leg outward."], llist=llist)
                            first_pose = True

                        elif (self.left_knee and (self.left_knee and 101 <= self.left_knee <= 180)):
                            self.all_methods.reset_after_40_sec()
                            self.all_methods.play_after_40_sec(["Bend your left knee and place your left foot under your right thigh"], llist=llist)
                            first_pose = True
                                
                        else:

                            # RIGHT KNEE
                            if self.right_knee and 0 <= self.right_knee <= 80:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["gently move your right leg outward slight"], llist=llist)

                            elif self.right_knee and 151 <= self.right_knee <= 180:
                                self.all_methods.reset_after_40_sec()
                                self.all_methods.play_after_40_sec(["Bend your right knee and place on top of your left knee"], llist=llist)
                            
                            else:

                                # LEFT SHOULDER
                                if self.left_shoulder and 0 <= self.left_shoulder <= 9:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["please move your left side chest part and shoulder and hold your right hand"], llist=llist)


                                elif self.left_shoulder and 31 <= self.left_shoulder <= 180:
                                    self.all_methods.reset_after_40_sec()
                                    self.all_methods.play_after_40_sec(["gently move your left side chest inward and hold your right hand"],llist=llist)

                                else:

                                    # RIGHT SHOULDER
                                    if self.right_shoulder and 0 <= self.right_shoulder <= 149:
                                        self.all_methods.reset_after_40_sec()
                                        self.all_methods.play_after_40_sec(["please raise your right hand and hold your left hand"], llist=llist)
                                            
                                    else:
                            
                                        # RIGHT SHOULDER-ELBOW SLOPE
                                        if self.right_shoulder_elbow_slope and 0 <= self.right_shoulder_elbow_slope <= 69:
                                            self.all_methods.reset_after_40_sec()
                                            self.all_methods.play_after_40_sec(["please keep your right hand in back side hold your left hand with right hand"], llist=llist)
                                                
                                        else:

                                            # LEFT SHOULDER-ELBOW SLOPE
                                            if self.left_shoulder_elbow_slope and 0 <= self.left_shoulder_elbow_slope <= 69:
                                                self.all_methods.reset_after_40_sec()
                                                self.all_methods.play_after_40_sec(["please keep your left hand in back side and hold your right hand"], llist=llist)
                                                
                                            else:

                                                # LEFT SHOULDER-HIP SLOPE
                                                if self.left_shoulder_hip_slope and 0 <= self.left_shoulder_hip_slope <= 74:
                                                    self.all_methods.reset_after_40_sec()
                                                    self.all_methods.play_after_40_sec(["maintain upper body straight"], llist=llist)     
                                                        
                                                else:
                                                    # RIGHT SHOULDER-HIP SLOPE
                                                    if self.right_shoulder_hip_slope and 0 <= self.right_shoulder_hip_slope <= 74:
                                                        self.all_methods.reset_after_40_sec()
                                                        self.all_methods.play_after_40_sec(["keep your upper body straight"], llist=llist)
                                                            
                                                    else:
                                                        # RIGHT KNEE-ANKLE SLOPE
                                                        if self.right_knee_ankle_slope and 61 <= self.right_knee_ankle_slope <= 90:
                                                            self.all_methods.reset_after_40_sec()
                                                            self.all_methods.play_after_40_sec(["please gently move your right leg inwards"], llist=llist)
                                                                
                                                        else:
                                                            # LEFT KNEE-ANKLE SLOPE
                                                            if self.left_knee_ankle_slope and 61 <= self.left_knee_ankle_slope <= 90:
                                                                self.all_methods.reset_after_40_sec()
                                                                self.all_methods.play_after_40_sec(["please gently move your left leg inwards"], llist=llist)   

                                                            else:
                                                                # HEAD POSITION
                                                                if self.head_position and self.head_position != "Forward":
                                                                    self.all_methods.reset_after_40_sec()
                                                                    self.all_methods.play_after_40_sec(["you should face your head to camera side"], llist=llist)

                                                                else:
                                                                    return True


    def gomukhasana_name(self,frames):

            correct = (
                    self.left_knee and 50 <= self.left_knee <= 100 and
                    self.right_knee and 81 <= self.right_knee <= 150 and
                    self.left_shoulder and 10 <= self.left_shoulder <= 30 and
                    self.right_shoulder and 150 <= self.right_shoulder <= 180 and
                    self.left_shoulder_elbow_slope and 70 <= self.left_shoulder_elbow_slope <= 95 and
                    self.right_shoulder_elbow_slope  and 70 <= self.right_shoulder_elbow_slope  <= 95 and
                    self.left_shoulder_hip_slope and 75 <= self.left_shoulder_hip_slope <= 100 and
                    self.right_shoulder_hip_slope and 75 <= self.right_shoulder_hip_slope <= 100 and
                    self.left_knee_ankle_slope and 0 <= self.left_knee_ankle_slope <= 60 and
                    self.right_knee_ankle_slope and 0 <= self.right_knee_ankle_slope <= 60 and
                    self.head_position and self.head_position == "Forward"
                   )
            
            if correct:
                self.all_methods.reset_after_40_sec()
                self.all_methods.play_after_40_sec(["you completed your yoga pose corrected"],llist=llist)
                return True

    def reverse_gomukhasana(self,frames,llist):

        reverse_correct = (
                    self.left_knee and 50 <= self.left_knee <= 100 and
                    self.right_knee and 81 <= self.right_knee <= 150 and
                    self.left_shoulder and 10 <= self.left_shoulder <= 30 and
                    self.right_shoulder and 150 <= self.right_shoulder <= 180 and
                    self.left_shoulder_elbow_slope and 70 <= self.left_shoulder_elbow_slope <= 95 and
                    self.right_shoulder_elbow_slope  and 70 <= self.right_shoulder_elbow_slope  <= 95 and
                    self.left_shoulder_hip_slope and 75 <= self.left_shoulder_hip_slope <= 100 and
                    self.right_shoulder_hip_slope and 75 <= self.right_shoulder_hip_slope <= 100 and
                    self.left_knee_ankle_slope and 0 <= self.left_knee_ankle_slope <= 60 and
                    self.right_knee_ankle_slope and 0 <= self.right_knee_ankle_slope <= 60 and
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
        img = cv.imread("images/image1.webp")
        # resized_img = cv.resize(img, None, fx=3.0, fy=3.0, interpolation=cv.INTER_LINEAR)

        # ret_ref, ref_frame = ref_video.read()
        # if not ret_ref:
        #     ref_video.set(cv.CAP_PROP_POS_FRAMES, 0)  # loop video
        #     ret_ref, ref_frame = ref_video.read()

        # # Resize reference video frame and place it at top-left
        # if ret_ref:
        #     ref_frame = cv.resize(ref_frame, (400, 300))
        #     h, w, _ = ref_frame.shape
        #     frames[0:h, 0:w] = ref_frame  # Overlay in corner


        if not flag:

            detect.pose_positions(frames,draw=False)
            llist = detect.pose_landmarks(frames,False)

            if len(llist) is None:
                return None

            if len(llist) != 0:

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