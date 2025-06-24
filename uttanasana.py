import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math
# import pyttsx3 as pyt
from threading import Thread
from allMethods import allmethods
from voiceModule import VoicePlay
from face_detect import HeadPoseEstimator

class uttanasana:
     
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
        self.voice_thread = None
        self.voice_detect = False
        self.all_methods = allmethods()
        self.caculate_angle = self.all_methods.calculate_angle
        self.head_point = self.all_methods.head_detect
        self.head_pose = HeadPoseEstimator()
        self.left_heel_detect = self.all_methods.left_heel_detect
        self.right_heel_detect = self.all_methods.right_heel_detect
        self.voice = VoicePlay()
        self.round = True

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"


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
                px,py,pz = int(poselms.x * self.w) , int(poselms.y * self.h),poselms.z

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py,pz),5,(255,255,0),2)

        return self.lpslist   

    def right_Uttanasana(self,frames,llist,elbow, hip, knee, shoulder,left_knee,draw = True):

        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 

        self.ground_right = self.all_methods.ground_distance_right(frames=frames,lmlist=llist)
        self.min_ground_right = self.ground_right-30
        

        self.right_elbow,points_cor1 = self.all_methods.calculate_angle(frames=frames,points= elbow, lmList=llist)
        self.right_hip ,points_cor2= self.all_methods.calculate_angle(frames=frames,points=hip, lmList=llist)
        self.right_knee,points_cor3 = self.all_methods.calculate_angle(frames=frames,points=knee, lmList=llist)
        self.right_shoulder,points_cor4 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.left_knee1,points_cor9 = self.all_methods.calculate_angle(frames=frames,points=left_knee, lmList=llist)

        
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
            # if points_cor9:
            #     cv.putText(frames, str(int(self.left_knee1)), (points_cor9[2]+10, points_cor9[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)


        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder


    def left_uttanasana(self,frames, llist,elbow, hip,knee,shoulder,right_knee,draw = True):

        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 

        self.ground_left = self.all_methods.ground_distance_left(frames=frames,lmlist=llist)
        self.min_ground_left = self.ground_left-30
        #print(self.ground_left)
        # cv.putText(frames,str(self.ground_left),(280,350),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)

        self.left_elbow,points_cor5 = self.all_methods.calculate_angle(frames=frames,points=elbow,lmList=llist)
        self.left_hip,points_cor6 = self.all_methods.calculate_angle(frames=frames,points=hip,lmList=llist)
        self.left_knee,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=knee,lmList=llist)
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder,lmList=llist)
        self.right_knee1,points_cor10 = self.all_methods.calculate_angle(frames=frames,points=right_knee, lmList=llist)


        if draw:

            if points_cor5:
                cv.putText(frames, str(int(self.left_elbow)), (points_cor5[2]+10, points_cor5[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor6:
                cv.putText(frames, str(int(self.left_hip)), (points_cor6[2]+10, points_cor6[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor7:
                cv.putText(frames, str(int(self.left_knee)), (points_cor7[2]+10, points_cor7[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor8:
                cv.putText(frames, str(int(self.left_shoulder)), (points_cor8[2]+10, points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            # if points_cor10:
            #     cv.putText(frames, str(int(self.right_knee1)), (points_cor10[2]+10, points_cor10[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)


        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder
    
        
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
    
    def start_left_exercise(self,frames):

        #this one check  hasta uttanasana before exercise uttanasana
        self.hasta_left = (
            self.head_x < self.left_heel_x and
            self.left_elbow and 110 <= self.left_elbow <= 180 and
            self.left_hip and 100 <= self.left_hip <= 180 and
            self.left_knee and 100 <= self.left_knee <= 180 and 
            self.left_shoulder and 100 <= self.left_shoulder <= 180)

        #this one check ashwa sanchalasana and do uttanasana
        self.ashwa_left = (
            self.min_ground_left-40 <= self.left_finger_y <= self.ground_left and
            ((self.left_finger_z < self.right_foot_z) or (self.left_finger_z < self.right_foot_z < self.right_finger_z)) and
            self.left_elbow and 140 <= self.left_elbow <= 180 and
            self.left_hip and 130 <= self.left_hip <= 180 and
            self.left_knee and 120 <= self.left_knee <= 180 and
            self.left_shoulder and 50 <= self.left_shoulder <= 100  and 
            self.head_position and self.head_position == "Left" #and
            # self.right_knee1 and 50 <= self.right_knee1 <= 100
            # self.right_leg and 45 <= self.right_leg <= 60
            )
        
        if self.round:
            
            if self.hasta_left:
                self.voice.playAudio(["move to next pose  uttanasana","bend your hip front side and maintain your elbows straight"],play=True)
                return True
            else:
                
                self.round = False
                return False
            
        elif not self.round:
            if self.ashwa_left:
                self.voice.playAudio(["move to next pose  uttanasana","bend your hip front side and maintain your elbows straight"],play=True)
                return True
            else:
                self.round = True
                return False

        # return False


    def start_right_exercise(self,frames):

        #this one check  hasta uttanasana before exercise uttanasana
        self.hasta_right = (
            self.head_x > self.right_heel_x and 
            self.right_elbow  and 110 <= self.right_elbow <= 180 and 
            self.right_hip and 100 <= self.right_hip <= 180 and  
            self.right_knee and 100 <= self.right_knee <= 180 and
            self.right_shoulder and 100 <= self.right_shoulder <= 180)

        #this one check ashwa sanchalasana and do uttanasana
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

        
        if self.round:
            if self.hasta_right:
                self.voice.playAudio(["move to next pose uttanasana ","bend your hip front side and maintain your elbows straight"],play=True)
                return True
            else:
                self.round = False
                return False
            
        elif not self.round:
            if self.ashwa_right:
                self.voice.playAudio(["move to next pose uttanasana ","bend your hip front side and maintain your elbows straight"],play=True)
                return True
            
            else:
                self.round = True
                return False

        # return False

    
    def wrong_right(self,frames):

        #to check right hip
        right_hp_correct = (self.right_hip) 
        if right_hp_correct:
            right_hip = (self.right_hip and 46 <= self.right_hip <= 180)
            if right_hip:
                self.voice.playAudio(["bend down your hip"],play=True)
                # self.all_methods.voice("your hip is in slightly up and bend down then your hip will bend")


        #to check right leg is correct not
        right_knee_correct = (self.right_knee)
        if right_knee_correct:
            right_knee = (self.right_knee and 0 <= self.right_knee <= 119)
            if right_knee:
                self.voice.playAudio(["please straight your legs"],play=True)
                # self.all_methods.voice("your right leg is not correctly bend please be straight your right leg")


        #to check right elbow
        right_elbow_correct = (self.right_elbow)
        if right_elbow_correct:
            right_elbow = (self.right_elbow and 0 <= self.right_elbow <= 119)
            if right_elbow:
                self.voice.playAudio(["please straight your elbows"],play=True)
                # self.all_methods.voice("your elbow is set in wrong position please be staright your elbow dont bend too much")


        # hand to touch the ground or not
        right_ground = (self.right_finger_y)
        if right_ground:
            right_ground_touch = (0 <= self.right_finger_y <= 599)
            if right_ground_touch:
                self.voice.playAudio(["come near to camera your hand is not touch to ground"],play=True)
                # self.all_methods.voice("your hand is not touch to ground perfectly please touch ground perfectly and maintain wrist is up on the right heel")


        #right shoulder in down position raised up
        right_shoulder = (self.right_shoulder)
        if right_shoulder:
            right_shoulder_up = (self.right_shoulder and 0 <= self.right_shoulder <= 79)
            if right_shoulder_up:
                self.voice.playAudio(["please open your shoulders"],play=True)
                # self.all_methods.voice("your shoulders are in down bended position please raise your shoulders slightly which is in image")


            #right shoulder in up position down shoulder
            right_shoulder_down = (self.right_shoulder and 115 <= self.right_shoulder <= 180)
            if right_shoulder_down:
                self.voice.playAudio(["please close your shoulders"],play=True)
                # self.all_methods.voice("your shoulders are in up position please down your shoulders slightly which is in image")


        #right side view to check hands will be before to legs or not
        right_side_view = ((self.right_finger_z > self.right_foot_z) or (self.right_finger_z > self.right_foot_z > self.right_foot_z > self.right_finger_z))
        if right_side_view:
            self.voice.playAudio(["your hand is always before your right leg and right hand is view to camera"],play =True)
            # self.all_methods.voice("your hand is always before your right leg and right hand is view to camera first")


        #head position
        head_correct = (self.head_position)
        if head_correct:
            head = (self.head_position and self.head_position != "Down")
            if head:
                self.voice.playAudio("your head must be in down position")

        #ground touch
        right_ground_touch_correct = (self.right_finger_y)
        if right_ground_touch_correct:
            right_ground_touch = (0 <= self.right_finger_y <= self.min_ground_right-20)
            if right_ground_touch:
                self.voice.playAudio(["your hand must touch ground"],play=True)

        else:
            return 

    def wrong_left(self,frames):

        #to check left hip 
        left_hp_correct = (self.left_hip) 
        if left_hp_correct:
            left_hip = (self.left_hip and 46 <= self.left_hip <= 180)
            if left_hip:
                self.voice.playAudio(["bend down your hip"],play=True)
                # self.all_methods.voice("your hip is in slightly up and bend down then your hip will bend")


        #to check left leg is correct not
        left_knee_correct = (self.left_knee)
        if left_knee_correct:
            left_knee = (self.left_knee and 0 <= self.left_knee <= 119)
            if left_knee:
                self.voice.playAudio(["please straight your legs"],play=True)
                # self.all_methods.voice("your left leg is not correctly bend please be straight your left leg")


        #to check left elbow
        left_elbow_correct = (self.left_elbow)
        if left_elbow_correct:
            left_elbow = (self.left_elbow and 0 <= self.left_elbow <= 119)
            if left_elbow:
                self.voice.playAudio(["please straight your elbows"],play=True)
                # self.all_methods.voice("your elbow is set in wrong position please be left your elbow dont bend too much")


        # hand to touch the ground or not
        left_ground = (self.left_finger_y)
        if left_ground:
            left_ground_touch = (0 <= self.left_finger_y <= 599)
            if left_ground_touch:
                self.voice.playAudio(["come near to camera your hand is not touch to ground"],play=True)
                # self.all_methods.voice("your hand is not touch to ground perfectly please touch ground perfectly and maintain wrist is up on the left heel")


        #left shoulder in down position raised up
        left_shoulder = (self.left_shoulder)
        if left_shoulder:
            left_shoulder_up = (self.left_shoulder and 0 <= self.left_shoulder <= 79)
            if left_shoulder_up:
                self.voice.playAudio(["please open your shoulders"],play=True)
                # self.all_methods.voice("your shoulders are in down bended position please raise your shoulders slightly which is in image")

            #left shoulder in up position down shoulder
            left_shoulder_down = (self.left_shoulder and 115 <= self.left_shoulder <= 180)
            if left_shoulder_down:
                self.voice.playAudio(["please close your shoulders"],play=True)
                # self.all_methods.voice("your shoulders are in up position please down your shoulders slightly which is in image")


        #left side view to check hands will be before to legs or not
        left_side_view = ((self.left_finger_z > self.left_foot_z) or (self.left_finger_z > self.left_foot_z > self.right_foot_z > self.right_finger_z))
        if left_side_view:
            self.voice.playAudio(["your hand is always before your left leg and left hand is view to camera"],play =True)
            # self.all_methods.voice("your hand is always before your left leg and left hand is view to camera first")


        #head position
        head_correct = (self.head_position)
        if head_correct:
            head = (self.head_position and self.head_position != "Down")
            if head:
                self.voice.playAudio("your head must be in down position")

        #griound touch
        left_ground_touch_correct = (self.left_finger_y)
        if left_ground_touch_correct:
            left_ground_touch = (0 <= self.left_finger_y <= self.min_ground_left-20)
            if left_ground_touch:
                self.voice.playAudio(["your hand must touch ground"],play=True)

        else:
            return 
        

    def left_side_uttanasana_name(self,frames):

        # self.all_methods.voice("move to next exercise")

        if(self.head_position and self.head_position == "Down" and
             self.min_ground_left <= self.left_finger_y <= self.ground_left  and
            ((self.left_finger_z < self.left_foot_z) or (self.left_finger_z < self.left_foot_z < self.right_foot_z < self.right_finger_z)) and
            self.left_elbow and 120 <= self.left_elbow <= 180 and 
            self.left_hip and 25 <= self.left_hip <= 45 and 
            ((self.left_knee and 120 <= self.left_knee <= 150) or (self.left_knee and 170 <= self.left_knee <= 180))and 
            self.left_shoulder and 80 <= self.left_shoulder <= 110): # and
            

            return True
        return False

    def right_side_uttanasana_name(self,frames): 

        # self.all_methods.voice("move to next exercise")
        
        if(self.head_position and self.head_position == "Down" and
             self.min_ground_right <= self.right_finger_y <= self.ground_right and
            ((self.right_foot_z > self.right_finger_z) or (self.right_finger_z < self.right_foot_z < self.left_foot_z < self.left_finger_z)) and
            self.right_elbow and 120 <= self.right_elbow <= 180 and 
            self.right_hip and 25 <= self.right_hip <= 45 and 
            ((self.right_knee and 120 <= self.right_knee <= 150) or (self.right_knee and 151 <= self.right_knee <= 180)) and 
            self.right_shoulder and 80 <= self.right_shoulder <= 110):
        
            cv.putText(frames,str("Right_uttanasana"),(140,150),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            return True
            
        return False
    

    def head_value(self,frames,point):

        if len(llist) != 0:
            self.point_x,self.point_y,self.point_z = llist[0][1:]
            
            self.head_val = self.head_point(frames=frames,lmlist=llist,points=point)

            if self.head_val :
                self.head_x = self.head_val[0]
                self.head_y = self.head_val[1]

                cv.circle(frames,(self.head_x,self.head_y),5,(255,0,0),2)

                return self.head_x
            

            return False
        return False
    
    def standing_side_view_detect(self,frames,llist,draw = True):

        if len(llist) != 0:

            self.check_stand = self.body_position.is_person_standing_sitting(frames=frames,llist=llist,hip_points=(11,23,25),leg_points=(23,25,27),elbow_points=(11,13,15),height=self.h,width=self.w)

            if self.check_stand != 'sleeping':

                left_shoulder_x,left_shoulder_y,left_shoulder_z = llist[11][1:]
                right_shoulder_x,right_shoulder_y,right_shoulder_z = llist[12][1:]
                left_ear = llist[7][1:]
                right_ear = llist[8][1:] 

                # Basic side detection using shoulder z-values or visibility
                z_diff = abs(left_shoulder_z - right_shoulder_z)

                if z_diff > 0.1 and z_diff < 0.31:
                    if left_shoulder_z > right_shoulder_z:
                        position = "right cross position"
                    else:
                        position = "left side cross position"

                elif z_diff > 0.31:
                    if left_shoulder_z > right_shoulder_z:
                        position = "right"
                    else:
                        position = "left"

                else:
                    position = "forward"

                cv.putText(frames, f"Position: {position}", (10, 100),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                cv.putText(frames, f"diff: {z_diff}", (10, 50),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
            elif self.check_stand == 'sleeping':
                self.voice.playAudio(["you must be in standing or sitting position suitable for exercise"],play=True)

            return position
        return False

    
def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720) 
    global llist
    all_methods = allmethods()
    detect = uttanasana()
    voice = VoicePlay()

    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        img1 = cv.imread("images/image3.webp")
        img = cv.resize(img1, None, fx=2.0, fy=2.0, interpolation=cv.INTER_LINEAR)
        detect.pose_positions(frames,draw = False)
        llist = detect.pose_landmarks(frames,draw=False)

        detect.hands_legs_correct_position(frames,llist=llist,points=(16,15,32,31))
        detect.head_value(frames,(0,2,5))
        side_view = all_methods.standing_side_view_detect(frames,llist=llist,height=height,width=width)

        if len(llist) != 0:

            
            if side_view == "right":
                
                detect.right_Uttanasana(frames,llist,elbow=(12,14,16), hip=(12,24,26), knee=(24,26,28), shoulder=(14,12,24),draw=False)
                wrong_right = detect.wrong_right(frames=frames)
                correct = detect.right_side_uttanasana_name(frames)


            elif side_view =="left":
                
                detect.left_uttanasana(frames,llist,elbow=(11,13,15),hip= (11,23,25),knee= (23,25,27), shoulder=(13,11,23),draw=False)
                wrong_left = detect.wrong_left(frames=frames)
                correct = detect.left_side_uttanasana_name(frames)
                if correct is True:
                    cv.putText(frames,str("left_Hasta uttanasana"),(170,180),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

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