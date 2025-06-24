import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math
from threading import Thread
from allMethods import allmethods
from face_detect import HeadPoseEstimator
from voiceModule import VoicePlay

class ashwa:
     
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
        # self.speak = self.all_methods.speak
        self.caculate_angle = self.all_methods.calculate_angle
        self.count = 0
        self.prev_position = 0
        self.head_point = self.all_methods.head_detect
        self.voice = VoicePlay()
        
        self.movement = False

        self.head_pose = HeadPoseEstimator()  

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        self.round = True

        self.head_x = None

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
                px,py,pz = (poselms.x * self.w) , (poselms.y * self.h),(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py,pz),5,(255,255,0),2)

        return self.lpslist

    def right_ashwa(self,frames,llist,elbow, hip, knee, shoulder,left_knee, draw = True):

        self.ground_right = self.all_methods.ground_distance_right(frames=frames,lmlist=llist)
        self.min_ground_right = self.ground_right-40
        # cv.putText(frames,str(self.ground_right),(280,350),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
        #print("ground_right",self.ground_right)

        self.right_elbow,points_cor1 = self.all_methods.calculate_angle(frames=frames,points= elbow, lmList=llist)
        self.right_hip ,points_cor2= self.all_methods.calculate_angle(frames=frames,points=hip, lmList=llist)
        self.right_knee,points_cor3 = self.all_methods.calculate_angle(frames=frames,points=knee, lmList=llist)
        self.right_shoulder,points_cor4 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.left_knee1,points_cor9 = self.all_methods.calculate_angle(frames=frames,points=left_knee, lmList=llist)
        
        if draw:
            
            if points_cor1:
                cv.putText(frames, str(int(self.right_elbow)), (points_cor1[2]+10, points_cor1[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor2:
                cv.putText(frames, str(int(self.right_hip)), (points_cor2[2]-20, points_cor2[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor3:
                cv.putText(frames, str(int(self.right_knee)), (points_cor3[2]-20, points_cor3[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            if points_cor4:
                cv.putText(frames, str(int(self.right_shoulder)), (points_cor4[2]+10, points_cor4[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        #     if points_cor9:
        #         cv.putText(frames, str(int(self.left_knee1)), (points_cor9[2]+10, points_cor9[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            
            
        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,#self.left_knee1
    

    def left_ashwa(self,frames,llist,elbow, hip,knee,shoulder,right_knee,draw = True):

        self.ground_left = self.all_methods.ground_distance_left(frames=frames,lmlist=llist)
        self.min_ground_left = self.ground_left-40
        # cv.putText(frames,str(self.ground_left),(280,350),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
        #print("ground_left",self.ground_left)

        self.left_elbow,points_cor5 = self.all_methods.calculate_angle(frames=frames,points=elbow, lmList=llist)
        self.left_hip,points_cor6 = self.all_methods.calculate_angle(frames=frames,points=hip, lmList=llist)
        self.left_knee,points_cor7 = self.all_methods.calculate_angle(frames=frames,points=knee, lmList=llist)
        self.left_shoulder,points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
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

        return self.left_elbow,self.left_knee,self.left_hip,self.left_shoulder #,self.right_knee1

    def hands_legs_correct_position(self,frames,llist,points):

        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 

        if len(llist) != 0:

            self.right_finger_x,self.right_finger_y,self.right_finger_z = llist[points[0]][1:]
            self.left_finger_x,self.left_finger_y,self.left_finger_z = llist[points[1]][1:]
            self.right_foot_x,self.right_foot_y,self.right_foot_z = llist[points[2]][1:]
            self.left_foot_x , self.left_foot_y,self.left_foot_z = llist[points[3]][1:]


            # cv.putText(frames,str(self.left_finger_y),(180,250),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)

            # cv.circle(frames,int(self.left_finger_x,self.left_finger_y),(5),(0,255,0),2)

            return self.left_finger_x,self.left_finger_y,self.left_foot_x,self.left_foot_y,self.right_finger_x,self.right_finger_y,self.right_foot_x,self.right_foot_y
        

    def right_leg_slope(self,frames,llist,right_hip,right_knee,height,width):

        self.right_leg = self.all_methods.slope(frames=frames,lmlist=llist,point1=right_hip,point2=right_knee,height=height,width=width,draw=True)
        cv.putText(frames, str(int(self.right_leg)),(50,80), cv.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 2)
        return self.right_leg

    def left_leg_slope(self,frames,llist,left_hip,left_knee,height,width):
        self.left_leg = self.all_methods.slope(frames=frames,lmlist=llist,point1=left_hip,point2=left_knee,height=height,width=width,draw=True)
        cv.putText(frames, str(int(self.left_leg)),(50,80), cv.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 2)
        return self.left_leg

    def left_distance(self,frames,llist,left_hand,left_toe_point,face_points):

        x2,y2 ,z2= llist[left_toe_point][1:]

        self.head_points = self.head_point(frames=frames,lmlist=llist,points= face_points)
        self.head_x = self.head_points[0]
        self.head_y = self.head_points[1]
        self.head_z = self.head_points[2]

        left_points_x,left_points_y ,left_points_z= llist[left_hand][1:]
        self.palm_distance = int(math.sqrt((left_points_y-y2)**2 +(left_points_x - x2) ** 2))
        self.head_distance = int(math.sqrt((self.head_y-y2) **2 + (self.head_x - x2) **2))

        return self.palm_distance,self.head_distance,self.head_x,self.head_y
    
    def right_distance(self,frames,llist,right_hand,right_toe_point,face_points):

        if len(llist) != 0:

            x2,y2,z2 = llist[right_toe_point][1:]

            self.head_points = self.head_point(frames=frames,lmlist=llist,points= face_points)
            self.head_x = self.head_points[0]
            self.head_y = self.head_points[1]
            self.head_z = self.head_points[2]

            right_points_x,right_points_y,right_points_z = llist[right_hand][1:]

            self.head_distance = int(math.sqrt((self.head_y-y2) **2 + (self.head_x - x2) **2))
            self.palm_distance = int(math.sqrt((right_points_y-y2)**2 +(right_points_x - x2) ** 2))
            
            return self.palm_distance,self.head_distance,self.head_x,self.head_y
        

    def start_left_exercise(self,frames):

        #this one is to check uttanasana and do ashwa sanchalasana
        self.uttas_left = (
            self.head_position and self.head_position == "Down" and
             self.min_ground_left <= self.left_finger_y <= self.ground_left  and
            ((self.left_finger_z < self.left_foot_z) or (self.left_finger_z < self.left_foot_z < self.right_foot_z < self.right_finger_z)) and
            self.left_elbow and 120 <= self.left_elbow <= 180 and 
            self.left_hip and 15 <= self.left_hip <= 55 and 
            ((self.left_knee and 120 <= self.left_knee <= 150) or (self.left_knee and 151 <= self.left_knee <= 180))and 
            self.left_shoulder and 50 <= self.left_shoulder <= 120)

        #this one is to check parvatasana and do ashwa sanchalasana
        self.parvatasana_left = (
            self.left_finger_z < self.head_z and
            self.palm_distance and 350 <= self.palm_distance <= 600 and
            self.left_elbow and 120 <= self.left_elbow <= 180 and 
            self.left_hip and 70 <= self.left_hip <= 140 and 
            self.left_shoulder and 140 <= self.left_shoulder <= 180 and
            self.left_knee and 160 <= self.left_knee <= 180)
        
        if self.round:
            if self.uttas_left:
                # print(self.uttas_left)
                self.voice.playAudio(["move to next pose ashwa sanchalasana","extend the left leg back and bend your right leg "],play=True)
                
                return True
            else:
                # print(self.uttas_left)
                self.round = False
                return False
            
        if not self.round:
            if self.parvatasana_left:
                self.voice.playAudio(["move to next pose ashwa sanchalasana","extend the left leg back and bend your right leg "],play=True)
            else:
                self.round = True
                return False

    def start_right_exercise(self,frames):

        #this one check uttasana and do ashwa sanchalasana
        self.uttas_right = (
            self.head_position and self.head_position == "Down" and
             self.min_ground_right <= self.right_finger_y <= self.ground_right and
            ((self.right_foot_z > self.right_finger_z) or (self.right_finger_z < self.right_foot_z < self.left_foot_z < self.left_finger_z)) and
            self.right_elbow and 120 <= self.right_elbow <= 180 and 
            self.right_hip and 15 <= self.right_hip <= 55 and 
            ((self.right_knee and 120 <= self.right_knee <= 150) or (self.right_knee and 151 <= self.right_knee <= 180)) and 
            self.right_shoulder and 50 <= self.right_shoulder <= 120)

        #this one check parvatasana and do ashwa sanchalasana
        self.parvatasana_right = (
            self.right_finger_z < self.head_z and
            self.palm_distance and 350 <= self.palm_distance <= 600 and
            self.right_elbow and 120 <= self.right_elbow <= 180 and 
            self.right_hip and 70 <= self.right_hip <= 140 and 
            self.right_shoulder and 140 <= self.right_shoulder <= 180 and
            self.right_knee and 160 <= self.right_knee <= 180)
        
        if self.round:
            if self.uttas_right:
                self.voice.playAudio(["move to next pose ashwa sanchalasana","extend the right leg back and bend your left leg "],play=True)
                return True
            else:
                self.round = False
                return False
            
        if not self.round:
            if self.parvatasana_right:
                self.voice.playAudio(["move to next pose ashwa sanchalasana","extend the right leg back and bend your left leg "],play=True)
                return True
            else:
                self.round = True
                return False


    def wrong_left(self,frames):

        #to check left leg is correct not
        left_knee_correct = (self.left_knee)
        if left_knee_correct:
            left_knee = (self.left_knee and 0 <= self.left_knee <= 139)
            if left_knee:
                self.voice.playAudio(["your left leg must be in back straight position"],play=True)

        right_knee_down_correct = (self.right_knee1)
        if right_knee_down_correct:
            right_knee_down = (self.right_knee1 and 101 <= self.right_knee1 <= 189)
            if right_knee_down:
                self.voice.playAudio(["please align your right leg between hands"],play=True)
                

        right_knee_up_correct = (self.right_knee1)
        if right_knee_up_correct:
            right_knee_up = (self.right_knee1 and 0 <= self.right_knee1 <= 69)
            if right_knee_up:
                self.voice.playAudio(["please align your right leg between hands"],play=True)


        #this one is for left leg
        right_leg = (self.right_leg)
        if right_leg:
            # right_leg_down = (self.right_leg and 0 <= self.right_leg <= 44)
            # if right_leg_down:
            #     self.voice.playAudio(["please open your right leg"],play=True)

            right_leg_up = (self.right_leg and 16 <= self.right_leg <= 180)
            if right_leg_up:
                self.voice.playAudio(["please align your right leg approximately 90 degrees"],play=True)

        #to check left side hip 
        left_hip_correct = (self.left_hip)
        if left_hip_correct:
            left_hip = (self.left_hip and 0 <= self.left_hip <= 129)
            if left_hip:
                self.voice.playAudio(["your hip is in straight position"],play=True)
                # self.all_methods.voice("your hip is in wrong position please correct from image")


        #shoulder raised up
        left_shoulder_up_correct = (self.left_shoulder)
        if left_shoulder_up_correct:
            left_shoulder_up = ((self.left_shoulder and 0 <= self.left_shoulder <= 49))
            if left_shoulder_up:
                self.voice.playAudio(["please open your shoulders"],play=True)
                # self.all_methods.voice("your shoulders are in down bended position please raise up shown in image")


        #shoulder raised down
        left_shoulder_down_correct = (self.left_shoulder)
        if left_shoulder_down_correct:
            left_shoulder_down = (self.left_shoulder and 86 <= self.left_shoulder <= 180)
            if left_shoulder_down:
                self.voice.playAudio(["please close your shoulders"],play=True)
                # self.all_methods.voice("your shoulders are in up please bend your shoulders down shown in image")


        #to check left elbow
        left_elbow_correct  = (self.left_elbow)
        if left_elbow_correct:
            left_elbow = (self.left_elbow and 0 <= self.left_elbow <= 139)
            if left_elbow:
                self.voice.playAudio(["please be staright your elbow dont bend too much"],play=True)
                # self.all_methods.voice("your elbow is set in wrong position please be staright your elbow dont bend too much")


        # hand to touch the ground or not
        left_ground_touch_correct = (self.left_finger_y)
        if left_ground_touch_correct:
            left_ground_touch = (0 <= self.left_finger_y <= self.min_ground_left)
            if left_ground_touch:
                self.voice.playAudio(["your hand must touch ground"],play=True)
                # self.all_methods.voice("your hand is not touch to ground perfectly please touch ground perfectly and maintain wrist is up on the left heel")


        #left side view to check hands will be before to legs or not
        left_side_view = ((self.left_finger_z > self.right_foot_z) or (self.left_finger_z > self.right_foot_z > self.right_finger_z))
        if left_side_view:
            self.voice.playAudio(["your left hand is always before your left leg and left hand is view to camera first"],play=True)
            # self.all_methods.voice("your hand is always before your left leg and left hand is view to camera first")


        #check head status
        left_head_correct = (self.head)
        if left_head_correct:
            left_head = (self.head and self.head_position != "Left")
            if left_head:
                self.voice.playAudio(["you have to look upside only"],play=True)
                # self.all_methods.voice("you have to look upside only")


        else:
            return 
        
    def wrong_right(self,frames):

        #to check left leg is correct not
        right_knee_correct = (self.right_knee)
        if right_knee_correct:
            right_knee = (self.right_knee and 0 <= self.right_knee <= 139)
            if right_knee:
                self.voice.playAudio(["your right leg must be in back straight position"],play=True)

        left_knee_down_correct = (self.left_knee1)
        if left_knee_down_correct:
            left_knee_down = (self.left_knee1 and 101 <= self.left_knee1 <= 189)
            if left_knee_down:
                self.voice.playAudio(["please align your left leg between hands"],play=True)
                

        left_knee_up_correct = (self.left_knee1)
        if left_knee_up_correct:
            left_knee_up = (self.left_knee1 and 0 <= self.left_knee1 <= 69)
            if left_knee_up:
                self.voice.playAudio(["please align your left leg between hands"],play=True)


        #this one is for left leg
        left_leg = (self.left_leg)
        if left_leg:
            left_leg_down = (self.left_leg and 16 <= self.left_leg <= 180)
            if left_leg_down:
                self.voice.playAudio(["please align your left leg approximately 90 degrees"],play=True)

            # left_leg_up = (self.left_leg and 61 <= self.left_leg <= 180)
            # if left_leg_up:
            #     self.voice.playAudio(["please close your left leg"],play=True)


        #to check right side hip 
        right_hip_correct = (self.right_hip)
        if right_hip_correct:
            right_hip = (self.right_hip and 0 <= self.right_hip <= 129)
            if right_hip:
                self.voice.playAudio(["your hip is in staright position"],play=True)
                # self.all_methods.voice("your hip is in wrong position please correct from image")


        #shoulder raised up
        right_shoulder_up_correct = (self.right_shoulder)
        if right_shoulder_up_correct:
            right_shoulder_up = ((self.right_shoulder and 0 <= self.right_shoulder <= 49))
            if right_shoulder_up:
                self.voice.playAudio(["please open your shoulders"],play=True)
                # self.all_methods.voice("your shoulders are in down bended position please raise up shown in image")


        #shoulder raised down
        right_shoulder_down_correct = (self.right_shoulder)
        if right_shoulder_down_correct:
            right_shoulder_down = (self.right_shoulder and 86 <= self.right_shoulder <= 180)
            if right_shoulder_down:
                self.voice.playAudio(["please close your shoulders"],play=True)
                # self.all_methods.voice("your shoulders are in up please bend your shoulders down shown in image")


        #to check right elbow
        right_elbow_correct  = (self.right_elbow)
        if right_elbow_correct:
            right_elbow = (self.right_elbow and 0 <= self.right_elbow <= 139)
            if right_elbow:
                self.voice.playAudio(["please be staright your elbow dont bend too much"],play=True)
                # self.all_methods.voice("your elbow is set in wrong position please be staright your elbow dont bend too much")


        # hand to touch the ground or not
        right_ground_touch_correct = (self.right_finger_y)
        if right_ground_touch_correct:
            right_ground_touch = (0 <= self.right_finger_y <= self.min_ground_right)
            if right_ground_touch:
                self.voice.playAudio(["your hand must touch ground"],play=True)
                # self.all_methods.voice("your hand is not touch to ground perfectly please touch ground perfectly and maintain wrist is up on the right heel")


        #right side view to check hands will be before to legs or not
        right_side_view = ((self.right_finger_z > self.right_foot_z) or (self.right_finger_z > self.right_foot_z > self.right_finger_z))
        if right_side_view:
            self.voice.playAudio(["your right hand is always before your right leg and right hand is view to camera first"],play=True)
            # self.all_methods.voice("your hand is always before your right leg and right hand is view to camera first")


        #check head status
        right_head_correct = (self.head)
        if right_head_correct:
            right_head = (self.head and self.head_position != "right")
            if right_head:
                self.voice.playAudio(["you have to look upside only"],play=True)
                # self.all_methods.voice("you have to look upside only")

        else:
            return  

    def right_ashwa_name(self,frames):

        if (
            ((self.right_finger_z < self.left_foot_z) or (self.right_finger_z < self.left_foot_z < self.left_finger_z)) and
            self.min_ground_right <= self.right_finger_y <= self.ground_right and
            self.right_elbow and 140 <= self.right_elbow <= 180 and  
            self.right_hip and 130 <= self.right_hip <= 180 and
            self.right_knee and  140 <= self.right_knee <= 180 and
            self.right_shoulder and 50 <= self.right_shoulder <= 85 and
            self.head_position and self.head_position == "Right" and
            self.left_knee1 and 70 <= self.left_knee1 <= 100 and
            self.left_leg and 0 <= self.left_leg <= 20):
                
                cv.putText(frames,"ashwa",(40,300),cv.FONT_HERSHEY_DUPLEX,2,(255,0,255),2)

                return True
        
        return False
    
    def left_ashwa_name(self,frames):

        if (
            self.min_ground_left <= self.left_finger_y <= self.ground_left and
            ((self.left_finger_z < self.right_foot_z) or (self.left_finger_z < self.right_foot_z < self.right_finger_z)) and
            self.left_elbow and 140 <= self.left_elbow <= 180 and
            self.left_hip and 130 <= self.left_hip <= 180 and
            self.left_knee and 140 <= self.left_knee <= 180 and
            self.left_shoulder and 50 <= self.left_shoulder <= 85  and 
            self.head_position and self.head_position == "Left" and
            self.right_knee1 and 70 <= self.right_knee1 <= 100  and
            self.right_leg and 0 <= self.right_leg <= 20):

            cv.putText(frames,"Ashwa",(40,300),cv.FONT_HERSHEY_DUPLEX,2,(255,0,255),2)
            
            return True
        
        return False
    
    def head_value(self,frames,llist,point):

        if len(llist) != 0:
            self.head_val = self.head_point(frames=frames,lmlist=llist,points=point)

            if self.head_val :
                self.head_x = self.head_val[0]
                self.head_y = self.head_val[1]

                cv.circle(frames,(self.head_x,self.head_y),5,(255,0,0),2)

                return self.head_x
            

            return False
        return False

 
def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) #Height
    global llist
    all_methods = allmethods()
    detect = ashwa()
    voice = VoicePlay()

    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        # if not isTrue:
        #     print("Error: Couldn't read the frame")
        #     break                                                                                  #(23,25,27)
        img = cv.imread("images/image4.jpg")
        detect.pose_positions(frames,draw = False)
        llist = detect.pose_landmarks(frames,draw=False)
        head_point = detect.head_value(frames,llist=llist,point=(0,2,5))
        detect.hands_legs_correct_position(frames,llist=llist,points=(16,15,32,31))

        side_view = all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=detect.HEAD_POSITION,head=detect.head_x)
        standing_side_detect = all_methods.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)

        
        if len(llist) != 0:

            if standing_side_detect == "forward":

                voice.playAudio(["Bend forward from your hips, keeping your back straight and your knees slightly bent"],play=True)

                if side_view == "right":
                    
                    detect.right_ashwa(frames,llist=llist,elbow=(12,14,16), hip=(12,24,26), knee=(24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),draw=False)
                    detect.left_leg_slope(frames=frames,llist=llist,left_hip=23,left_knee=25,height=detect.h,width=detect.w)
                    detect.wrong_right(frames=frames)
                    correct_one = detect.right_ashwa_name(frames)
                    

                elif side_view == "left":
                    
                    detect.left_ashwa(frames,llist=llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) ,shoulder= (13,11,23),right_knee=(24,26,28),draw=False)
                    detect.right_leg_slope(frames=frames,llist=llist,right_hip=24,right_knee=26,height=detect.h,width=detect.w)
                    detect.wrong_left(frames=frames)
                    correct_one = detect.left_ashwa_name(frames)
                
            # elif side_view == "left side cross position":
            #     voice.playAudio(["turn total left side"],play=True)

            # elif side_view == "right side cross position":
            #     voice.playAudio(["turn total right side"],play=True)

            # else:
            #     voice.playAudio(["turn left side or right side and do ashwa sanchalanasana"],play=True)

        cv.imshow("video",frames)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()

main()
