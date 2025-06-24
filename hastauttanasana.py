import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math

from threading import Thread
from allMethods import allmethods
from voiceModule import VoicePlay
from face_detect import HeadPoseEstimator

class hastauttanasana:
     
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
                px,py,pz = (poselms.x * self.w) , (poselms.y * self.h),(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lpslist 
    

    def right_hastauttanasana(self,frames,llist,elbow, hip, knee, shoulder, draw = True):

        # self.voice.playAudio("move to next exercise")

        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 
        
        self.ground_right = self.all_methods.ground_distance_right(frames=frames,lmlist=llist)
        self.min_ground_right = self.ground_right-30

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

        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder,self.head_position
       
    def left_hastauttanasana(self,frames,llist,elbow, hip, knee,shoulder,draw = True):


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


    def head_heel_points(self,frames,llist,face_points,left_heel_point,right_heel_point,draw = True):

        self.head_points = self.head_detect(frames=frames,lmlist=llist,points=face_points) 

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
    
    def check_stand(self,frames,llist,height,width):

        
        self.view_position = self.all_methods.is_person_standing_sitting(frames=frames,llist=llist,hip_points=(11,23,25),leg_points=(23,25,27),elbow_points=(11,13,15),height=height,width=width)

        # return self.view_position

        if self.view_position == "standing" :#or self.view_position == "standing":

            # self.voice.playAudio(["you can start your yoga","stand in a left side or right side position and do pranamasana"],play=True)
            return True
        
        elif self.view_position != "standing" :#or self.view_position != "standing":
            self.voice.playAudio(["you must be in standing position"],play=True)

    
    def start_left_exercise(self,frames):

        #this one check pranamasana and do hasta uttanasana

        self.pranam_left = (
            self.left_shoulder and 0 <= self.left_shoulder <= 25 and
            self.left_elbow and 40 <= self.left_elbow <= 90 and
            self.left_hip and 160 <= self.left_hip <= 180 and
            self.left_knee and 170 <= self.left_knee <= 180 and
            self.head_position and self.head_position == "Left")
        
        #this one check uttanasana and do hasta uttanasana
        self.uttas_left = (
            self.head_position and self.head_position == "Down" and
             self.min_ground_left <= self.left_finger_y <= self.ground_left  and
            ((self.left_finger_z < self.left_foot_z) or (self.left_finger_z < self.left_foot_z < self.right_foot_z < self.right_finger_z)) and
            self.left_elbow and 90 <= self.left_elbow <= 180 and 
            self.left_hip and 15 <= self.left_hip <= 65 and 
            ((self.left_knee and 120 <= self.left_knee <= 150) or (self.left_knee and 151 <= self.left_knee <= 180))and 
            self.left_shoulder and 50 <= self.left_shoulder <= 120)

        if self.round:
            
            if self.pranam_left :
                # print(self.hasta_left)
                
                self.voice.playAudio(["move to next pose Hasta Uttanasana","Gently bend back at the waist, "],play=True)
                
                return True

            else:
                self.round = False
                return False
        elif not self.round:
            if self.uttas_left:
                
                self.voice.playAudio(["move to next pose Hasta Uttanasana","Gently bend back at the waist"],play=True)
                
                return True
        
            else:
                self.round = True
                return False
        # return False
    
    def start_right_exercise(self,frames):
        
        
        #this one check pranamasana and do hasta uttanasana
        self.pranam_right = (
            self.right_elbow and 40 <= self.right_elbow <= 90 and          
            self.right_hip and 160 <= self.right_hip <= 180 and            
            self.right_knee and 170 <= self.right_knee <= 180 and                                      
            self.right_shoulder and 0 <= self.right_shoulder <= 25 and
            self.head_position and self.head_position == "Right")
        

        #this one check uttanasana and do hasta uttanasana
        self.uttas_right = (
            self.head_position and self.head_position == "Down" and
             self.min_ground_right <= self.right_finger_y <= self.ground_right and
            ((self.right_foot_z > self.right_finger_z) or (self.right_finger_z < self.right_foot_z < self.left_foot_z < self.left_finger_z)) and
            self.right_elbow and 90 <= self.right_elbow <= 180 and 
            self.right_hip and 15 <= self.right_hip <= 65 and 
            (self.right_knee and 120 <= self.right_knee <= 150) or (self.right_knee and 151 <= self.right_knee <= 180) and 
            self.right_shoulder and 50 <= self.right_shoulder <= 120)
        
        if self.round:
            
            if self.pranam_right :
                
                self.voice.playAudio(["move to next pose Hasta uttanasana","Gently bend back at the waist"],play=True)
                return True

            else:
                self.round = False
                return False
    
        elif not self.round:
            if self.uttas_right:
                
                self.voice.playAudio(["move to next pose Hasta uttanasana","Gently bend back at the waist"],play=True)
                
                return True
        
            else:
                self.round = True
                return False


    def wrong_left(self,frames):

        #this one is for hip back side
        left_hip = (self.left_hip)
        if left_hip:
            left_hip_back = (self.left_hip and 161 <= self.left_hip <= 180)
            if left_hip_back:
                self.voice.playAudio(["bend back side"],play=True)
                # self.all_methods.voice("your hip not bend pefect move your hip to front and bend back side")

            #this one is for hip front side
            left_hip_front = (self.left_hip and 0 <= self.left_hip <= 119)
            if left_hip_front:
                self.voice.playAudio(["bend front side"],play=True)
                # self.all_methods.voice("your hip is bend in fron position so you are in bad position so bend your body back side")

        
        #this one is for left knee
        left_knee_correct = (self.left_knee)
        if left_knee_correct:
            left_knee = (self.left_knee and 0 <= self.left_knee <= 149)
            if left_knee:
                self.voice.playAudio(["please bend legs slight"],play=True)
                # self.all_methods.voice("your left leg is not in correct position please be straight")

            # left_knee_straight = (self.left_knee and 171 <= self.left_knee <= 180)
            # if left_knee_straight:
            #     self.voice.playAudio(["please bend your legs front side"],play=True)
                # self.all_methods.voice("you legs are in straight position please bend your legs front side")

        
        #left elbow
        left_elbow_correct = (self.left_elbow)
        if left_elbow_correct:
            left_elbow = (self.left_elbow and 0 <= self.left_elbow <= 159)
            if left_elbow:
                self.voice.playAudio(["keep your elbows in straight"],play=True)

        #this one is for left shoulder
        left_shoulder_correct = (self.left_shoulder)
        if left_shoulder_correct:
            left_shoulder = (self.left_shoulder and 0<= self.left_shoulder <= 159)
            if left_shoulder:
                self.voice.playAudio(["bend back side your shoulders"],play=True)
                # self.all_methods.voice("your left shoulder are in bad position please bend back side your sleft shoulder")


        #this one is for head
        head_correct = (self.head_x and self.left_heel_x)
        if head_correct:
            left = (self.head_x < self.left_heel_x)
            if left:
                self.voice.playAudio(["bend back side your head"],play=True)
                # self.all_methods.voice("your body is in left position but your head is not cross the your heel so do the head to cross your heel back side")

        else:
            return 
    

    def wrong_right(self,frames):

        #this one is for hip back side
        right_hip = (self.right_hip)
        if right_hip:
            right_hip_back = (self.right_hip and 161 <= self.right_hip <= 180)
            if right_hip_back:
                self.voice.playAudio(["bend back side"],play=True)
                # self.all_methods.voice("your hip not bend pefect move your hip to front and bend back side")

            #this one is for hip front side
            right_hip_front = (self.right_hip and 0 <= self.right_hip <= 119)
            if right_hip_front:
                self.voice.playAudio(["bend front side"],play=True)
                # self.all_methods.voice("your hip is bend in bad position so you are in bad position so bend your body back side")

        #this one is for right knee
        right_knee_correct = (self.right_knee)
        if right_knee_correct:
            # right_knee_straight = (self.right_knee and 171 <= self.right_knee <= 180)
            # if right_knee_straight:
            #     self.voice.playAudio(["please bend legs front slight"],play=True)
                # self.all_methods.voice("you legs are in stright position please bend your legs front side")

            
            #this one is for right knee
            right_knee =(self.right_knee and 0 <= self.right_knee <= 149)
            if right_knee:
                self.voice.playAudio(["please bend legs back slight"],play=True)
                # self.all_methods.voice("your right leg is not in correct position please be straight")


        #left elbow
        right_elbow_correct = (self.right_elbow)
        if right_elbow_correct:
        #right elbow
            right_elbow = (self.right_elbow and 0 <= self.right_elbow <= 159)
            if right_elbow:
                self.voice.playAudio(["keep your elbows in straight"],play=True)
                # self.all_methods.voice("your elbow is in bad position please bend back side staright your right hand")


        #this one is for right shoulder
        right_shoulder_correct = (self.right_shoulder)
        if right_shoulder_correct:
            right_shoulder = (self.right_shoulder and 0<= self.right_shoulder <= 159)
            if right_shoulder:
                self.voice.playAudio(["bend back side your shoulders"],play=True)
                # self.all_methods.voice("your right shoulder are in bad position please bend back side your sright shoulder")


        #this one is for head
        head_correct = (self.head_x and self.right_heel_x)
        if head_correct:
            right = (self.head_x < self.right_heel_x)
            if right:
                self.voice.playAudio(["bend back side your head"],play=True)

        else:
            return

    def right_hastauttanasana_name(self,frames):  

        if not self.head_x or not self.right_elbow or not self.right_hip or not  self.right_knee or not self.right_shoulder:
            return
       
        if (
            self.head_x > self.right_heel_x and 
            self.right_elbow  and 160 <= self.right_elbow <= 185 and 
            self.right_hip and 130 <= self.right_hip <= 160 and  
            self.right_knee and 150 <= self.right_knee <= 180 and
            self.right_shoulder and 160 <= self.right_shoulder <= 170):# and
            
                # self.voice.playAudio("Forward Bend, is a yoga pose that involves folding forward from a hasta uttanasana position, with legs straight and hands on the ground or")
                cv.putText(frames,str("Hasta Uttanasana"),(80,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                self.voice.playAudio(["you done your yoga pose perfect"],play=True)
                
                return True
        
        return False  
    

    def left_hastauttanasana_name(self,frames):

        if not self.head_x or not self.left_elbow or not self.left_hip or not  self.left_knee or not self.left_shoulder:
            return

        if (
            self.head_x < self.left_heel_x and
            self.left_elbow and 160 <= self.left_elbow <= 180 and
            self.left_hip and 130 <= self.left_hip <= 160 and
            self.left_knee and 150 <= self.left_knee <= 180 and 
            self.left_shoulder and 160 <= self.left_shoulder <= 170):

                cv.putText(frames,str("hasta Uttanasana"),(80,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                self.voice.playAudio(["you done your yoga pose perfect"],play=True)
                
                return True
        
        return False  


    def head_value(self,frames,llist,point):

        if len(llist) != 0:
            self.head_val = self.head_detect(frames=frames,lmlist=llist,points=point)
            

            if self.head_val :
                self.head_x = self.head_val[0]
                self.head_y = self.head_val[1]

                cv.circle(frames,(self.head_x,self.head_y),5,(255,0,0),2)

                return self.head_x

            return False
        return False
    
    def heel_value(self,frames,llist,heel_point):

        if len(llist) != 0:

            self.heel_val = self.heel_detect(frames=frames,lmlist=llist,point=heel_point)

            self.left_heel_x = self.heel_val[0]

            return self.left_heel_x
        return False
    

   
def main():
    global llist
    all_methods = allmethods()
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980)
    detect = hastauttanasana()
    voice = VoicePlay()
    all_methods = allmethods()
    voice_detect = False
    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        img = cv.imread("images/image2.jpg")
        resized_img = cv.resize(img, None, fx=1.0, fy=1.0, interpolation=cv.INTER_LINEAR)
        detect.pose_positions(frames,draw = False)
        llist = detect.pose_landmarks(frames,draw=False)
        detect.head_heel_points(frames=frames,llist=llist,face_points=(0,2,5),left_heel_point=29,right_heel_point=30)
        side_view = all_methods.standing_side_view_detect(frames,llist=llist,height=height,width=width)
        
        if len(llist) != 0:

            
            if side_view ==  "right":

                detect.right_hastauttanasana(frames=frames,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),draw=False)
                stand_position = detect.check_stand(frames=frames,llist=llist,height=height,width=width)
                if stand_position:
                    wrong_right = detect.wrong_right(frames=frames)
                correct = detect.right_hastauttanasana_name(frames)

            elif side_view == "left":

                detect.left_hastauttanasana(frames=frames,llist=llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),draw=False)
                stand_position = detect.check_stand(frames=frames,llist=llist,height=height,width=width)
                if stand_position:
                    wrong_left = detect.wrong_left(frames=frames)
                correct = detect.left_hastauttanasana_name(frames)

            elif side_view == "left side cross position":
                voice.playAudio(["turn total left side"],play=True)

            elif side_view == "right side cross position":
                voice.playAudio(["turn total right side"],play=True)

            else:
                voice.playAudio(["turn left side or right side and do hasta uttanasana"],play=True)

        cv.imshow("video",frames)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()

main()

