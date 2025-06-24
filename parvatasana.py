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
from body_position import bodyPosition
# from hastauttanasana import hastauttanasana

class parvatasana:
     
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
        self.palm_distance = 0
        self.head_point = self.all_methods.head_detect
        self.voice = VoicePlay()
        self.head_pose = HeadPoseEstimator()

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
                    cv.circle(frames,(px,py,pz),5,(255,255,0),2)

        return self.lpslist  
    

    def parvatasana_right(self,frames,llist,elbow,hip,knee,shoulder,left_knee,left_shoulder,left_elbow, draw = True):

        # self.voice.playAudio(["do parvatasana"],play=True)

        self.ground_right = self.all_methods.ground_distance_right(frames=frames,lmlist=llist)

        self.right_elbow,points_cor1 = self.all_methods.calculate_angle(frames=frames,points= elbow, lmList=llist)
        self.right_hip ,points_cor2= self.all_methods.calculate_angle(frames=frames,points=hip, lmList=llist)
        self.right_knee,points_cor3 = self.all_methods.calculate_angle(frames=frames,points=knee, lmList=llist)
        self.right_shoulder,points_cor4 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        self.left_knee1,points_cor9 = self.all_methods.calculate_angle(frames=frames,points=left_knee, lmList=llist)
        self.left_shoulder1,self.points_cor11 = self.all_methods.calculate_angle(frames=frames,points=left_shoulder, lmList=llist)
        self.left_elbow1,self.points_cor13 = self.all_methods.calculate_angle(frames=frames,points=left_elbow, lmList=llist)
        
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

    def parvatasana_left(self,frames,llist,elbow,hip,knee,shoulder,right_knee,right_shoulder,right_elbow,draw=True):

        self.ground_left = self.all_methods.ground_distance_left(frames=frames,lmlist=llist)

        self.left_elbow,self.points_cor5 = self.all_methods.calculate_angle(frames=frames,points=elbow, lmList=llist)
        self.left_hip,self.points_cor6 = self.all_methods.calculate_angle(frames=frames,points=hip, lmList=llist)
        self.left_knee,self.points_cor7 = self.all_methods.calculate_angle(frames=frames,points=knee, lmList=llist)
        self.left_shoulder,self.points_cor8 = self.all_methods.calculate_angle(frames=frames,points=shoulder, lmList=llist)
        # if self.right_knee1:
        self.right_knee1,points_cor10 = self.all_methods.calculate_angle(frames=frames,points=right_knee, lmList=llist)
        self.right_shoulder1,self.points_cor12 = self.all_methods.calculate_angle(frames=frames,points=right_shoulder, lmList=llist)
        self.right_elbow1,self.points_cor14 = self.all_methods.calculate_angle(frames=frames,points=right_elbow, lmList=llist)

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
        
    def hands_legs_correct_position(self,frames,llist,points):

        self.head = self.head_pose.head_detect(frames=frames,llist=llist,points=(0,2,5))
        self.left_ear = self.head_pose.left_ear_detect(frames=frames,llist=llist,left_point=7)
        self.right_ear = self.head_pose.right_ear_detect(frames=frames,llist=llist,right_point=8)
        self.head_position = self.head_pose.ear_name(frames=frames,llist=llist)
        
        if self.head_position is None or self.left_ear is None or self.right_ear is None or self.head is None:
            return 

        if len(llist) != 0:

            self.right_finger_x,self.right_finger_y,self.right_finger_z = llist[points[0]][1:]
            # print("right_finger",self.right_finger_y)
            self.left_finger_x,self.left_finger_y,self.left_finger_z= llist[points[1]][1:]
            # cv.putText(frames,str(self.left_finger_z),(180,250),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
            self.right_foot_x,self.right_foot_y,self.right_foot_z = llist[points[2]][1:]
            self.left_foot_x , self.left_foot_y,self.left_foot_z = llist[points[3]][1:]
            
            return self.right_finger_x,self.right_finger_y,self.left_finger_x,self.left_finger_y,self.left_foot_x,self.left_foot_y,self.right_foot_x,self.right_foot_y,self.right_finger_z,self.left_finger_z,self.right_foot_z,self.left_foot_z
        
        return False
    
    def start_left_exercise(self,frames):

        #this one check bhujangasana and do parvatasana
        self.bhujang_left = (
            self.left_elbow and 140 <= self.left_elbow <= 180 and 
            self.left_knee and 140 <= self.left_knee <= 180 and
            self.left_hip and 130 <= self.left_hip <= 180 and
            self.left_shoulder and 15 <= self.left_shoulder <= 60  and
            self.head_position and self.head_position == "Up")
        
        if self.bhujang_left:
            self.voice.playAudio(["move to next pose parvatasana bend your hip and elbows and legs"],play=True)
            return True
        else:
            return False
        
    
    def start_right_exercise(self,frames):

        #this one check bhujangasana and do parvatasana
        self.bhujang_right = (
            self.right_elbow and 140 <= self.right_elbow <= 180 and 
            self.right_knee and 140 <= self.right_knee <= 180 and
            self.right_hip and 130 <= self.right_hip <= 180 and
            self.right_shoulder and 15 <= self.right_shoulder <= 60 and
            self.head_position and self.head_position == "Up")
        
        if self.bhujang_right:
            self.voice.playAudio(["move to next pose parvatasana bend your hip and elbows and legs "],play=True)
            return True
        else:
            return False
        
        
    def wrong_left(self,frames):

        #head is between hands or not
        left_hand_correct = (self.head_z and self.left_finger_z)
        if left_hand_correct:
            left_hand = (self.left_finger_z > self.head_z)
            if left_hand:
                self.voice.playAudio(["your head is between your two elbows"],play=True)
                # self.all_methods.voice("your head is between your two elbows")
        
        
        palm_distance = (self.palm_distance)
        if palm_distance:
             #palm distance is less to increase
            palm_distance_short = (self.palm_distance and 0<= self.palm_distance <= 449)
            if palm_distance_short:
                self.voice.playAudio(["keep maintain a distance between hands and legs"],play=True)
                # self.all_methods.voice("stretch your hands and legs keep maintain a distance between hands and legs")

            #palm distace is more to decrease 
            palm_distance_more = (self.palm_distance and 601<= self.palm_distance <= 1000)
            if palm_distance_more:
                self.voice.playAudio(["close the distance of hands and legs"],play=True)
                # self.all_methods.voice("your are stretching your hands and legs too much just decrease distance between them")


        #check left elbow
        left_elbow_correct = (self.left_elbow)
        if left_elbow_correct:
            left_elbow = (self.left_elbow and 0 <= self.left_elbow <= 139)
            if left_elbow:
                self.voice.playAudio(["your elbow must be in staright"],play=True)
                # self.all_methods.voice("your elbow must be in staright")


        #check if hip is in down position
        left_hip = (self.left_hip)
        if left_hip:
            left_hip_up = (self.left_hip and 0 <= self.left_hip <= 79)
            if left_hip_up:
                self.voice.playAudio(["please raise up your hip slightly"],play=True)
                # self.all_methods.voice("you are bending too much you use to raise your hip slightly til you reach a correct position")

            #check if hip is in up position
            left_hip_down = (self.left_hip and 141 <= self.left_hip <= 180)
            if left_hip_down:
                self.voice.playAudio(["please bend your hip slightly down "],play=True)
                # self.all_methods.voice("you use to bend your hip slightly down until you reach a correct position")


        #check shoulder it is up or down
        left_shoulder_correct = (self.left_shoulder)
        if left_shoulder_correct:
            left_shoulder = (self.left_shoulder and 0 <= self.left_shoulder <= 149)
            if left_shoulder:
                self.voice.playAudio(["your shoulders must be in straight"],play=True)
                # self.all_methods.voice("your shoulders mustbe in straight")


        #this one is to check left knee 
        left_knee_correct = (self.left_knee)
        if left_knee_correct:
            left_knee = (self.left_knee and 0 <= self.left_knee <= 159)
            if left_knee:
                self.voice.playAudio(["your legs must be in straight position"],play=True)
                # self.all_methods.voice("your legs must be in straight position")

        #this one is to check right knee 
        right_knee_correct = (self.right_knee1)
        if right_knee_correct:
            right_knee = (0 <= self.right_knee1 <= 159)
            if right_knee:
                self.voice.playAudio(["your legs must be in straight position"],play=True)
                # self.all_methods.voice("your legs must be in straight position")

        else:
            return 
        
    def wrong_right(self,frames):

        #head is between hands or not
        right_hand_correct = (self.head_z and self.right_finger_z)
        if right_hand_correct:
            right_hand = (self.right_finger_z > self.head_z)
            if right_hand:
                self.voice.playAudio(["your head is between your two elbows"],play=True)
                # self.all_methods.voice("your head is between your two elbows")


        palm_distance = (self.palm_distance)
        if palm_distance:
             #palm distance is less to increase
            palm_distance_short = (0<= self.palm_distance <= 449)
            if palm_distance_short:
                self.voice.playAudio(["keep maintain a distance between hands and legs"],play=True)
                # self.all_methods.voice("stretch your hands and legs keep maintain a distance between hands and legs")

            #palm distace is more to decrease 
            palm_distance_more = (601<= self.palm_distance <= 1000)
            if palm_distance_more:
                self.voice.playAudio(["close the distance of hands and legs"],play=True)
                # self.all_methods.voice("your are stretching your hands and legs too much just decrease distance between them")


        #check right elbow
        right_elbow_correct = (self.right_elbow)
        if right_elbow_correct:
            right_elbow = (0 <= self.right_elbow <= 139)
            if right_elbow:
                self.voice.playAudio(["your elbow must be in staright"],play=True)
                # self.all_methods.voice("your elbows must be in staright")


        #check if hip is in down position
        right_hip = (self.right_hip)
        if right_hip:
            right_hip_up = (0 <= self.right_hip <= 79)
            if right_hip_up:
                self.voice.playAudio(["please raise up your hip slightly"],play=True)
                # self.all_methods.voice("you are bending too much you use to raise your hip slightly til you reach a correct position")

            #check if hip is in up position
            right_hip_down = (141 <= self.right_hip <= 180)
            if right_hip_down:
                self.voice.playAudio(["please bend your hip slightly down "],play=True)
                # self.all_methods.voice("you use to bend your hip slightly down until you reach a correct position")


        #check shoulder it is up or down
        right_shoulder_correct = (self.right_shoulder)
        if right_shoulder_correct:
            right_shoulder = (0 <= self.right_shoulder <= 149)
            if right_shoulder:
                self.voice.playAudio(["your shoulders must be in straight"],play=True)
                # self.all_methods.voice("your shoulders mustbe in straight")


        #this one is to check right knee 
        right_knee_correct = (self.right_knee)
        if right_knee_correct:
            right_knee = (0 <= self.right_knee <= 159)
            if right_knee:
                self.voice.playAudio(["your legs must be in straight position"],play=True)
                # self.all_methods.voice("your legs must be in straight position")

        #this one is to check right knee 
        left_knee_correct = (self.left_knee1)
        if left_knee_correct:
            left_knee = (0 <= self.left_knee1 <= 159)
            if left_knee:
                self.voice.playAudio(["your legs must be in straight position"],play=True)
                # self.all_methods.voice("your legs must be in straight position")

        else:
            return

    def left_parvatasana_name(self,frames,llist):

        if (
            self.left_finger_z < self.head_z and
            self.palm_distance and 450 <= self.palm_distance <= 600 and
            self.left_elbow and 140 <= self.left_elbow <= 180 and 
            self.left_hip and 80 <= self.left_hip <= 140 and 
            self.left_shoulder and 150 <= self.left_shoulder <= 170 and
            self.left_knee and 160 <= self.left_knee <= 180 and
            self.right_knee1 and 160 <= self.right_knee1 <= 180):
            cv.putText(frames,str("Parvatasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
            cv.putText(frames,str("left_side"),(140,150),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                                                            
            return True
        return False
        
    def right_parvatasana_name(self,frames,llist):
      
        if (
            self.right_finger_z < self.head_z and
            self.palm_distance and 450 <= self.palm_distance <= 600 and
            self.right_elbow and 140 <= self.right_elbow <= 180 and 
            self.right_hip and 80 <= self.right_hip <= 140 and 
            self.right_shoulder and 150 <= self.right_shoulder <= 170 and
            self.right_knee and 160 <= self.right_knee <= 180 and
            self.left_knee1 and 160 <= self.left_knee1 <= 180):
            cv.putText(frames,str("parvatasana"),(40,50),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

            return True
        return False
    
    #to create a head_point for side view
    def head_value(self,frames,point):

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
    
    global llist
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)  #set Height
    detect = parvatasana()
    all_methods = allmethods()
    voice=VoicePlay()
    body_position = bodyPosition()
    while True:

        isTrue,frames = video_capture.read()
        height, width, _ =  frames.shape
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        img = cv.imread("images/image5.webp")
        detect.pose_positions(frames,draw = False)
        llist = detect.pose_landmarks(frames,draw=False)

        head_point = detect.head_value(frames,point=(0,2,5))
        side_view = all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=detect.HEAD_POSITION,head=head_point)
        detect.hands_legs_correct_position(frames=frames,llist=llist,points=(16,15,32,31))
        side_view = all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=detect.HEAD_POSITION,head=detect.head_x)
        standing_side_detect = all_methods.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)
        standing_sitting_position = all_methods.is_person_standing_sitting(frames=frames,llist=llist,hip_points=(11,23,25),leg_points=(23,25,26),elbow_points=(11,13,15),height=detect.h,width=detect.w)

        if len(llist) != 0:

            if standing_sitting_position == "sitting":

                if standing_side_detect == "forward":
                    voice.playAudio(["please turn your total body left side or right side"],play=True)

                elif standing_side_detect == "right":

                    if side_view =="right":

                        voice.playAudio(["you have to move table pose","then touch your toe on ground slightly lift your heel up"," then lift your knees up lift your spine up"],play=True)
                        
                        detect.parvatasana_right(frames,llist=llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),left_shoulder=(15,11,23),left_elbow=(11,13,15),draw=False)
                        detect.right_distance(frames,llist=llist,right_hand=16,right_toe_point=32,face_points=(0,2,5))
                        
                        detect.wrong_right(frames=frames)
                        correct = detect.right_parvatasana_name(frames,llist=llist)
                        cv.putText(frames,str("right_side"),(70,80),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

                elif standing_side_detect == "left":
                    if side_view =="left":
                        voice.playAudio(["you have to move table pose","then touch your toe on ground slightly lift your heel up"," then lift your knees up lift your spine up"],play=True)
                        detect.parvatasana_left(frames,llist=llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),right_knee=(24,26,28),right_shoulder=(16,12,24),right_elbow=(12,14,16),draw=False)
                        detect.left_distance(frames,llist=llist,left_hand=15,left_toe_point=31,face_points=(0,2,5))
                        
                        detect.wrong_left(frames=frames)
                        correct = detect.left_parvatasana_name(frames,llist=llist)
                        cv.putText(frames,str("left_side"),(70,80),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)


            else:
                voice.playAudio(["please be in sitting position"],play=True)

        cv.imshow("video",frames)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()