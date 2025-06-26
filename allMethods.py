import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math

import pyttsx3 as pyt
from threading import Thread
import threading
from voiceModule import VoicePlay
# from face_detect import HeadPoseEstimator

# from body_position import bodyPosition


class allmethods:
    
    def __init__(self,mode = False,mindetectconf=0.5,mintrcackconf=0.5):
        # c += 1
        # print(c)
        # print('again......................................................................')
        self.results = None
        self.mode = mode
        self.mindetectconf = mindetectconf
        self.mintrcackconf = mintrcackconf

        self.counter = 0
        self.after_40 = 0

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
        self.voice_play_counter = 0


        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        #shoulder hip
        self.MAX_BACK_LIMT = 45
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
    

    #find head values based on nose,eyes
    def head_detect(self,frames,lmlist,points):

        if not points or points is None:
            return

        if len(lmlist) == 0:
            return 

        nose_x,nose_y,nose_z = lmlist[points[0]][1:]
        right_eye_x,right_eye_y,right_eye_z = lmlist[points[1]][1:]
        left_eye_x,left_eye_y,left_eye_z = lmlist[points[2]][1:]

        self.head_x = int(nose_x+right_eye_x+left_eye_x) // 3
        self.head_y = int(nose_y+right_eye_y+left_eye_y) // 3
        self.head_z = int(nose_z+right_eye_z+left_eye_z) // 3

        cv.circle(frames, (self.head_x,self.head_y), 10, (0, 0, 255), 1) 

        return (self.head_x,self.head_y,self.head_z)


    #heel detect left leg
    def left_heel_detect(self,frames,lmlist,point):

        if len(lmlist) == 0:
            return 

        self.left_heel_x , self.left_heel_y ,self.left_heel_z= lmlist[point][1:]
        # cv.circle(frames, (self.left_heel_x, self.left_heel_y), 10, (0, 0, 255), -1)  

        return (self.left_heel_x,self.left_heel_y,self.left_heel_z)
    

    #heel detect right leg
    def right_heel_detect(self,frames,lmlist,point):

        if lmlist is None:
            return

        if len(lmlist) == 0:
            return None

        self.right_heel_x , self.right_heel_y ,self.right_heel_z= lmlist[point][1:]
        # cv.circle(frames, (self.right_heel_x, self.right_heel_y), 10, (0, 0, 255), -1)  

        return (self.right_heel_x, self.right_heel_y,self.right_heel_z)


    #calculate the angle
    def calculate_angle(self, frames, lmList, points, draw = True):

        if not lmList or len(lmList) < max(points):
            return None, None
    
        self.x1, self.y1,self.z1 = lmList[points[0]][1:]  
        self.x2, self.y2,self.z2 = lmList[points[1]][1:]
        self.x3, self.y3,self.z3 = lmList[points[2]][1:]

        if draw:
            cv.line(frames, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), (0, 255, 0), 3)
            cv.line(frames, (int(self.x3), int(self.y3)), (int(self.x2), int(self.y2)), (0, 255, 0), 3)

        self.angle = abs(math.degrees(math.atan2(self.y3 - self.y2, self.x3 - self.x2) - math.atan2(self.y1 - self.y2, self.x1 - self.x2)))
        return self.angle if self.angle <= 180 else 360 - self.angle, (self.x1, self.y1, self.x2, self.y2, self.x3, self.y3)


    #ground_left for left heel
    def ground_distance_left(self,frames,lmlist):

        if len(lmlist) == 0:
            return None

        if len(lmlist) != 0:

            left_heel_x,left_heel_y,left_heel_z = lmlist[29][1:]
            right_heel_x,right_heel_y,right_heel_z = lmlist[30][1:]
            right_toe_x,right_toe_y,right_toe_z = lmlist[32][1:]
            left_toe_x,left_toe_y,left_toe_z = lmlist[31][1:]

            max_ground_distance = max(left_heel_y,right_heel_y,right_toe_y,left_toe_y)
            # print("Left_side",max_ground_distance)

            return max_ground_distance
        return False
    

    #ground_right for right heel
    def ground_distance_right(self,frames,lmlist):

        if len(lmlist) == 0:
            return None

        if len(lmlist) != 0:

            left_heel_x,left_heel_y,left_heel_z = lmlist[29][1:]
            right_heel_x,right_heel_y,right_heel_z = lmlist[30][1:]
            right_toe_x,right_toe_y,right_toe_z = lmlist[32][1:]
            left_toe_x,left_toe_y,left_toe_z = lmlist[31][1:]

            max_ground_distance = max(right_heel_y,left_heel_y,left_toe_y,right_toe_y)
            # print("right_side",max_ground_distance)

            return max_ground_distance
        return False


    #side view based on frame
    def findSideView(self, frame, FLAG_HEAD_OR_TAIL_POSITION,head):

        if FLAG_HEAD_OR_TAIL_POSITION == self.HEAD_POSITION:
            self.RIGHT_SIDE_VIEW = "right"
            self.LEFT_SIDE_VIEW = "left"
        else:
            self.RIGHT_SIDE_VIEW = "left"
            self.LEFT_SIDE_VIEW = "right"
    
        # Get the image width and height
        h, w, _ = frame.shape
        # print(w//2)

        # Initialize side view variable
        side_view = None
        
        try:

            # Get the x-coordinate of the nose (relative to the image width)
            head_x = head

            # Determine the side view based on the x-coordinate of the nose
            if head_x < w // 2:
                side_view = self.LEFT_SIDE_VIEW
            elif head_x > w //2:
                side_view = self.RIGHT_SIDE_VIEW

            return side_view

        except Exception:
            # Handle case where the nose is not in the expected list (should not happen with proper model)
            return None
    

    #slope create 
    def slope(self,frames,lmlist,point1,point2,height,width,draw):

        if height is None or width is None:
            return None

        if len(lmlist) == 0:

            return 
        
        if point1 and point2 is None:
            return None

        if len(lmlist) != 0:
            if point1 and point2 is not None:

                y2,y1 = lmlist[point2][2] , lmlist[point1][2]
                x2,x1 = lmlist[point2][1] , lmlist[point1][1]

                y = (y2 / height) - (y1/height)
                x = (x2 / width)  - (x1/width)

                if draw:
                    cv.line(frames,(int(x2),int(y2)),(int(x1),int(y1)),(255,0,0),2)
                    # cv.putText(frames,f'shoulder_hip=>{str(abs(int(math.degrees(angle))))}',(40,90),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

                m=y/x

                angle = np.arctan(m)

                # cv.putText(frames,f'shoulder_hip=>{str(abs(int(math.degrees(angle))))}',(40,190),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                return abs(int(math.degrees(angle)))
        else:
            return None

    def is_person_standing_sitting(self,frames,llist,hip_points,leg_points,elbow_points,height,width):

        if height is None or width is None:
            return None

        if llist is None or len(llist) < 31:
            return

        head_x,head_y,head_z = llist[0][1:]
        left_heel_y = llist[29][2]
        right_heel_y = llist[30][2]
        left_knee_y = llist[25][2]
        right_knee_y = llist[26][2]
        leg_y = min(left_knee_y,right_knee_y)
        foot_y = min(left_heel_y,right_heel_y)

        if head_x is None and head_y is None and head_z is None and left_heel_y is None and right_heel_y :
            return None

        #left slope condition
        self.left_shoulder_hip = self.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width,draw=False)
        self.left_hip_knee = self.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=False)

        # # right slope condition
        self.right_shoulder_hip = self.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=False)
        self.right_hip_knee = self.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=False)

        #ground_left
        ground_left = self.ground_distance_left(frames=frames,lmlist=llist)
        #ground_right
        ground_right = self.ground_distance_right(frames=frames,lmlist=llist)

        hip_points,points_hip = self.calculate_angle(frames=frames,lmList=llist,points=hip_points)
        leg_points,points_leg = self.calculate_angle(frames=frames,lmList=llist,points=leg_points)
        elbow_points,point_elbow = self.calculate_angle(frames=frames,lmList=llist,points=elbow_points)

        # cv.putText(frames, f"hip: {hip_points}", (10, 50),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # cv.putText(frames, f"leg: {leg_points}", (10, 100),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # cv.putText(frames, f"elbow: {elbow_points}", (10, 150),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        
        if hip_points is None and leg_points is None and elbow_points is None:
            return None
            
        tolerance=0.05
        hip_y = points_hip[3]
        leg_y = points_leg [3]
        shoulder_y = point_elbow[1]
        hip_foot_diff = abs(int(hip_y - foot_y))
        # cv.putText(frames, f"hip: {hip_foot_diff}", (10, 50),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        

        if hip_points and leg_points:

            if( 
            ((ground_left and ground_left-300 <= leg_y <= ground_left) or (ground_right <= leg_y <= ground_right)) and
            ((ground_left and ground_left-150 <= hip_y <= ground_left) or (ground_right <= hip_y <= ground_right)) and
            ((self.MIN_BACK_LIMT <= self.left_shoulder_hip <= self.MAX_BACK_LIMT) or (self.MIN_BACK_LIMT <= self.right_shoulder_hip <= self.MAX_BACK_LIMT)) and
            ((self.MIN_THIGH_LIMT <=self.left_hip_knee <= self.MAX_THIGH_LIMT) or (self.MIN_THIGH_LIMT <=self.left_hip_knee <= self.MAX_THIGH_LIMT))):
                
                position = "sleeping"

            elif(head_y >hip_y> leg_y > foot_y):
                # print(head_y,leg_y)
                position = "reverse"

            elif(
                shoulder_y < hip_y < leg_y < foot_y and
                45 <= min(self.left_shoulder_hip,self.right_shoulder_hip) <= 90 and
                45 <= min(self.left_hip_knee,self.right_hip_knee) <= 90 and
                hip_points and 160 <= hip_points <= 180 and
                leg_points and 160 <= leg_points <= 180
                ):
                
                position = "standing"

            else:
                position = "sitting"

            cv.putText(frames, f"Position: {position}", (10, 100),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
            return position
        return False

    #based on shoulder to detect side
    def standing_side_view_detect(self,frames,llist,height,width,draw = True):

        if height is None and width is None:
            return None

        self.left_shoulder_hip = self.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=height,width=width,draw=False)
        self.left_hip_knee = self.slope(frames=frames,lmlist=llist,point1=23,point2=25,height=height,width=width,draw=False)

        # # right slope condition
        self.right_shoulder_hip = self.slope(frames=frames,lmlist=llist,point1=12,point2=24,height=height,width=width,draw=False)
        self.right_hip_knee = self.slope(frames=frames,lmlist=llist,point1=24,point2=26,height=height,width=width,draw=False)

        if llist is None or len(llist) == 0 or height is None or width is None:
            return None

        if len(llist) != 0:

            position = "unknown"

            self.check_stand = self.is_person_standing_sitting(frames=frames,llist=llist,hip_points=(11,23,25),leg_points=(23,25,27),elbow_points=(11,13,15),height=height,width=width)

            if self.check_stand == "sitting":

                left_shoulder_x,left_shoulder_y,left_shoulder_z = llist[11][1:]
                right_shoulder_x,right_shoulder_y,right_shoulder_z = llist[12][1:]
                left_ear = llist[7][1:]
                right_ear = llist[8][1:] 

                # Basic side detection using shoulder z-values or visibility
                z_diff = abs(left_shoulder_z - right_shoulder_z)
                    
                if 0 <= z_diff <= 0.15:
                    position = "forward"
                elif z_diff > 0.15:
                    if left_shoulder_z > right_shoulder_z:
                        position = "right"
                    else:
                        position = "left"

                # cv.putText(frames, f"left_shoulder_hip: {position}", (10, 200),
                #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


            elif self.check_stand == "standing":

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

                elif 0 < z_diff < 0.1:
                    position = "forward"

                # cv.putText(frames, f"Position: {position}", (10, 100),
                            # cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # cv.putText(frames, f"diff: {z_diff}", (10, 50),
                            # cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
            elif self.check_stand == "sleeping":

                nose_x = llist[0][0]  

                self.find_view = self.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=self.HEAD_POSITION,head=nose_x)

                if self.find_view == "left":
                    position = "left"
                else:
                    position = "right"

                # cv.putText(frames, f"left_shoulder_hip: {position}", (10, 100),
                #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # cv.putText(frames, f"left_hip_knee: {self.left_hip_knee}", (10, 50),
                #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
            elif self.check_stand == "reverse":

                nose_x = llist[0][0]  

                self.find_view = self.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=self.HEAD_POSITION,head=nose_x)

                if self.find_view == "left":
                    position = "left"
                else:
                    position = "right"

                # cv.putText(frames, f"left_shoulder_hip: {position}", (10, 100),
                #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # cv.putText(frames, f"left_hip_knee: {self.left_hip_knee}", (10, 50),
                #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            return position
        
        return False
    

    # this voice reset after every 30 sec
    def reset_voice(self):

        if self.voice.isVoicePlaying:
            if self.counter >= 31:
                self.counter = 0

    #this voice play after every 30 sec
    def play_voice(self,message,llist):
        
        if len(llist) == 0 and not llist :
            self.counter = 0
            # return 
            
        if not self.voice.isVoicePlaying:
            self.counter += 1
            print(self.counter)
           
        if self.counter > 30:
            self.voice.playAudio([message],play=True)
            self.counter = 0
            return True
            
        return self.counter
    

    #this reset after every 40 sec
    def reset_after_40_sec(self):

        if self.voice.isVoicePlaying:
            if self.after_40 >= 36:
                self.after_40 = 0

    #this voice play after 40 sec
    def play_after_40_sec(self,message,llist):
        
        if len(llist) == 0 and not llist :
            self.after_40 = 0
            # return 
            
        if not self.voice.isVoicePlaying:
            self.after_40 += 1
            print(self.after_40)
           
        if self.after_40 > 35:
            self.voice.playAudio([message],play=True)
            self.after_40 = 0
            return True
            
        return self.after_40 
    
    def all_x_values(self,frames,llist):

        if len(llist) == 0:
            return None
        
        self.nose_x = llist[0][1]
        self.l_hip_x = llist[23][1]
        self.r_hip_x = llist[24][1]
        self.l_knee_x = llist[25][1]
        self.r_knee_x = llist[26][1]
        self.l_ankle_x = llist[27][1]
        self.r_ankle_x = llist[28][1]
        self.l_elbow_x = llist[13][1]
        self.r_elbow_x = llist[14][1]
        self.l_wrist_x = llist[15][1]
        self.r_wrist_x = llist[16][1]
        self.l_shoulder_x = llist[11][1]
        self.r_shoulder_x = llist[12][1]

        print(self.nose_x)

        return self.nose_x,self.l_hip_x,self.r_hip_x,self.l_knee_x,self.r_knee_x,self.l_ankle_x,self.r_ankle_x,self.l_elbow_x,self.r_elbow_x,self.l_wrist_x,self.r_wrist_x,self.l_shoulder_x,self.r_shoulder_x


def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    detect = allmethods()
    pose_detected = None
    global llist
    # global llist1

    while True:

        isTrue,frames = video_capture.read()
        height,width , _ = frames.shape
        img = cv.imread("videos/standing.jpg")
        detect.pose_positions(frames,draw = False)
        llist = detect.pose_landmarks(frames,draw=False)
        # llist1 = detect.pose_landmarks_without_multiply(frames=frames,draw=True)
        # detect.slope(frames=frames,lmlist=llist,point1=11,point2=23,height=detect.h,width=detect.w)
        # detect.is_person_standing_sitting(frames=frames,llist=llist,.hip_points=(11,23,25),leg_points=(23,25,27),elbow_points=(11,13,15),height=height,width=width)
        # detect.sleep_position_detect(frames=frames,llist=llist)
        # detect.camera_distance(frames=frames,llist=llist)
        detect.all_x_values(frames=frames,llist=llist)
        # detect.standing_side_view_detect(frames=frames,llist=llist,height=height,width=width)
        # detect.ground_distance_left(frames)
        # detect.ground_distance_right(frames)

        cv.imshow("video",frames)
        

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
# main()