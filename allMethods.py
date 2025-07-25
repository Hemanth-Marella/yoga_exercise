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
        self.first_count = 0
        self.voice_delay_ready = True

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
        self.trigger = None


        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        #shoulder hip
        self.MAX_BACK_LIMT = 30
        self.MIN_BACK_LIMT = 0
        
        #hip knee
        self.MAX_THIGH_LIMT = 30
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
            cv.line(frames, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), (0, 255, 0), 16)
            cv.line(frames, (int(self.x3), int(self.y3)), (int(self.x2), int(self.y2)), (0, 255, 0), 16)

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

        if head is None:
            return None

        elif FLAG_HEAD_OR_TAIL_POSITION == self.HEAD_POSITION:
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
                55 <= min(self.left_shoulder_hip,self.right_shoulder_hip) <= 90 and
                55 <= min(self.left_hip_knee,self.right_hip_knee) <= 90 and
                hip_points and 160 <= hip_points <= 180 and
                leg_points and 160 <= leg_points <= 180
                ):
                
                position = "standing"

            else:
                position = "sitting"
        
            return position
        return False

    # def is_person_standing_OR_sitting(self, shoulder_left, shoulder_right, hip_left, hip_right, ankle_left, ankle_right, knee_left, knee_right,
    #                                   Y_THRESHOLD = 0.05, ANGLE_THRESHOLD = 160, ):
            
    #         # Calculate the average Y-values for shoulders, hips, and ankles
    #         avg_shoulder_y = (shoulder_left[1] + shoulder_right[1]) / 2
    #         avg_hip_y = (hip_left[1] + hip_right[1]) / 2
    #         avg_ankle_y = (ankle_left[1] + ankle_right[1]) / 2
            
    #         # Calculate the angle between shoulder, hip, and knee
    #         left_a = [shoulder_left[0] - hip_left[0], shoulder_left[1] - hip_left[1]]
    #         left_b = [knee_left[0] - hip_left[0], knee_left[1] - hip_left[1]]
    #         left_dot_product = left_a[0] * left_b[0] + left_a[1] * left_b[1]
    #         left_magnitude_a = math.sqrt(left_a[0]*2 + left_a[1]*2)
    #         left_magnitude_b = math.sqrt(left_b[0]*2 + left_b[1]*2)
    #         left_cos_angle = left_dot_product / (left_magnitude_a * left_magnitude_b)
    #         # Left_Angle:
    #         left_angle = math.acos(left_cos_angle) * (180.0 / math.pi)
            
            
    #         right_a = [shoulder_right[0] - hip_right[0], shoulder_right[1] - hip_right[1]]
    #         right_b = [knee_right[0] - hip_right[0], knee_right[1] - hip_right[1]]
    #         right_dot_product = right_a[0] * right_b[0] + right_a[1] * right_b[1]
    #         right_magnitude_a = math.sqrt(right_a[0]*2 + right_a[1]*2)
    #         right_magnitude_b = math.sqrt(right_b[0]*2 + right_b[1]*2)
    #         right_cos_angle = right_dot_product / (right_magnitude_a * right_magnitude_b)
    #         # Right_Angle:
    #         right_angle = math.acos(right_cos_angle) * (180.0 / math.pi)

    #         # Determine the position (standing or sitting) based on Y-values and angles
    #         if avg_shoulder_y < avg_hip_y - Y_THRESHOLD and avg_hip_y < avg_ankle_y - Y_THRESHOLD and (left_angle > ANGLE_THRESHOLD and right_angle > ANGLE_THRESHOLD):
    #             self.POSITION = 1
    #         elif avg_ankle_y > avg_hip_y + Y_THRESHOLD > avg_shoulder_y and (left_angle < ANGLE_THRESHOLD or right_angle < ANGLE_THRESHOLD):
    #             self.POSITION = 2
    #         else:
    #             self.POSITION = None  # In case the posture is ambiguous or body parts are not well detected
                
    #         return self.POSITION

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

                left_shoulder_x,left_shoulder_y,self.left_shoulder_z = llist[11][1:]
                right_shoulder_x,right_shoulder_y,self.right_shoulder_z = llist[12][1:]
                left_ear = llist[7][1:]
                right_ear = llist[8][1:] 

                # Basic side detection using shoulder z-values or visibility
                z_diff = abs(self.left_shoulder_z  - self.right_shoulder_z)
                    
                if 0 <= z_diff <= 0.15:
                    position = "forward"
                elif z_diff > 0.15:
                    if self.left_shoulder_z  > self.right_shoulder_z:
                        position = "right"
                    else:
                        position = "left"

                # cv.putText(frames, f"left_shoulder_hip: {position}", (10, 200),
                #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


            elif self.check_stand == "standing":

                left_shoulder_x,left_shoulder_y,self.left_shoulder_z  = llist[11][1:]
                right_shoulder_x,right_shoulder_y,self.right_shoulder_z = llist[12][1:]
                left_ear = llist[7][1:]
                right_ear = llist[8][1:] 

                # Basic side detection using shoulder z-values or visibility
                z_diff = abs(self.left_shoulder_z  - self.right_shoulder_z)

                if z_diff > 0.25:
                    if self.left_shoulder_z  > self.right_shoulder_z:
                        position = "right"
                    else:
                        position = "left"

                elif 0 < z_diff < 0.25:
                    position = "forward"
                
            elif self.check_stand == "sleeping":

                nose_x = llist[0][0]  
                left_hip_z,right_hip_z = llist[23][3] , llist[24][3]
                left_knee_z,right_knee_z = llist[25][3],llist[26][3]
                left_ankle_z,right_ankle_z = llist[27][3],llist[28][3]

                #SHOULDER TOLERANCE
                tolerance_shoulders = abs(int(self.left_shoulder_z  - self.right_shoulder_z))   
                # HIP TOLERANCE
                tolerance_hips = abs(int(left_hip_z - right_hip_z))
                # KNEE TOLERANCE
                tolerance_knees = abs(int(left_knee_z - right_knee_z))
                # ANKLE TOLERANCE
                tolerance_ankles = abs(int(left_ankle_z - right_ankle_z))

                if tolerance_ankles <= 30 and tolerance_hips <= 30 and tolerance_knees <= 30 and tolerance_shoulders <= 30:
                    position = "side position"

                elif (
                    left_hip_z < right_hip_z and
                    self.left_shoulder_z  < self.right_shoulder_z and
                    left_knee_z < right_knee_z and
                    left_ankle_z < right_ankle_z and
                    tolerance_hips > 30 and tolerance_shoulders > 30 and tolerance_knees > 30 and tolerance_ankles > 30
                ):
                    position = "left"

                # RIGHT SIDE (body turned showing right side closer to camera)
                elif (
                    right_hip_z < left_hip_z and
                    self.right_shoulder_z < self.left_shoulder_z  and
                    right_knee_z < left_knee_z and
                    right_ankle_z < left_ankle_z and
                    tolerance_hips > 30 and tolerance_shoulders > 30 and tolerance_knees > 30 and tolerance_ankles > 30
                ):
                    position = "right"

                elif(
                    left_hip_z > right_hip_z and
                    self.left_shoulder_z  > self.right_shoulder_z and
                    left_knee_z > right_knee_z and
                    left_ankle_z > right_ankle_z and
                    tolerance_hips > 30 and tolerance_shoulders > 30 and tolerance_knees > 30 and tolerance_ankles > 30
                ):
                    position = "left_reverse"

                elif(
                    right_hip_z < left_hip_z and
                    self.right_shoulder_z < self.left_shoulder_z  and
                    right_knee_z < left_knee_z and
                    right_ankle_z < left_ankle_z and
                    tolerance_hips > 30 and tolerance_shoulders > 30 and tolerance_knees > 30 and tolerance_ankles > 30
                ):
                    position = "right_reverse"
                # cv.putText(frames, f"left_shoulder_hip: {position}", (10, 100),
                #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # cv.putText(frames, f"left_hip_knee: {self.left_hip_knee}", (10, 50),
                #             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            return position
        
        return False
    

    # this voice reset after every 30 sec
    def reset_voice(self):
        # self.after_40 = 0
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
            # print(self.counter)
           
        if self.counter > 31:
            self.voice.playAudio([message],play=True)
            self.counter = 0
            return True
            
        return False

    #this reset after every 40 sec
    def reset_after_40_sec(self):
        
        if self.voice.isVoicePlaying:
            if self.after_40 >= 34:
                self.after_40 = 0

    #this voice play after 40 sec
    def play_after_40_sec(self,message,llist):

        # self.after_40 = 0
        if len(llist) == 0 and not llist :
            self.after_40 = 0
            # return 
            
        if not self.voice.isVoicePlaying:
            
            self.after_40 += 1
           
        if self.after_40 > 35:
            self.trigger = self.voice.playAudio([message],play=True)
            self.after_40 = 0
            return True
            
        return False

    
    def stop_sometime(self):

        if not self.voice.isVoicePlaying:
            self.first_count += 1
            print(self.first_count)

        if self.voice.isVoicePlaying:
            self.first_count = 0

        return self.first_count
    
    def all_x_values(self,frames,llist):

        if llist is None or len(llist) == 0 :
            return None
        
        self.nose_x = llist[0][1]

        if not self.nose_x:
            return None
        
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
        self.r_toe_x = llist[32][1]
        self.l_toe_x = llist[31][1]

        # print(self.nose_x)

        return self.nose_x,self.l_hip_x,self.r_hip_x,self.l_knee_x,self.r_knee_x,self.l_ankle_x,self.r_ankle_x,self.l_elbow_x,self.r_elbow_x,self.l_wrist_x,self.r_wrist_x,self.l_shoulder_x,self.r_shoulder_x


    def all_y_values(self,frames,llist):

        if llist is None or len(llist) == 0 :
            return None
        
        self.nose_y = llist[0][2]

        if not self.nose_y:
            return None
        
        self.l_hip_y = llist[23][2]
        self.r_hip_y = llist[24][2]
        self.l_knee_y = llist[25][2]
        self.r_knee_y = llist[26][2]
        self.l_ankle_y = llist[27][2]
        self.r_ankle_y = llist[28][2]
        self.l_elbow_y = llist[13][2]
        self.r_elbow_y = llist[14][2]
        self.l_wrist_y = llist[15][2]
        self.r_wrist_y = llist[16][2]
        self.l_shoulder_y = llist[11][2]
        self.r_shoulder_y = llist[12][2]
        self.r_toe_y = llist[32][2]
        self.l_toe_y = llist[31][2]

        return self.nose_y,self.l_hip_y,self.r_hip_y,self.l_knee_y,self.r_knee_y,self.l_ankle_y,self.r_ankle_y,self.l_elbow_y,self.r_elbow_y,self.l_wrist_y,self.r_wrist_y,self.l_shoulder_y,self.r_shoulder_y

    def all_z_values(self,frames,llist):

        if llist is None or len(llist) == 0:
            return None
        
        self.nose_z = llist[0][3]
        self.l_hip_z = llist[23][3]
        self.r_hip_z = llist[24][3]
        self.l_knee_z = llist[25][3]
        self.r_knee_z = llist[26][3]
        self.l_ankle_z = llist[27][3]
        self.r_ankle_z = llist[28][3]
        self.l_elbow_z = llist[13][3]
        self.r_elbow_z = llist[14][3]
        self.l_wrist_z = llist[15][3]
        self.r_wrist_z = llist[16][3]
        self.l_shoulder_z = llist[11][3]
        self.r_shoulder_z = llist[12][3]
        self.r_toe_z = llist[32][3]
        self.l_toe_z = llist[31][3]

        return self.nose_z,self.l_hip_z,self.r_hip_z,self.l_knee_z,self.r_knee_z,self.l_ankle_z,self.r_ankle_z,self.l_elbow_z,self.r_elbow_z,self.l_wrist_z,self.r_wrist_z,self.l_shoulder_z,self.r_shoulder_z

    def check_same_y_axis_between_twoPoints(self, point1, point2, tolerance=40):
        point1_y = point1[1]
        point2_y = point2[1]
        if abs(point1_y - point2_y) <= tolerance :
            return True
        return False
    
    def check_same_x_axis_between_twoPoints(self, point1, point2, tolerance=40):
        point1_x = point1[0]
        point2_x = point2[0]
        if abs(point1_x - point2_x) <= tolerance :
            return True
        return False

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