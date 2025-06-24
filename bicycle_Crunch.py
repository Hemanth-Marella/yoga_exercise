import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math
from BodyPoseValues import BodyPoseValues


class bicycle_crunch:
     
    def __init__(self,mode = False,mindetectconf=0.5,mintrcackconf=0.5):
        self.bodyPoseValues = BodyPoseValues()
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

        
        self.angle =0
    

    def pose_positions(self,frames,draw = True):

        imgRB = cv.cvtColor(frames,cv.COLOR_BGR2RGB)
        h,w,cl = frames.shape
        self.results = self.pose.process(imgRB)

        # if self.results.pose_landmarks:
        #     self.bodyPoseValues.process_pose_landmarks(pose_landmarks=self.results.pose_landmarks, width=w, height=h, image=frames)
        #     cv.putText(frames, f'POSITION:=> {self.bodyPoseValues._calculate_angle(self.bodyPoseValues.left_wrist, self.bodyPoseValues.left_elbow, self.bodyPoseValues.left_shoulder, draw=True,  color = (0,255,0), thickness =4 )}', (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
            
        if draw:
            self.mpDraw.draw_landmarks(frames,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

    def pose_landmarks(self,frames,draw = True):

        self.lpslist =[]

        if self.results.pose_landmarks:

            self.h,self.w,cl = frames.shape

            for id,poselms in enumerate(self.results.pose_landmarks.landmark):
                self.px,self.py = int(poselms.x ) , int(poselms.y )

                self.lpslist.append([id,self.px,self.py])

                if draw:
                    cv.circle(frames,(self.px,self.py),5,(255,255,0),2)

        return self.lpslist
    
    
    def left_bicycle_Crunch(self,frames,draw =True):

        if self.results.pose_landmarks:
            self.left_elbow = self.bodyPoseValues.process_pose_landmarks(pose_landmarks=self.results.pose_landmarks, width=self.w, height=self.h, image=frames)
            self.left_hip = self.bodyPoseValues.process_pose_landmarks(pose_landmarks=self.results.pose_landmarks, width=self.w, height=self.h, image=frames)
            self.left_knee = self.bodyPoseValues.process_pose_landmarks(pose_landmarks=self.results.pose_landmarks, width=self.w, height=self.h, image=frames) 
            self.left_shoulder = self.bodyPoseValues.process_pose_landmarks(pose_landmarks=self.results.pose_landmarks, width=self.w, height=self.h, image=frames)
            
            if draw :

                cv.putText(frames, f'elbow_POSITION:=> {self.bodyPoseValues._calculate_angle(self.bodyPoseValues.left_wrist, self.bodyPoseValues.left_elbow, self.bodyPoseValues.left_shoulder, draw=True,  color = (0,255,0), thickness =4 )}', (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
                cv.putText(frames, f'hip_POSITION:=> {self.bodyPoseValues._calculate_angle(self.bodyPoseValues.left_shoulder, self.bodyPoseValues.left_hip, self.bodyPoseValues.left_knee, draw=True,  color = (0,255,0), thickness =4 )}', (10, 60), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
                cv.putText(frames, f'knee_POSITION:=> {self.bodyPoseValues._calculate_angle(self.bodyPoseValues.left_hip, self.bodyPoseValues.left_knee, self.bodyPoseValues.left_ankle, draw=True,  color = (0,255,0), thickness =4 )}', (10, 90), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
                cv.putText(frames, f'shoulder_POSITION:=> {self.bodyPoseValues._calculate_angle(self.bodyPoseValues.left_elbow, self.bodyPoseValues.left_shoulder, self.bodyPoseValues.left_hip, draw=True,  color = (0,255,0), thickness =4 )}', (10, 120), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)

        return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder

    def right_bicycle_Crunch(self,frames,draw=True):

        if self.results.pose_landmarks:
            self.right_elbow = self.bodyPoseValues.process_pose_landmarks(pose_landmarks=self.results.pose_landmarks, width=self.w, height=self.h, image=frames)
            self.right_hip = self.bodyPoseValues.process_pose_landmarks(pose_landmarks=self.results.pose_landmarks, width=self.w, height=self.h, image=frames)
            self.right_knee = self.bodyPoseValues.process_pose_landmarks(pose_landmarks=self.results.pose_landmarks, width=self.w, height=self.h, image=frames) 
            self.right_shoulder = self.bodyPoseValues.process_pose_landmarks(pose_landmarks=self.results.pose_landmarks, width=self.w, height=self.h, image=frames)
            
            if draw :

                cv.putText(frames, f'elbow_POSITION:=> {self.bodyPoseValues._calculate_angle(self.bodyPoseValues.right_wrist, self.bodyPoseValues.right_elbow, self.bodyPoseValues.right_shoulder, draw=True,  color = (0,255,0), thickness =4 )}', (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
                cv.putText(frames, f'hip_POSITION:=> {self.bodyPoseValues._calculate_angle(self.bodyPoseValues.right_shoulder, self.bodyPoseValues.right_hip, self.bodyPoseValues.right_knee, draw=True,  color = (0,255,0), thickness =4 )}', (10, 60), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
                cv.putText(frames, f'knee_POSITION:=> {self.bodyPoseValues._calculate_angle(self.bodyPoseValues.right_hip, self.bodyPoseValues.right_knee, self.bodyPoseValues.right_ankle, draw=True,  color = (0,255,0), thickness =4 )}', (10, 90), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
                cv.putText(frames, f'shoulder_POSITION:=> {self.bodyPoseValues._calculate_angle(self.bodyPoseValues.right_elbow, self.bodyPoseValues.right_shoulder, self.bodyPoseValues.right_hip, draw=True,  color = (0,255,0), thickness =4 )}', (10, 120), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)

        return self.right_elbow,self.right_hip,self.right_knee,self.right_shoulder

    def left_head_hand_touch_(self,frames,draw=True):

        if self.results.pose_landmarks:

            left_ear_x,left_ear_y      = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_EAR].x,self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_EAR].y
            left_pinky_x,left_pinky_y  =  self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_PINKY].x,self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_PINKY].y
            left_index_x,left_index_y  =  self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_INDEX].x,self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_INDEX].y
            left_wrist_x,left_wrist_y  =  self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_WRIST].x,self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_WRIST].y

            left_palm_average_x = (left_pinky_x+left_index_x+left_wrist_x )/3
            
            if left_palm_average_x:

                if left_palm_average_x > left_ear_x:
                    cv.putText(frames,str('it cross the left ear in left side'),(10,180),cv.FONT_HERSHEY_DUPLEX,2,(255,0,0),2)
    
def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980) 
    global  llist
    detect = bicycle_crunch()
    
    while True:

        isTrue,frames = video_capture.read()

        detect.pose_positions(frames,draw=False)
        llist = detect.pose_landmarks(frames,False)

        detect.left_bicycle_Crunch(frames=frames,draw=True)
        detect.left_head_hand_touch_(frames=frames,draw=True)
              
        cv.imshow("video",frames)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()
main()
