import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import math


from voiceModule import VoicePlay
from allMethods import treePose
from body_position import bodyPosition


class side_view_position:
     
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
        self.tree_pose = treePose()
        self.body_position = bodyPosition()
        self.voice = VoicePlay()

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
                px,py,pz= (poselms.x * self.w) , (poselms.y * self.h),(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lpslist 
    
    def side_view(self,frames,lmlist,draw=True):

        self.check_stand = self.body_position.is_person_standing_sitting(frames=frames,llist=llist,hip_points=(11,23,25),leg_points=(23,25,27),elbow_points=(11,13,15),height=self.h,width=self.w)

        if self.check_stand != 'sleeping':

            left_shoulder_x,left_shoulder_y,left_shoulder_z = lmlist[11][1:]
            right_shoulder_x,right_shoulder_y,right_shoulder_z = lmlist[12][1:]
            left_ear = lmlist[7][1:]
            right_ear = lmlist[8][1:] 

            # Basic side detection using shoulder z-values or visibility
            z_diff = abs(left_shoulder_z - right_shoulder_z)

            if z_diff > 0.1 and z_diff < 0.25:
                if left_shoulder_z > right_shoulder_z:
                    position = "right cross position"
                else:
                    position = "left side cross position"

            elif z_diff > 0.25:
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
            
        # elif self.check_stand == 'sleeping':
        #     self.voice.playAudio(["you must be in standing or sitting position suitable for exercise"],play=True)

        return position

def main():

    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980)  #height

    detect = side_view_position()
    global llist

    while True:

        isTrue,frames = video_capture.read()

        if not isTrue:

            print("frame was ended")

        detect.pose_positions(frames=frames,draw=True)
        llist = detect.pose_landmarks(frames=frames,draw=False)
        detect.side_view(frames=frames,lmlist=llist,draw=True)

        cv.imshow("video",frames)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()

main()
      