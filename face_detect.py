

import cv2 as cv
import mediapipe as mp
import numpy as np

from allMethods import allmethods

class HeadPoseEstimator:

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

        self.lmlist = []
        self.all_methods = allmethods()
        self.side = None

    def pose_positions(self,frames,draw = True):

        imgRB = cv.cvtColor(frames,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frames,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

    def pose_landmarks(self,frames,draw = True):

        self.lmlist =[]

        if self.results.pose_landmarks:

            h,w,cl = frames.shape

            for id,poselms in enumerate(self.results.pose_landmarks.landmark):
                px,py ,pz= int(poselms.x * w) , int(poselms.y * h),(poselms.z)

                self.lmlist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py,pz),5,(255,255,0),2)

        return self.lmlist
    

    def head_detect(self,frames,llist,points):
        
        self.head_points = self.all_methods.head_detect(frames=frames,lmlist=llist,points=points)

        if not self.head_points:
            return
        
        self.head_x =self.head_points[0]
        self.head_y =self.head_points[1]
        self.head_z =self.head_points[2] 

        return self.head_x,self.head_y,self.head_z


    def left_ear_detect(self,frames,llist,left_point):

        if len(llist) != 0:

            self.left_ear_x,self.left_ear_y,self.left_ear_z = llist[left_point][1:]

            return self.left_ear_x,self.left_ear_y,self.left_ear_z
        return False
    
    def right_ear_detect(self,frames,llist,right_point):

        if len(llist) != 0:

            self.right_ear_x,self.right_ear_y,self.right_ear_z = llist[right_point][1:]

            return self.right_ear_x,self.right_ear_y,self.right_ear_z
        return False
    
    def head_position_detect(self,frames,llist):

        if len(llist) == 0:
            return None

        if len(llist) != 0:

            self.side = None

            ear_avg_y = (self.left_ear_y + self.right_ear_y) // 2

            self.diff = self.head_y - ear_avg_y
        
            if self.diff < -10:
                self.side = "Up"
                # self.all_methods.reset_voice()
                # self.all_methods.play_voice(["you are in up side"],llist=llist)
                
            elif self.diff > 15:
                self.side = "Down"
                # self.all_methods.reset_voice()
                # self.all_methods.play_voice(["you are in down side"],llist=llist)

            elif ((self.left_ear_x > self.head_x > self.right_ear_x) and (-9 <= self.diff <= 15)):

                self.side = "Forward"
                # self.all_methods.reset_voice()
                # self.all_methods.play_voice(["you are in forward side"],llist=llist)

            elif self.right_ear_x < self.head_x:
                self.side = "Right"
                # self.all_methods.reset_voice()
                # self.all_methods.play_voice(["you are in right side"],llist=llist)

            elif self.left_ear_x > self.head_x:
                self.side = "Left"
                # self.all_methods.reset_voice()
                # self.all_methods.play_voice(["you are in left side"],llist=llist)

            elif self.diff == None:
                # self.all_methods.reset_voice()
                # self.all_methods.play_voice(["you are in back side"],llist=llist)
                return

            cv.putText(frames,str(self.side),(14,18),(cv.FONT_HERSHEY_PLAIN),2,(255,0,0),2)

            return self.side
        
        return False
    

def main():
    video_capture = cv.VideoCapture(0)
    detect = HeadPoseEstimator()
    pose_detected = None
    global llist

    while True:

        isTrue,frames = video_capture.read()
        img = cv.imread("videos/standing.jpg")
        detect.pose_positions(frames,draw = False)
        llist = detect.pose_landmarks(frames,draw=False)
        detect.head_detect(frames=frames,llist=llist,points=(0,2,5))
        detect.left_ear_detect(frames=frames,llist=llist,left_point=7)
        detect.right_ear_detect(frames=frames,llist=llist,right_point=8)
        detect.head_position_detect(frames=frames,llist=llist)

        cv.imshow("video",frames)
        if cv.waitKey(10) & 0xFF == ord('d'):
            break
    
    video_capture.release()
    cv.destroyAllWindows()

# main()