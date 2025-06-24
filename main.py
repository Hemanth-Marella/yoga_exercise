import mediapipe as mp
import cv2 as cv
from allMethods import allmethods

from pranamasana import pranamasana
from hastauttanasana import hastauttanasana
from uttanasana import uttanasana
from ashwaSanchalanasana import ashwa
from parvatasana import parvatasana
from ashtangaNamaskar import ashtanga
from bhujangasana import bhujangasana
from plank import plankpose
import os
from voiceModule import VoicePlay
from face_detect import HeadPoseEstimator
import time


# ***************************************************************************************
# Prevent From Turning Off The System Screen When App Is Running:
import platform
import ctypes
import os
def prevent_sleep():
    if platform.system() == 'Windows' :  # Windows
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x80000000)
    elif platform.system() == 'Darwin' :  # macOS
        os.system("caffeinate -d &")
    elif platform.system() == 'Linux' : # Linux
        os.system("xdg-screensaver reset")  # Works for most Linux distros

# Call this function at the start of your application
prevent_sleep()
# **************************************************************************************


class DisplayAll:

    def __init__(self, mode=False, max_hands=2, mindetectconf=0.5, mintrackconf=0.5):
        self.mode = mode
        self.mindetectconf = mindetectconf
        self.mintrackconf = mintrackconf
        self.max_hands = max_hands

        self.mpHand = mp.solutions.hands
        self.hand = self.mpHand.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.mindetectconf,
            min_tracking_confidence=self.mintrackconf
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Initializing all pose objects
        self.pranamasana = pranamasana()
        self.hasta = hastauttanasana()
        self.uttanasana = uttanasana()
        self.ashwa = ashwa()
        self.parvatasana = parvatasana()
        self.ashtanga = ashtanga()
        self.bhujangasana = bhujangasana()
        self.plank = plankpose()

        self.mpPose = mp.solutions.pose
        self.pose= self.mpPose.Pose(
            static_image_mode = self.mode,
            min_detection_confidence = self.mindetectconf,
            min_tracking_confidence = self.mintrackconf
        )
        #self.mpDraw= mp.solutions.drawing_utils

        self.lpslist = []
        self.lmslist = []
        self.angle =0
        self.voice_thread = None
        self.voice_detect = False
        self.all_methods = allmethods()
        self.standing_side_view = self.all_methods.standing_side_view_detect
        
        # self.calculate_angle = self.all_methods.calculate_angle

        self.head_point = self.all_methods.head_detect
        self.left_heel_point = self.all_methods.left_heel_detect
        self.right_heel_point = self.all_methods.right_heel_detect

        self.HEAD_POSITION = "head"
        self.TAIL_POSITION = "tail"
        self.LEFT_SIDE_VIEW = "left"
        self.RIGHT_SIDE_VIEW = "right"

        self.suryanamskar = None

        self.head_pose = HeadPoseEstimator()

    def pose_positions(self,frames,draw = True):

        imgRB = cv.cvtColor(frames,cv.COLOR_BGR2RGB)
        self.pose_results = self.pose.process(imgRB)

        if self.pose_results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frames,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

    def pose_landmarks(self,frames,draw = True):

        self.lpslist =[]

        if self.pose_results.pose_landmarks:

            self.h,self.w,cl = frames.shape

            for id,poselms in enumerate(self.pose_results.pose_landmarks.landmark):
                px,py,pz = (poselms.x * self.w) , (poselms.y * self.h),(poselms.z)

                self.lpslist.append([id,px,py,pz])

                if draw:
                    cv.circle(frames,(px,py),5,(255,255,0),2)

        return self.lpslist
    

    def find_hands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hand.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)

        return img

    def get_landmarks(self, img, hand_no=0, draw=True):
        lms_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            h, w, c = img.shape

            for id, lm in enumerate(my_hand.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                lms_list.append([id, px, py])

                if draw:
                    cv.circle(img, (px, py), 2, (0, 255, 255), 2)

        return lms_list

    
    def pranam(self,frames):

        self.pranam_name = None
        self.head_distance = self.pranamasana.head_heel_points(frames,llist=pose_llist,face_points=(0,2,5),left_heel_point=29,right_heel_point=30)
        check_side = self.standing_side_view(frames=frames,llist = pose_llist)

        if check_side == "left":
            self.pranamasana_angles = self.pranamasana.left_pranamasana(frames=frames,llist=pose_llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),draw=False)
            stand_position = self.pranamasana.check_stand(frames=frames,llist=pose_llist,height=self.h,width=self.w)
            if stand_position:
                start_left = self.pranamasana.start_left_exercise(frames=frames)
                if not start_left:
                    wrong_left = self.pranamasana.wrong_pranamasana_left(frames=frames)
            self.pranam_name = self.pranamasana.left_pranamasana_name(frames=frames)
           

        elif check_side == "right":
            self.pranamasana_angles = self.pranamasana.right_pranamasana(frames=frames,llist=pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),draw=False)
            start_right = self.pranamasana.start_right_exercise(frames=frames)
            if not start_right:
                wrong_right = self.pranamasana.wront_right_pranamasana(frames=frames)
            self.pranam_name = self.pranamasana.right_pranamasana_name(frames=frames)

        if self.pranam_name:
            
            return True
        return False
    
    def hastauttasana(self,frames):

        self.fingers_toe = self.hasta.hands_legs_correct_position(frames=frames,llist=pose_llist,points=(16,15,32,31))
        self.head_distance = self.hasta.head_heel_points(frames,llist=pose_llist,face_points=(0,2,5),left_heel_point=29,right_heel_point=30)
        self.hasta_name = None
        check_side = self.standing_side_view(frames=frames,llist = pose_llist)

        if check_side == "right":
            cv.putText(frames,str("right"),(140,150),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
            self.hasta_right = self.hasta.right_hastauttanasana(frames=frames,llist = pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),draw=False)
            start_right = self.hasta.start_right_exercise(frames=frames)
            if not start_right:
                right_wrong = self.hasta.wrong_right(frames=frames)
            self.hasta_name = self.hasta.right_hastauttanasana_name(frames=frames)

        elif check_side == "left":
            cv.putText(frames,str("left"),(140,150),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
            self.hasta_left = self.hasta.left_hastauttanasana(frames=frames,llist = pose_llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),draw=False)
            start_left = self.hasta.start_left_exercise(frames=frames)
            start_left = self.hasta.start_left_exercise(frames=frames)
            if not start_left:
                wrong_left = self.hasta.wrong_left(frames=frames) 
            self.hasta_name = self.hasta.left_hastauttanasana_name(frames=frames)

        # else:
        #     return self.hasta_name
        
        if self.hasta_name:
            return True
        
        else:
            return False
    
    def uttas(self,frames):

        self.fingers_toe = self.uttanasana.hands_legs_correct_position(frames=frames,llist=pose_llist,points=(16,15,32,31))
        self.head_distance = self.uttanasana.head_heel_points(frames,llist=pose_llist,face_points=(0,2,5),left_heel_point=29,right_heel_point=30)

        self.uttas_name = None
        check_side = self.standing_side_view(frames=frames,llist = pose_llist)

        if check_side == "left":
            cv.putText(frames,str("left"),(40,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            self.uttas_left = self.uttanasana.left_uttanasana(frames=frames,llist = pose_llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),right_knee=(24,26,28),draw=False)
            start_left = self.uttanasana.start_left_exercise(frames=frames)
            if not start_left:
                wrong_left = self.uttanasana.wrong_left(frames=frames)
            self.uttas_name = self.uttanasana.left_side_uttanasana_name(frames=frames)


        elif check_side == "right":
            cv.putText(frames,str("right"),(40,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            self.uttas_right = self.uttanasana.right_Uttanasana(frames=frames,llist=pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),draw=False)
            start_right = self.uttanasana.start_right_exercise(frames=frames)
            if not start_right:
                wrong_right = self.uttanasana.wrong_right(frames=frames)
            self.uttas_name = self.uttanasana.right_side_uttanasana_name(frames=frames)

        else:
            return self.uttas_name
        
        if self.uttas_name:
            return True
        
        else:
            return False
    
    def ashwas(self,frames):

        head_values = self.head_point(frames=frames,lmlist=pose_llist,points=(0,2,5))
        if head_values:
            head_x = head_values[0]

            side_view = self.all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=self.HEAD_POSITION,head=head_x)
            self.fingers_toe = self.ashwa.hands_legs_correct_position(frames=frames,llist=pose_llist,points=(16,15,32,31))
            self.ashwas_name = None
        else :
            return 

        if side_view == "right":
            
            self.ashwas_right = self.ashwa.right_ashwa(frames=frames,llist=pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),draw=False)
            self.distance = self.ashwa.right_distance(frames=frames,llist=pose_llist,right_hand=16,right_toe_point=32,face_points=(0,2,5))
            left_leg = self.ashwa.left_leg_slope(frames=frames,llist=pose_llist,left_hip=23,left_knee=25,height=self.h,width=self.w)
            start_right = self.ashwa.start_right_exercise(frames=frames)
            if not start_right:
                wrong_right = self.ashwa.wrong_right(frames=frames)
            self.ashwas_name = self.ashwa.right_ashwa_name(frames=frames)

        elif side_view == "left":

            # slope_check = self.ashwa.left_slope_points(llist=pose_llist,frames=frames,draw=True)
            self.ashwas_left = self.ashwa.left_ashwa(frames=frames,llist=pose_llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),right_knee=(24,26,28),draw=False)
            self.distance = self.ashwa.left_distance(frames=frames,llist=pose_llist,left_hand=15,left_toe_point=31,face_points=(0,2,5))
            right_leg = self.ashwa.right_leg_slope(frames=frames,llist=pose_llist,right_hip=24,right_knee=26,height=self.h,width=self.w)
            start_left = self.ashwa.start_left_exercise(frames=frames)
            if not start_left:
                wrong_left = self.ashwa.wrong_left(frames=frames)
            self.ashwas_name = self.ashwa.left_ashwa_name(frames=frames)

        else:
            return self.ashwas_name
        
        if self.ashwas_name:
            return True
        
        else:
            return False
        
    def plankp(self,frames):

        self.plank_name = None
        head_values = self.head_point(frames=frames,lmlist=pose_llist,points=(0,2,5))
        if head_values:
            head_x = head_values[0]

        if not head_values:
            return

        self.fingers_toe = self.plank.hands_legs_correct_position(frames=frames,llist=pose_llist,points=(16,15,32,31))
        side_view = self.all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=self.HEAD_POSITION,head=head_x)
        
        if side_view == "left":
            self.plank_angles = self.plank.left_plank(frames=frames,llist=pose_llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),right_knee=(24,26,28),draw=False)
            left_slope = self.plank.left_slope_condition(frames=frames,llist=pose_llist,height=self.h,width=self.w)
            start_left = self.plank.start_left_exercise(frames=frames)
            if not start_left:
                wrong_left = self.plank.wrong_left_plank(frames=frames)
            self.plank_name = self.plank.left_plank_name(frames=frames)
           

        elif side_view == "right":
            self.plank_angles = self.plank.right_plank(frames=frames,llist=pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),draw=False)
            right_slope = self.plank.right_slope_condition(frames=frames,llist=pose_llist,height=self.h,width=self.w)
            start_right = self.plank.start_right_exercise(frames=frames)
            if not start_right:
                wrong_right = self.plank.wront_right_plank(frames=frames)
            self.plank_name = self.plank.right_plank_name(frames=frames)

        if self.plank_name:
            
            return True
        return False

    
    def parvat(self,frames):

        self.fingers_toe = self.parvatasana.hands_legs_correct_position(frames=frames,llist=pose_llist,points=(16,15,32,31))
        self.parvat_name = None
        head_values = self.head_point(frames=frames,lmlist=pose_llist,points=(0,2,5))
        if head_values :
            head_x = head_values[0]
            side_view = self.all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=self.HEAD_POSITION,head=head_x)
        else:
            return

        if side_view == "right":
            self.distance = self.parvatasana.right_distance(frames=frames,llist=pose_llist,right_hand=16,right_toe_point=32,face_points=(0,2,5))
            self.paravt_right = self.parvatasana.parvatasana_right(frames,llist = pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),left_knee=(23,25,27),left_shoulder=(15,11,23),left_elbow=(11,13,15),draw=False)
            start_right_exercise = self.parvatasana.start_right_exercise(frames=frames)
            if not start_right_exercise:
                wrong_right = self.parvatasana.wrong_right(frames=frames)
            self.parvat_name = self.parvatasana.right_parvatasana_name(frames=frames,llist=pose_llist)

        elif side_view == "left":
            self.distance = self.parvatasana.left_distance(frames=frames,llist=pose_llist,left_hand=15,left_toe_point=31,face_points=(0,2,5))
            self.paravt_left = self.parvatasana.parvatasana_left(frames,llist = pose_llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),right_knee=(24,26,28),right_shoulder=(16,12,24),right_elbow=(12,14,16),draw=False)
            start_left_exercise = self.parvatasana.start_left_exercise(frames=frames)
            if not start_left_exercise:
                wrong_left = self.parvatasana.wrong_left(frames=frames)
            self.parvat_name = self.parvatasana.left_parvatasana_name(frames=frames,llist=pose_llist)
        else:
            return self.parvat_name
        
        if self.parvat_name:
            return True
        
        else:
            return False
    

    def ashta(self,frames):

        self.ashta_name = None
        self.head = self.ashtanga.horizontal_placed(frames=frames,face_point=(0,2,5),llist=pose_llist,points=(16,15,32,31))
        self.head_val = self.head_point(frames=frames,lmlist=pose_llist,points=(0,2,5))
        if self.head_val:
            head_x = self.head_val[0]
            side_view = self.all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=self.HEAD_POSITION,head=head_x)
        else:
            return

        if side_view == "right":
            cv.putText(frames,str("right"),(40,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            self.ashta_right = self.ashtanga.right_side_ashtanga(frames=frames,llist=pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),draw=False)
            right_slope = self.ashtanga.right_slope_condition(frames=frames,llist=pose_llist,height=self.h,width=self.w)
            start_right = self.ashtanga.start_right_exercise(frames=frames)
            if not start_right:
                wrong_right = self.ashtanga.wrong_right(frames=frames)
            self.ashta_name = self.ashtanga.right_ashtanga_name(frames=frames)

        elif side_view == "left":
            cv.putText(frames,str("Left_side"),(40,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            self.ashta_left = self.ashtanga.left_side_ashtanga(frames=frames,llist=pose_llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),draw=False)
            left_slope = self.ashtanga.left_slope_condition(frames=frames,llist=pose_llist,height=self.h,width=self.w)
            start_left = self.ashtanga.start_left_exercise(frames=frames)
            if not start_left:
                wrong_left = self.ashtanga.wrong_left(frames=frames)
            self.ashta_name = self.ashtanga.left_ashtanga_name(frames=frames)

        else:
            return self.ashta_name
        
        if self.ashta_name:
            return True
        
        else:
            return False

                                                                                    
    def bhujanga(self,frames):

        self.bhujanga_name = None
        head_val = self.head_point(frames=frames,lmlist=pose_llist,points=(0,2,5))
        if head_val:
            head_x = head_val[0]
            side_view = self.all_methods.findSideView(frame=frames,FLAG_HEAD_OR_TAIL_POSITION=self.HEAD_POSITION,head=head_x)
        else:
            return 

        if side_view == "right":
            self.right_bhujanga = self.bhujangasana.bhujangasana_right(frames=frames,llist=pose_llist,elbow=(12,14,16), hip=(12,24,26),knee= (24,26,28), shoulder=(14,12,24),draw=False)
            right_slope = self.bhujangasana.right_slope_condition(frames=frames,llist=pose_llist,height=self.h,width=self.w)
            start_right = self.bhujangasana.start_right_exercise(frames=frames)
            if not start_right:
                wrong_right = self.bhujangasana.wrong_right(frames=frames)
            self.bhujanga_name = self.bhujangasana.bhujangasana_right_name(frames=frames)

        elif side_view == "left":
            self.left_bhujanga = self.bhujangasana.bhujangasana_left(frames=frames,llist=pose_llist,elbow=(11,13 ,15),hip=(11,23,25),knee=(23,25,27) , shoulder=(13,11,23),draw=False)
            left_slope = self.bhujangasana.left_slope_condition(frames=frames,llist=pose_llist,height=self.h,width=self.w)
            start_left = self.bhujangasana.start_left_exercise(frames=frames)
            if not start_left:
                wrong_left = self.bhujangasana.wrong_left(frames=frames)
            self.bhujanga_name = self.bhujangasana.bhujangasana_left_name(frames=frames)

        else:
            return self.bhujanga_name
        
        if self.bhujanga_name:
            return True
        else:
            return False


    def suryanamaskar(self):

        if self.suryanamskar is None:
            from suryanamaskar import Main
            self.suryanamskar = Main()
            
        self.suryanamskar.show()
def main():
    video_capture = cv.VideoCapture(0)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1480)  #width
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 980)  #height

    finder="images"

    my_list= os.listdir(finder)

    overLay=dict({})

    modules = ["pranamasana","hastauttanasana", "uttanasana", "ashwa", "plankpose", "ashtanga", "bhujanga","Parvatasana1","Ashwa Sanchalasana","Uttanasana","Hasta Uttanasana","Pranamasana1" ]
    
    for i, img in enumerate(my_list):
        
        if i < len(my_list):
            image = cv.imread(f'{finder}/{img}')
            image = cv.resize(image, (300, 300))
            
            overLay[modules[i]] = image         

    detect = DisplayAll()
    all_methods = allmethods()
    voice=VoicePlay()
    thumb_detect = False
    module  = "pranamasana"
    print("hello")
    voice_detect = False

    while True:

        isTrue, video = video_capture.read()
        if not isTrue:
            print("Error: Couldn't read the frame")
            break
        global pose_llist
        global slope_llist

        img = cv.imread("videos/ashtanga2.jpg")
        detect.find_hands(video, False)
        detect.pose_positions(video,False)
        hand_llist = detect.get_landmarks(video, draw=True)
        pose_llist = detect.pose_landmarks(video,False)
        # slope_llist = detect.slope_landmarks(frames=video,draw=False)

        if not thumb_detect:
                cv.putText(video, "Show Thumb Up", (30, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
                voice.playAudio(["show thumb up to start the exercise"],play=True)

        if len(hand_llist) != 0:  # Ensure we have enough landmark points
            thumb_tip = hand_llist[3]
            thumb_tip1 = hand_llist[4]
            index_tip = hand_llist[5]
            index_tip1 = hand_llist[6]

            
            if not thumb_detect and thumb_tip[2] < index_tip[2] and thumb_tip[2] < index_tip1[2] and thumb_tip1[2] < thumb_tip[2]:
                # voice.stopEngine()
                module = "pranamasana"
                thumb_detect = True
                          

        if thumb_detect:
            # all_methods.stop()

            for mod in overLay:

                if module == "pranamasana" and not voice_detect and mod == module:
                    
                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.pranam(video)
                    if correct:
                        voice.playAudio(["move to next pose hasta uttanasana"],play=True)
                        # time.sleep(1)
                        module = "hastauttanasana"
                
                elif module == "hastauttanasana" and not voice_detect and mod == module:
                   
                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image
                    
                    correct = detect.hastauttasana(video)
                    if correct:
                        voice.playAudio(["move to next pose uttanasana"],play=True)
                        print("Hastauttanasana detected! Pose sequence complete.")
                        # time.sleep(1)
                        module = "uttanasana"
                    else:
                        module = "hastauttanasana"

                elif module == "uttanasana" and not voice_detect and mod == module:
                
                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.uttas(video)
                    if correct:
                        print("Switching to Ashwa Sanchalasana")
                        module = "ashwa"

                elif module == "ashwa" and not voice_detect and mod == module:

                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.ashwas(video)
                    if correct:
                        print("Switching to plankpose")
                        module = "plankpose"

                elif module == "plankpose" and not voice_detect and mod == module:

                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.plankp(video)
                    if correct:
                        print("Switching to ashtanga namaskar")
                        module = "ashtanga"

                elif module == "ashtanga" and not voice_detect and mod == module:

                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.ashta(video)
                    if correct:
                        print("Switching to bhujanga")
                        module = "bhujanga"

                elif module == "bhujanga" and not voice_detect and mod == module:

                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.bhujanga(video)
                    if correct:
                        print("Switching to parvatasana")
                        module = "Parvatasana1"
                
                elif module == "Parvatasana1" and not voice_detect and mod == module:

                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.parvat(video)
                    if correct:
                        print("Switching to Ashwa Sanchalasana")
                        module = "Ashwa Sanchalasana"

                elif module == "Ashwa Sanchalasana" and not voice_detect and mod == module:

                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.ashwas(video)
                    if correct:
                        print("Switching to Uttanasana")
                        module = "Uttanasana"

                elif module == "Uttanasana" and not voice_detect and mod == module:

                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.uttas(video)
                    if correct:
                        print("Switching to Hasta Uttasana")
                        module = "Hasta Uttanasana"

                elif module == "Hasta Uttanasana" and not voice_detect and mod == module:

                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.hastauttasana(video)
                    if correct:
                        print("Switching to Pranamasana")
                        module = "Pranamasana1"

                elif module == "Pranamasana1" and not voice_detect and mod == module:

                    image = overLay[mod]
                    h, w, _ = image.shape
                    video[0:h, 0:w] = image

                    correct = detect.pranam(video)
                    if correct:
                        print("Switching to suryanamaskar")
                        voice.playAudio(["surya Namaskar is completed"],play=True)
                        break
                        # module = "Surya Namaskar"
                
                # elif module == "Surya Namaskar":
                #     # voice.playAudio(["surya Namaskar is completed"],play=True)
                #     detect.suryanamaskar()
                #     break

            cv.putText(video, f"Current Pose:{module} ", (40, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
        cv.imshow("video", video)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break

    video_capture.release()
    cv.destroyAllWindows()


main()