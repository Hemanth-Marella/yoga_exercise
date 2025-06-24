 # def body_mid_point(self,frames,points1,points2,points3,points4,points5,points6,points7,points8,draw = True):

    #     if not self.lmlist :
    #         return None
        
    #     self.x1,self.y1 = self.lmlist[points1[0]][1:] 
    #     self.x2,self.y2 = self.lmlist[points1[1]][1:] 
    #     self.x3,self.y3 = self.lmlist[points1[2]][1:] 

    #     self.x4,self.y4 = self.lmlist[points2[0]][1:] 
    #     self.x5,self.y5 = self.lmlist[points2[1]][1:] 
    #     self.x6,self.y6 = self.lmlist[points2[2]][1:] 

    #     self.x7,self.y7 = self.lmlist[points3[0]][1:] 
    #     self.x8,self.y8 = self.lmlist[points3[1]][1:] 
    #     self.x9,self.y9 = self.lmlist[points3[2]][1:] 

    #     self.x10,self.y10 = self.lmlist[points4[0]][1:] 
    #     self.x11,self.y11 = self.lmlist[points4[1]][1:] 
    #     self.x12,self.y12 = self.lmlist[points4[2]][1:] 

    #     self.x13,self.y13 = self.lmlist[points5[0]][1:] 
    #     self.x14,self.y14 = self.lmlist[points5[1]][1:] 
    #     self.x15,self.y15 = self.lmlist[points5[2]][1:] 

    #     self.x16,self.y16 = self.lmlist[points6[0]][1:] 
    #     self.x17,self.y17 = self.lmlist[points6[1]][1:] 
    #     self.x18,self.y18 = self.lmlist[points6[2]][1:] 

    #     self.x19,self.y19 = self.lmlist[points7[0]][1:] 
    #     self.x20,self.y20 = self.lmlist[points7[1]][1:] 
    #     self.x21,self.y21 = self.lmlist[points7[2]][1:]

    #     self.x22,self.y22 = self.lmlist[points8[0]][1:] 
    #     self.x23,self.y23 = self.lmlist[points8[1]][1:] 
    #     self.x24,self.y24 = self.lmlist[points8[2]][1:] 

    #     if(self.x1 and self.x2 and self.x3 and self.x4 and self.x5 and self.x6 and self.x7 and self.x8 and self.x9 and self.x10 and 
    #         self.x11 and self.x12 and self.x13 and self.x14 and self.x15 and self.x16 and self.x17 and self.x18 and self.x18 and 
    #         self.x19 and self.x20 and self.x21 and self.x22 and self.x23 and self.x24     and        
    #         self.y1 and self.y2 and self.y3 and self.y4 and self.y5 and self.y6 and self.y7 and self.y8 and self.y9 and self.y10 and 
    #         self.y11 and self.y12 and self.y13 and self.y14 and self.y15 and self.y16 and self.y17 and self.y18 and self.y18 and 
    #         self.y19 and self.y20 and self.y21 and self.y2 and self.y3 and self.y24 ):

    #         self.x_points = (self.x1 + self.x2+self.x3+self.x4+self.x5+self.x6+self.x7+self.x8+self.x9+self.x10+
    #                       self.x11+self.x12+self.x13+self.x14+self.x15+self.x16+self.x17+self.x18+self.x18+
    #                       self.x19+self.x20+self.x21+self.x22+self.x23+self.x24) 
        
    #         self.y_poiints = (self.y1 + self.y2+self.y3+self.y4+self.y5+self.y6+self.y7+self.y8+self.y9+self.y10+
    #                       self.y11+self.y12+self.y13+self.y14+self.y15+self.y16+self.y17+self.y18+self.y18+
    #                       self.y19+self.y20+self.y21+self.y2+self.y3+self.y24)
        
    #     self.mid_point = (self.x_points+self.y_poiints) // 2

    #     print(self.mid_point)






# def ear_hip_side_view(self,frames):

#         left_shoulder = self.lmlist[11]
#         right_shoulder = self.lmlist[12]
#         left_hip = self.lmlist[23]
#         right_hip = self.lmlist[24]

#         try:

#             left_shoulder_x = int(left_shoulder.x)
#             right_shoulder_x = int(right_shoulder.x)
#             left_hip_x = int(left_hip.x)
#             right_hip_x = int(right_hip.x)

#             shoulder_mid_x = (left_shoulder_x + right_shoulder_x) // 2
#             hip_mid_x = (left_hip_x + right_hip_x) // 2
#             body_mid_x = (shoulder_mid_x + hip_mid_x) // 2

#             # Determine body orientation
#             body_direction = "Straight"
#             print(body_direction)
#             if right_shoulder_x < left_shoulder_x:  # Right shoulder forward
#                 body_direction = "Left"
#                 print(body_direction)
#             elif left_shoulder_x < right_shoulder_x:  # Left shoulder forward
#                 body_direction = "Right"
#                 print(body_direction)

#             return body_direction
# #         except Exception:
# #             return None


# dict1 = {1:"hemanth",2:"msd"}

# for value in dict1:
#     print(dict1[value])



# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# class HeadPoseEstimator:
#     def _init_(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
#         self.sideView = None
    
#     def process_frame(self, frame):
#         start = time.time()
#         img_h, img_w, _ = frame.shape


#         # Convert frame to RGB
#         image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
#         results = self.face_mesh.process(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         face_3d, face_2d = [], []
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 for idx, lm in enumerate(face_landmarks.landmark):
#                     if idx in [33, 263, 1, 61, 291, 199]:
#                         x, y = int(lm.x * img_w), int(lm.y * img_h)
#                         face_2d.append([x, y])
#                         face_3d.append([x, y, lm.z])
#                         if idx == 1:
#                             nose_2d = (x, y)
#                             nose_3d = (x, y, lm.x * 3000)
                
#                 face_2d = np.array(face_2d, dtype=np.float64)
#                 face_3d = np.array(face_3d, dtype=np.float64)

#                 focal_length = img_w
#                 cam_matrix = np.array([
#                     [focal_length, 0, img_w / 2],
#                     [0, focal_length, img_h / 2],
#                     [0, 0, 1]
#                 ])
#                 dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
#                 success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
#                 rmat, _ = cv2.Rodrigues(rot_vec)
#                 angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
#                 x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
                
#                 if y < -10:
#                     self.sideView = 'Looking LEFT'
#                 elif y > 10:
#                     self.sideView = 'Looking RIGHT'
#                 elif x < -10:
#                     self.sideView = 'Looking DOWN'
#                 elif x > 10:
#                     self.sideView = 'Looking UP'
#                 else:
#                     self.sideView = 'FORWARD'
                
#                 nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
#                 p1 = (int(nose_2d[0]), int(nose_2d[1]))
#                 p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
#                 cv2.line(image, p1, p2, (0, 0, 255), 3)
#                 cv2.circle(image, p2, 5, (255, 0, 0), -1)
                
#                 cv2.putText(image, self.sideView, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
#                 cv2.putText(image, f'x: {x:.2f}', (500, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#                 cv2.putText(image, f'y: {y:.2f}', (500, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#                 cv2.putText(image, f'z: {z:.2f}', (500, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
#         end = time.time()
#         fps = 1 / (end - start)
#         cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
#         return image
    
#     def run(self):
#         self.cap = cv2.VideoCapture(0)
#         while self.cap.isOpened():
#             success, frame = self.cap.read()
#             if not success:
#                 break
#             processed_frame = self.process_frame(frame)
#             cv2.imshow('HEAD POSE ESTIMATION', processed_frame)
#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break
#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     estimator = HeadPoseEstimator()  # Change camera_index for different cameras
#     estimator.run()




    # def left_pranamasana(self,frames,points4, points5, points6,points8,draw=True):

    #     self.left_elbow, points4_coords = self.tree_pose.calculate_angle(frames=frames, points=points4,lmList=llist)
    #     self.left_hip, points5_coords = self.tree_pose.calculate_angle(frames=frames,points= points5,lmList=llist)
    #     self.left_knee, points6_coords = self.tree_pose.calculate_angle(frames=frames,points= points6,lmList=llist)       
    #     self.left_shoulder,points_cor8 = self.tree_pose.calculate_angle(frames=frames,points=points8, lmList=llist)

    #     if draw:
    #         if points4_coords:
    #             cv.putText(frames, str(int(self.left_elbow)), (points4_coords[2]+10, points4_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    #         if points5_coords:
    #             cv.putText(frames, str(int(self.left_hip)), (points5_coords[2]+10, points5_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    #         if points6_coords:
    #             cv.putText(frames, str(int(self.left_knee)), (points6_coords[2]+10, points6_coords[3]+10), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    #         if points_cor8:
    #             cv.putText(frames, str(int(self.left_shoulder)), (points_cor8[2]+10, points_cor8[3]+10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    #     return self.left_elbow,self.left_hip,self.left_knee,self.left_shoulder







# LEFT_SIDE_VIEW, RIGHT_SIDE_VIEW, FORWARD_SIDE_VIEW, UP_SIDE_VIEW, DOWN_SIDE_VIEW = "LEFT", "RIGHT", "FORWARD", "UP", "DOWN"
     
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
#         self.sideView = None
    
#     def process_frame(self, image, draw=False):
#         start = time.time()
#         img_h, img_w, _ = image.shape
 
#         image = cv2.flip(image, 1)
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = self.face_mesh.process(rgb_image)
        
#         self.sideView = None
        
#         face_3d, face_2d = [], []
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 for idx, lm in enumerate(face_landmarks.landmark):
#                     if idx in [33, 263, 1, 61, 291, 199]:
#                         x, y = int(lm.x * img_w), int(lm.y * img_h)
#                         face_2d.append([x, y])
#                         face_3d.append([x, y, lm.z])
#                         if idx == 1:
#                             nose_2d = (x, y)
#                             nose_3d = (x, y, lm.z * 3000)
                
#                 face_2d = np.array(face_2d, dtype=np.float64)
#                 face_3d = np.array(face_3d, dtype=np.float64)

#                 focal_length = img_w
#                 cam_matrix = np.array([
#                     [focal_length, 0, img_w / 2],
#                     [0, focal_length, img_h / 2],
#                     [0, 0, 1]
#                 ])
#                 dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
#                 success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
#                 rmat, _ = cv2.Rodrigues(rot_vec)
#                 angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
#                 x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
                
#                 # Determine direction
#                 if y < -10:
#                     self.sideView = "Left"
#                 elif y > 10:
#                     self.sideView = "Right"
#                 elif x < -10:
#                     self.sideView = "Down"
#                 elif x > 10:
#                     self.sideView = "Up"
#                 else:
#                     self.sideView ="Forward"
                
#                 if draw:
#                     nose_3d_projection, _ = cv2.projectPoints(np.array([nose_3d]), rot_vec, trans_vec, cam_matrix, dist_matrix)
#                     p1 = (int(nose_2d[0]), int(nose_2d[1]))
#                     p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
#                     cv2.line(image, p1, p2, (0, 0, 255), 3)
#                     cv2.circle(image, p2, 5, (255, 0, 0), -1)
                    
#                     cv2.putText(image, f'Position: {self.sideView}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#                     cv2.putText(image, f'Pitch (x): {x:.2f}', (20, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
#                     cv2.putText(image, f'Yaw (y): {y:.2f}', (20, 130), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
#                     cv2.putText(image, f'Roll (z): {z:.2f}', (20, 160), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        
#         # if draw:
#         #     end = time.time()
#         #     fps = 1 / (end - start)
#         #     cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

#         return self.sideView
    
#     def main(self,image):

#         processed_frame = self.process_frame(image, draw=False)
#         cv2.imshow('HEAD POSE ESTIMATION', processed_frame)

#         return self.sideView
    
#     def run(self):
#         self.cap = cv2.VideoCapture(0)
#         while self.cap.isOpened():
#             success, frame = self.cap.read()
#             if not success:
#                 break
#             processed_frame, position = self.process_frame(frame, draw=True)
#             cv2.imshow('HEAD POSE ESTIMATION', processed_frame)
#             print(f'POSITION: {position}')
#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break
#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     estimator = HeadPoseEstimator()
#     estimator.run()

 

# import multiprocessing
# import pyttsx3
# import time
# from threading import Thread


# def threaded(fn):
#     def wrapper(*args, **kwargs):
#         thread = Thread(target=fn, args=args, kwargs=kwargs)
#         thread.start()
#         return thread
#     return wrapper

# def speak(phrase):
#     engine = pyttsx3.init()
#     engine.say(phrase)
#     engine.runAndWait()
#     engine.stop()

# def stop_speaker():
#     global term, t
#     if 't' in globals():
#         term = True
#         t.join()

# @threaded
# def manage_process(p):
#     global term
#     while p.is_alive():
#         if term:
#             p.terminate()
#             term = False
#         else:
#             continue

# def say(phrase):
#     global t, term
#     term = False
#     p = multiprocessing.Process(target=speak, args=(phrase,))
#     p.start()
#     t = manage_process(p)

# if _name_ == "_main_":
#     # First, we ensure that t is initialized before calling stop_speaker
#     stop_speaker()
#     say("this process is running right now Any answer regarding the given question. this process is running right now Any answer regarding the given question this process is running right now Any answer regarding the given question")
#     time.sleep(5)
#     stop_speaker()
#     say("It's sad because I like the libraries based approach of pyttsx module.")
#     time.sleep(3)
#     stop_speaker()

# import threading
# import pyttsx3 as pyt
# import time

# class voice_condition:

#     def __init__(self):

#         self.engine = pyt.init()
#         voices = self.engine.getProperty("voices")
#         self.engine.setProperty('voice',voices[1].id)
#         self.engine.setProperty("rate", 125)
        
#         self.voice_thread = None
#         self.voice_detect = False
#         self.stop_thread = None
#         # self.lock = threading.Lock
#         self.voice_stop = False

#     def voice(self,message):
            
#         if not self.voice_detect:
#             # start()
#             self.voice_detect = True
#             self.voice_thread = threading.Thread(target=self.speak,daemon=True, args=(message,))
#             self.voice_thread.start()

#     def speak(self,message):

#         # with lock:
#         self.engine.say(message)
#         self.engine.runAndWait()
#         self.voice_detect=False

#     def stop_voice(self):

#         if not self.voice_stop :
#             self.voice_stop = True
#             self.stop_thread = threading.Thread(target=self.engine.stop)
#             self.stop_thread.start()
#             # self.voice_detect = False
#             # threading.Thread()

#     def stop(self):

#         # with lock:
#         if self.engine:
#             self.engine.stop()
            
        

# detect = voice_condition()
# detect.voice("hi hello world welcome to the cricket Balanis, John Wiley, 1982, Chapter 9.41Numerical Electromagnetics Code – NEC-4 Method of Moments, Part I: User’s Manual, Part II: NEC Program Description - Theory, Part III: NEC Program description – Code, Gerald J.")
# time.sleep(5)
# detect.stop_voice()
# cont = True
# # time.sleep(2)
# if cont:
#     detect.voice(" A theoretical framework is a foundational review of existing theories that serves as a roadmap for developing the arguments you will use in your own work.")
#     time.sleep(5)
#     detect.stop_voice()



# used Method

    # def left_distance(self,frames,llist,left_hand,left_toe_point,face_points):

    #         x2,y2 ,z2= llist[left_toe_point][1:]

    #         self.head_points = self.head_point(frames=frames,lmlist=llist,points= face_points)
    #         self.head_x = self.head_points[0]
    #         self.head_y = self.head_points[1]
    #         self.head_z = self.head_points[2]

    #         cv.putText(frames,str(self.head_z),(140,150),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)

    #         left_points_x,left_points_y ,left_points_z= llist[left_hand][1:]

    #         self.head_distance = int(math.sqrt((self.head_y-y2) **2 + (self.head_x - x2) **2))
    #         cv.line(frames,(self.head_x,self.head_y),(x2,y2),(0,255,0),2)
    #         cv.putText(frames,str(self.head_distance),(x2-10,y2+10),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
    #         self.palm_distance = int(math.sqrt((left_points_y-y2)**2 +(left_points_x - x2) ** 2))
    #         cv.line(frames,(left_points_x,left_points_y),(x2,y2),(0,255,0),2)
    #         # cv.putText(frames,str(self.distance),(x2-10,y2+10),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)

    #         return self.palm_distance,self.head_distance,self.head_x,self.head_y
        
    # def right_distance(self,frames,llist,right_hand,right_toe_point,face_points):

    #     if len(llist) != 0:

    #         x2,y2,z2 = llist[right_toe_point][1:]

    #         self.head_points = self.head_point(frames=frames,lmlist=llist,points= face_points)
    #         self.head_x = self.head_points[0]
    #         self.head_y = self.head_points[1]
    #         self.head_z = self.head_points[2]
    #         cv.putText(frames,str(self.head_z),(140,150),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)

    #         right_points_x,right_points_y,right_points_z = llist[right_hand][1:]

    #         self.head_distance = int(math.sqrt((self.head_y-y2) **2 + (self.head_x - x2) **2))
    #         cv.line(frames,(self.head_x,self.head_y),(x2,y2),(0,255,0),2)
    #         # cv.putText(frames,str(self.head_distance),(x2-10,y2+10),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
    #         self.palm_distance = int(math.sqrt((right_points_y-y2)**2 +(right_points_x - x2) ** 2))
    #         cv.line(frames,(right_points_x,right_points_y),(x2,y2),(0,255,0),2)
    #         cv.putText(frames,str(self.palm_distance),(x2-10,y2+10),cv.FONT_HERSHEY_PLAIN,2,(0,0,255),3)

    #         return self.palm_distance,self.head_distance,self.head_x,self.head_y        


# this all one about calculate slope

# def get_Co_Ordinates(self, pose_landmarks, endPoint):
    #     if pose_landmarks is not None or pose_landmarks.landmark is not None:
    #         landmarks_list = [landmark for landmark in pose_landmarks.landmark]
    #         return landmarks_list[endPoint].x, landmarks_list[endPoint].y
        
    # def calculate_slope(self, point1, point2):
    #     if point1 is not None and point2 is not None:
    #         y = point2[1]-point1[1]
    #         x = point2[0]-point1[0]
    #         m = y/x
    #         angle = np.arctan(m)
    #         return int(math.degrees(angle))
        
    # def findSlope(self, endPoint1, endPoint2):
    #     if self.results.pose_landmarks:
    #         return abs(self.calculate_slope(point1=self.get_Co_Ordinates(pose_landmarks=self.results.pose_landmarks, endPoint=endPoint1),
    #                                                 point2=self.get_Co_Ordinates(pose_landmarks=self.results.pose_landmarks, endPoint=endPoint2))
    #                     )




import platform
import pyttsx3
import os
import threading
    

class VoicePlay():
    
    def _init_(self):
        super()._init_()
        self.system = platform.system()
        self.engine = None
        self.callBack = None
        self.opId = -1
        self.playingText = []
        self.isAudio_Playing_Completed = True
        self.isVoicePlaying = False
        if self.system == 'Windows':
            try:
                self.engine = pyttsx3.init()
                self.configureEngine(self.engine)
            except Exception as e:
                pass
                # print(f"Error initializing TTS engine on Windows: {e}")

    def setID(self, id):
        self.exerciseID = id
    
    def configureEngine(self, engine):
        """Configures the TTS engine for voice settings."""
        try:
            voices = engine.getProperty('voices')
            for voice in voices:
                # print(f"ID: {voice.id}, Name: {voice.name}, Language: {voice.languages}")
                
                if 'en_US' in voice.languages:
                    engine.setProperty('voice', voice.id)
                    break
        except KeyError as e:
            pass
            # print(f"Voice configuration error: {e}")
        except Exception as e:
            pass
            # print(f"Unexpected error during voice configuration: {e}")

    def playAudio(self, callBack, textToPlay, opId, play=False ):
        self.isVoicePlaying = True
        self.isAudio_Playing_Completed = False
        self.playingText = textToPlay
        self.callBack = callBack
        self.play=play
        self.opId = opId
        self.playVoiceBackground(play=self.play) 


    def playVoice(self, play):
        if not self.playingText:
            # print("Error: playingText is empty. No text to play.")
            return

        if self.system == 'Darwin':  # macOS
            try:
                self.length = 0
                if play:
                    for text_to_play in self.playingText :
                        self.length+=1
                        os.system(f'say "{text_to_play}"')
                          
                if self.callBack:
                    self.isVoicePlaying = False
                    self.isAudio_Playing_Completed = True
                    self.callBack(self.opId)
                else:
                    self.isVoicePlaying = False
                    self.isAudio_Playing_Completed = True
                    return
                
            except Exception as e:
                print(f"Error using system 'say' command on macOS: {e}")
        
        elif self.system == 'Windows' and self.engine is not None:  # Windows
            try:
                self.length = 0
                if play:
                    for text_to_play in self.playingText :
                        self.length+=1
                        self.engine.say(text_to_play)
                        self.engine.runAndWait()
                
                if self.callBack:
                    self.isVoicePlaying = False
                    self.isAudio_Playing_Completed = True
                    self.callBack(self.opId)
                else:
                    self.isVoicePlaying = False
                    self.isAudio_Playing_Completed = True
                    return
                    
            except Exception as e:
                pass
                # print(f"Error during speech playback on Windows: {e}")
        
        elif self.system == 'Linux':  # Linux
            try:
                self.length = 0
                if play:
                    for text_to_play in self.playingText :
                        self.length+=1
                        self.isAudio_Playing_Completed = False
                        os.system(f'espeak "{text_to_play}"')
                        
                self.isAudio_Playing_Completed = True
                if self.callBack:
                    self.callBack(self.opId)
                else:
                    return
            except Exception as e:
                pass
                # print(f"Error using system 'espeak' command on Linux: {e}")
        else:
            pass
            # print(f"Unsupported operating system: {self.system}")

    def playVoiceBackground(self, play):
        """Plays the voice in the background asynchronously."""
        threading.Thread(target = self.playVoice, 
                         daemon = True,
                         args = (play,)
                        ).start()
        
    def stopEngine(self):
        self.engine.stop()
    
    
    def startEngine(self):
        try:
            
            self.engine = pyttsx3.init()
            self.configureEngine(self.engine)
            return True
        except Exception as e:
            pass
            # print(f"Error initializing TTS engine on Windows: {e}")













#vajrasana right side wrong check:
# knee_correct = ( self.right_knee) 
#         if knee_correct:
#             right_knee_correct1 = (self.right_knee and 31 <= self.right_knee <= 180)
            
#             if right_knee_correct1:
#                 # print('right knee detected')
#                 if self.right_count == 0:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["please keep your right leg back"],llist=llist)
#                     self.right_count += 1

#                 elif self.right_count == 1:
#                      self.all_methods.reset_voice()
#                      self.all_methods.play_voice(["keep your right leg back and fold"],llist=llist)
#                      self.right_count += 1

#                 elif self.right_count == 2:
#                      self.all_methods.reset_voice()
#                      self.all_methods.play_voice(["please fold your right leg like on reference video"],llist=llist)
#                      self.right_count -= 1

#             else:
#                 self.right_count = 0
#         # else:
#         #     self.right_count = 0
#             # print('right knee not detected')
            
#         left_knee1_correct = (self.left_knee1 )
#         if left_knee1_correct:
#             left_knee1_correct = (self.left_knee1 and 31 <= self.left_knee1 <= 180)
#             if left_knee1_correct:

#                 if self.right_count == 0:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["please keep left leg your back"],llist=llist)
#                     self.right_count += 1

#                 elif self.right_count == 1:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["keep your left leg back and fold"],llist=llist)
#                     self.right_count += 1
                
#                 elif self.right_count == 2:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["please fold your left leg like on reference video"],llist=llist)
#                     self.right_count -= 1

#             else:
#                 self.right_count = 0
#         # else:
#         #     self.right_count = 0

#         elbow_correct = (self.right_elbow)
#         if elbow_correct:
#             right_elbow_correct_down = (self.right_elbow and 0 <= self.right_elbow <= 149)
#             if right_elbow_correct_down:
#                 if self.right_count == 0:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice([" please keep your elbow staright"],llist=llist)
#                     self.right_count += 1

#                 elif self.right_count == 1:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["keep your hands straight and touch palm to knees"],llist=llist)
#                     self.right_count = 0

#             else:
#                 self.right_count -= 1
#         # else:
#         #     self.right_count = 0


#         shoulder_correct = (self.right_shoulder)
#         if shoulder_correct:
#             right_shoulder_correct_up = (self.right_shoulder and 41 <= self.right_shoulder <= 180)
#             if right_shoulder_correct_up:
#                 if self.right_count == 0:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["please slight close your shoulders"],llist=llist)
#                     self.right_count += 1

#                 elif self.right_count == 1:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["close slight your shoulders"],llist=llist)
#                     self.right_count = 0

#             else:
#                 self.right_count = 0

                

#             right_shoulder_correct_down = (self.right_shoulder and 0 <= self.right_shoulder <= 26)
#             if right_shoulder_correct_down:
#                 if self.right_count == 0:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["please slight open your shoulders"],llist=llist)
#                     self.right_count += 1

#                 elif self.right_count == 1:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["open slight your shoulders"],llist=llist)
#                     self.right_count = 0

#             else:
#                 self.right_count = 0
        
#         # else:
#         #     self.right_count = 0

#         hip_correct = ( self.right_hip)
#         if hip_correct:
#             right_hip_correct = (self.right_hip and 0 <= self.right_hip <= 79)
#             if right_hip_correct:
#                 if self.right_count == 0:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["keep your upper body straight "],llist=llist)
#                     self.right_count += 1

#                 elif self.right_count == 1:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["please keep your spinal cord straight"],llist=llist)
#                     self.right_count = 0

#             else:
#                 self.right_count = 0

            

#             right_hip_correct = (self.right_hip and 116 <= self.right_hip <= 180)
#             if right_hip_correct:
#                 if self.right_count == 0:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["please keep your upper body staright"],llist=llist)
#                     self.right_count += 1

#                 elif self.right_count == 1:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["keep your spinal cord straight"],llist=llist)
#                     self.right_count = 0

#             else:
#                 self.right_count = 0
#         # else:
#         #     self.right_count = 0

#         head_correct = (self.head)
#         if head_correct:
#             up_correct =  (self.head_position and self.head_position != "Right")
#             if up_correct:
#                 if self.right_count == 0:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["you should to face your head to right side"],llist=llist)
#                     self.right_count += 1

#                 elif self.right_count == 1:
#                     self.all_methods.reset_voice()
#                     self.all_methods.play_voice(["turn your head right side"],llist=llist)
#                     self.right_count = 0

#             else:
#                 self.right_count = 0  



#left vajrasana

# knee_correct = (self.left_knee )
        # if knee_correct:
           
        #     left_knee_correct = (self.left_knee and 31 <= self.left_knee <= 180)
        #     if left_knee_correct:

        #         # if self.left_knee > 31:
                    
        #         if self.l_k_c == 0:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please keep your left leg back"],llist=llist)
        #             self.l_k_c += 1

        #         elif self.l_k_c == 1:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["fold your left leg"],llist=llist)
        #             self.l_k_c += 1

        #         elif self.l_k_c == 2:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please fold your left leg like on reference video"],llist=llist)
        #             self.l_k_c -= 1

        #     else:
        #         self.l_k_c = 0

        # # else:
        # #     # print("left knee not detected")
        # #     self.l_k_c = 0
        #     # self.left_count = 1

        # right_knee1_correct = (self.right_knee1 )
        # if right_knee1_correct:
        #     right_knee1_correct1 = (self.right_knee1 and 31 <= self.right_knee1 <= 180)
        #     if right_knee1_correct1:

        #         if self.r_k_c1 == 0:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please keep your right leg back"],llist=llist)
        #             self.r_k_c1 += 1

        #         elif self.r_k_c1 == 1:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["fold your right leg"],llist=llist)
        #             self.r_k_c1 += 1
                
        #         elif self.r_k_c1 == 2:
        #                 self.all_methods.reset_after_40_sec()
        #                 self.all_methods.play_after_40_sec(["please fold your left leg like on reference video"],llist=llist)
        #                 self.r_k_c1 -= 1

        #     else:
        #         self.r_k_c1 = 0

        # # if not right_knee1_correct:
        # #     self.r_k_c1 = 0

        # #this one is for elbow
        # elbow_correct = (self.left_elbow)
        # if elbow_correct:
        #     left_elbow_correct = (self.left_elbow and 0 <= self.left_elbow <= 149)
        #     if left_elbow_correct:
                
        #         if self.l_eu_c == 0:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please keep your elbows straight"],llist=llist)
        #             self.l_eu_c += 1

        #         elif self.l_eu_c == 1:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["touch your hands to knees"],llist=llist)
        #             self.l_eu_c += 1

        #         elif self.l_eu_c == 2:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["straight your elbows and touch your hands to knees"],llist=llist)
        #             self.l_eu_c -= 1

        #     else:
        #         self.l_eu_c = 0

        # # if not elbow_correct:
        # #     self.l_eu_c = 0

        # #this one is for shoulder
        # shoulder_correct =  self.left_shoulder
        # if shoulder_correct:
        #     left_shoulder_correct_up = (self.left_shoulder and 41 <= self.left_shoulder <= 180)
        #     if left_shoulder_correct_up:

        #         if self.l_sd_c == 0:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please fold your shoulders"],llist=llist)
        #             self.l_sd_c += 1

        #         elif self.l_sd_c == 1:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["your shoulders slight open please fold"],llist=llist)
        #             self.l_sd_c -= 1

        #     else:
        #         self.l_sd_c = 0

        #     left_shoulder_correct_down = (self.left_shoulder and 0 <= self.left_shoulder <= 26)
        #     if left_shoulder_correct_down:
                
        #         if self.l_su_c == 0:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please open your shoulders"],llist=llist)
        #             self.l_su_c += 1

        #         elif self.l_su_c == 1:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["your shoulders slight close please open"],llist=llist)
        #             self.l_su_c -= 1

        #     else:
        #         self.l_su_c = 0

        # # if not shoulder_correct:
        # #     self.l_sd_c = 0
        # #     self.l_su_c = 0
                    

        # #this one for both sides of hips
        # hip_correct = (self.left_hip)
        # if hip_correct:
        #     left_hip_correct1 = (self.left_hip and 0 <= self.left_hip <= 79)
        #     if left_hip_correct1:

        #         if self.l_hip_d_c == 0:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please keep your upper body staright"],llist=llist)
        #             self.l_hip_d_c += 1

        #         elif self.l_hip_d_c == 1:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please be staright your spinal cord"],llist=llist)
        #             self.l_hip_d_c -= 1

        #     else:
        #         self.l_hip_d_c = 0
                    

        #     left_hip_correct = (self.left_hip and 116 <= self.left_hip <= 180)
        #     if left_hip_correct:
        #         if self.l_hip_up_c == 0:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please keep your upper body staright"],llist=llist)
        #             self.l_hip_up_c += 1

        #         elif self.l_hip_up_c == 1:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please be staright your spinal cord"],llist=llist)
        #             self.l_hip_up_c -= 1

        #     else:
        #         self.l_hip_up_c = 0

        # # if not hip_correct:
        # #     self.l_hip_d_c = 0
        # #     self.l_hip_up_c = 0


        # head_correct = (self.head)
        # if head_correct:
        #     up_correct =  (self.head_position and self.head_position != "Left")
            
        #     if up_correct:
        #         if self.l_h == 0:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please turn your head left side"],llist=llist)
        #             self.l_h += 1

        #         elif self.l_h == 1:
        #             self.all_methods.reset_after_40_sec()
        #             self.all_methods.play_after_40_sec(["please face your head left"],llist=llist)
        #             self.l_h -= 1

        #     else:
        #         self.l_h = 0
        # if not head_correct:
        #     self.l_h = 0