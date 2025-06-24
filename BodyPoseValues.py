import numpy as np
import mediapipe as mp
import math
from cv2 import line
class BodyPoseValues:


    def process_pose_landmarks(self, pose_landmarks,width,height, image):
        self.image = image
        self.width = width
        self.height = height
        Index_Nose = mp.solutions.pose.PoseLandmark.NOSE.value
        Index_LeftEye = mp.solutions.pose.PoseLandmark.LEFT_EYE.value
        Index_RightEye = mp.solutions.pose.PoseLandmark.RIGHT_EYE.value
        Index_left_shoulder = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        Index_left_elbow = mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value
        Index_left_wrist = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        Index_left_hip = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
        Index_left_knee = mp.solutions.pose.PoseLandmark.LEFT_KNEE.value
        Index_left_ankle = mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value
        Index_left_ear = mp.solutions.pose.PoseLandmark.LEFT_EAR.value
        Index_left_heel = mp.solutions.pose.PoseLandmark.LEFT_HEEL.value
        Index_left_foot = mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value

        Index_right_shoulder = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
        Index_right_elbow = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value
        Index_right_wrist = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value
        Index_right_hip = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
        Index_right_knee = mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value
        Index_right_ankle = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value
        Index_right_ear = mp.solutions.pose.PoseLandmark.RIGHT_EAR.value
        Index_right_heel = mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value
        Index_right_foot = mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value


        

        landmarks_list = [landmark for landmark in pose_landmarks.landmark]
        self.nose = landmarks_list[Index_Nose].x, landmarks_list[Index_Nose].y
        self.left_eye = landmarks_list[Index_LeftEye].x, landmarks_list[Index_LeftEye].y
        self.right_eye = landmarks_list[Index_RightEye].x, landmarks_list[Index_RightEye].y

        self.left_elbow = landmarks_list[Index_left_elbow].x, landmarks_list[Index_left_elbow].y
        self.left_shoulder = landmarks_list[Index_left_shoulder].x, landmarks_list[Index_left_shoulder].y
        self.left_hip = landmarks_list[Index_left_hip].x, landmarks_list[Index_left_hip].y
        self.left_knee = landmarks_list[Index_left_knee].x, landmarks_list[Index_left_knee].y
        self.left_ankle = landmarks_list[Index_left_ankle].x, landmarks_list[Index_left_ankle].y
        self.left_wrist = landmarks_list[Index_left_wrist].x, landmarks_list[Index_left_wrist].y
        self.left_ear = landmarks_list[Index_left_ear].x, landmarks_list[Index_left_ear].y
        self.left_heel = landmarks_list[Index_left_heel].x, landmarks_list[Index_left_heel].y
        self.left_foot = landmarks_list[Index_left_foot].x, landmarks_list[Index_left_foot].y

        self.right_elbow = landmarks_list[Index_right_elbow].x, landmarks_list[Index_right_elbow].y
        self.right_shoulder = landmarks_list[Index_right_shoulder].x, landmarks_list[Index_right_shoulder].y
        self.right_hip = landmarks_list[Index_right_hip].x, landmarks_list[Index_right_hip].y
        self.right_knee = landmarks_list[Index_right_knee].x, landmarks_list[Index_right_knee].y
        self.right_ankle = landmarks_list[Index_right_ankle].x, landmarks_list[Index_right_ankle].y
        self.right_wrist = landmarks_list[Index_right_wrist].x, landmarks_list[Index_right_wrist].y
        self.right_ear = landmarks_list[Index_right_ear].x, landmarks_list[Index_right_ear].y
        self.right_heel = landmarks_list[Index_right_heel].x, landmarks_list[Index_right_heel].y
        self.right_foot = landmarks_list[Index_right_foot].x, landmarks_list[Index_right_foot].y

        self.Left_ShoulderAngle = self._calculate_angle(self.left_elbow, self.left_shoulder, self.left_hip)
        self.left_shoulder_center_midpoint = ((self.left_shoulder[0] + self.right_shoulder[0]) / 2,
                                              (self.left_shoulder[1] + self.right_shoulder[1]) / 2)

        self.right_shoulder_center_midpoint = ((self.right_shoulder[0] + self.left_shoulder[0]) / 2,
                                               (self.right_shoulder[1] + self.left_shoulder[1]) / 2)
        self.Right_ShoulderAngle = self._calculate_angle(self.right_elbow, self.right_shoulder, self.right_hip)
        self.Left_ElbowAngle = self._calculate_angle(self.left_wrist, self.left_elbow, self.left_shoulder)
        self.Right_ElbowAngle = self._calculate_angle(self.right_wrist, self.right_elbow, self.right_shoulder)
        self.left_knee_angle = self._calculate_angle(self.left_hip, self.left_knee, self.left_ankle)
        self.right_knee_angle = self._calculate_angle(self.right_hip, self.right_knee, self.right_ankle)
        self.Left_HipAngle = self._calculate_angle(self.left_shoulder, self.left_hip, self.left_knee)
        self.Right_HipAngle = self._calculate_angle(self.right_shoulder, self.right_hip, self.right_knee)
        
        #Reverse Crunches
        self.angle_at_left_hip = self.calculate_angle_360(self.left_ankle,
                                            self.left_hip
                                            ,self.left_shoulder)
        self.angle_at_right_hip = self.calculate_angle_360(self.right_ankle,
                                            self.right_hip
                                            ,self.right_shoulder)
        """Scope  between the joints connecting any two landmark point"""
        # Left side
        self.left_forearm_scope = abs(self.calculate_slope(self.left_wrist, self.left_elbow))
        self.left_upper_arm_scope = abs(self.calculate_slope(self.left_elbow, self.left_shoulder))
        self.left_fullhand_scope = abs(self.calculate_slope(self.left_shoulder, self.left_wrist))
        self.left_side_scope = abs(self.calculate_slope(self.left_shoulder, self.left_hip))
        self.left_thigh_scope = abs(self.calculate_slope(self.left_hip, self.left_knee))
        self.left_lower_leg_scope = abs(self.calculate_slope(self.left_knee, self.left_ankle))
        self.left_foot_scope = abs(self.calculate_slope(self.left_heel, self.left_foot))
        self.left_back_scope = abs(abs(self.calculate_slope(self.left_hip, self.left_shoulder)))

        #ELbow Plank
        self.left_knee_ear_scope =  abs(self.calculate_slope(self.left_knee, self.left_ear))
        self.right_knee_ear_scope =  abs(self.calculate_slope(self.right_knee, self.right_ear))
        self.shoulder_scope = abs(self.calculate_slope(self.right_shoulder, self.left_shoulder))

        # Right side
        self.right_forearm_scope = abs(self.calculate_slope(self.right_wrist, self.right_elbow))
        self.right_upper_arm_scope = abs(self.calculate_slope(self.right_elbow, self.right_shoulder))
        self.right_fullhand_scope = abs(self.calculate_slope(self.right_shoulder, self.right_wrist))
        self.right_side_scope = abs(self.calculate_slope(self.right_shoulder, self.right_hip))
        self.right_thigh_scope = abs(self.calculate_slope(self.right_hip, self.right_knee))
        self.right_lower_leg_scope = abs(self.calculate_slope(self.right_knee, self.right_ankle))
        self.right_foot_scope = abs(self.calculate_slope(self.right_heel, self.right_foot))
        self.right_back_scope = abs(self.calculate_slope(self.right_hip, self.right_shoulder))

        
        self.left_ear_coord = tuple(np.multiply(self.left_ear,[width,height]).astype(int))
        self.right_ear_coord = tuple(np.multiply(self.right_ear,[width,height]).astype(int))
        self.left_shoulder_coord = tuple(np.multiply(self.left_shoulder,[width,height]).astype(int))
        self.right_shoulder_coord = tuple(np.multiply(self.right_shoulder,[width,height]).astype(int))
        self.left_hip_coord = tuple(np.multiply(self.left_hip,[width,height]).astype(int))
        self.right_hip_coord = tuple(np.multiply(self.right_hip,[width,height]).astype(int))
        self.left_knee_coord = tuple(np.multiply(self.left_knee,[width,height]).astype(int))
        self.right_knee_coord = tuple(np.multiply(self.right_knee,[width,height]).astype(int))
        self.left_heel_coord = tuple(np.multiply(self.left_heel,[width,height]).astype(int))
        self.right_heel_coord = tuple(np.multiply(self.right_heel,[width,height]).astype(int))
        self.left_foot_coord = tuple(np.multiply(self.left_foot,[width,height]).astype(int))
        self.right_foot_coord = tuple(np.multiply(self.right_foot,[width,height]).astype(int))
        self.left_ankle_coord = tuple(np.multiply(self.left_ankle,[width,height]).astype(int))
        self.right_ankle_coord = tuple(np.multiply(self.right_ankle,[width,height]).astype(int))
        self.left_elbow_coord = tuple(np.multiply(self.left_elbow,[width,height]).astype(int))
        self.right_elbow_coord = tuple(np.multiply(self.right_elbow,[width,height]).astype(int))
        self.left_wrist_coord = tuple(np.multiply(self.left_wrist,[width,height]).astype(int))
        self.right_wrist_coord = tuple(np.multiply(self.right_wrist,[width,height]).astype(int))

        self.midpoint_hip_coord = tuple(np.multiply(self.midpoint(self.left_hip,self.right_hip),[width,height]).astype(int))
        self.mid_hip_angle_to_foots = self._calculate_angle(self.right_ankle_coord,self.midpoint_hip_coord,self.left_ankle_coord)
        
        self.userDistanceFromCamera = self.personDistanceFromCamera(head=self.nose, left_ankle=self.left_ankle, right_ankle=self.right_ankle, 
                                                                    image_height=height, image_width=width
                                                                    )
        self.standing_OR_sitting = self.is_person_standing_OR_sitting(shoulder_left=self.left_shoulder, shoulder_right=self.right_shoulder,
                                                                      hip_left=self.left_hip, hip_right=self.right_hip,
                                                                      knee_left=self.left_knee, knee_right=self.right_knee,
                                                                      ankle_left=self.left_ankle, ankle_right=self.right_ankle)
        
        self.userSittingBodyOrientation = self.userBodyOrientation_whenSitting(nose=landmarks_list[Index_Nose].z, left_shoulder=landmarks_list[Index_left_shoulder].z, right_shoulder=landmarks_list[Index_right_shoulder].z)
        
        self.leftLayDown = self.check_same_y_axis(shoulder=self.left_shoulder, hip=self.left_hip, knee=self.left_knee)
        self.rightLayDown = self.check_same_y_axis(shoulder=self.right_shoulder, hip=self.right_hip, knee=self.right_knee)
        self.shoulders_Y_axis = self.check_same_y_axis_between_twoPoints(point1=self.left_shoulder_coord, point2=self.right_shoulder_coord)
        
        if int(self.nose[0]*width) in range(int(width/2)):
            self.one_part = True
            user_in = "First Half (R S V)"
        else:
            user_in = "SecondHalf (L S V)"
            self.one_part = False 




    def _calculate_angle(self, point1, center, point2,  color = (0,255,0), thickness = 1, draw=False):
        vector1 = np.array([point1[0] - center[0], point1[1] - center[1]])
        vector2 = np.array([point2[0] - center[0], point2[1] - center[1]])

        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if norm_product == 0:
            return 0  # Return 0 if division by zero would occur

        dot_product = np.dot(vector1, vector2)
        cos_theta = dot_product / norm_product
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        if draw:
            line(img = self.image, pt1 = tuple(np.multiply(point1,[self.width,self.height]).astype(int)), pt2 = tuple(np.multiply(center,[self.width,self.height]).astype(int)), color = color, thickness = thickness)
            line(img = self.image, pt1 = tuple(np.multiply(center,[self.width,self.height]).astype(int)), pt2 = tuple(np.multiply(point2,[self.width,self.height]).astype(int)), color = color, thickness = thickness)

        return int(angle_deg)

    def calculate_slope(self, point1, point2):
        y = point2[1]-point1[1]
        x = point2[0]-point1[0]
        m = y/x
        angle = np.arctan(m)
        return int(math.degrees(angle))
    
    def calculate_angle_360(self,a, b, c):
        a = np.array(a)  
        b = np.array(b)  
        c = np.array(c)  
        
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        # if angle > 180.0:
        #     angle = 360.0 - angle
        return int(angle)
    
    
    def midpoint(self,p1,p2):
        midpoint_x = (p1[0] + p2[0]) / 2
        midpoint_y = (p1[1] + p2[1]) / 2
        
        mid = (midpoint_x, midpoint_y)
        return mid
        
        # mid = [int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2)]
        # return mid
        # Calculate the midpoint between the shoulders
        
        

    def personDistanceFromCamera(self, head, left_ankle, right_ankle, image_height, image_width,  KNOWN_HEIGHT=1.7, FOCAL_LENGTH=615):
        
        head_y = head[1]* image_height 
        left_ankle_y = left_ankle[1]* image_height
        right_ankle_y = right_ankle[1]* image_height
        foot_y = max(left_ankle_y, right_ankle_y)

        pixel_height = abs(foot_y - head_y)

        # Distance estimation
        if pixel_height > 0:
            self.userDistanceFromCamera =  (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height
            
        return self.userDistanceFromCamera
           
            
    def check_same_y_axis(self, shoulder, hip, knee, tolerance=0.05):
        # Get the Y-coordinates of each point
        y_shoulder = shoulder[1]
        y_hip = hip[1]
        y_knee = knee[1]
        
        # Check if the absolute difference between Y-coordinates is within tolerance
        if abs(y_shoulder - y_hip) <= tolerance and abs(y_hip - y_knee) <= tolerance:
            return True
        return False
    
    
    def check_same_y_axis_between_twoPoints(self, point1, point2, tolerance=40):
        point1_y = point1[1]
        point2_y = point2[1]
        if abs(point1_y - point2_y) <= tolerance :
            return True
        return False
    
    def is_person_standing_OR_sitting(self, shoulder_left, shoulder_right, hip_left, hip_right, ankle_left, ankle_right, knee_left, knee_right,
                                      Y_THRESHOLD = 0.05, ANGLE_THRESHOLD = 160, ):
            
            # Calculate the average Y-values for shoulders, hips, and ankles
            avg_shoulder_y = (shoulder_left[1] + shoulder_right[1]) / 2
            avg_hip_y = (hip_left[1] + hip_right[1]) / 2
            avg_ankle_y = (ankle_left[1] + ankle_right[1]) / 2
            
            # Calculate the angle between shoulder, hip, and knee
            left_a = [shoulder_left[0] - hip_left[0], shoulder_left[1] - hip_left[1]]
            left_b = [knee_left[0] - hip_left[0], knee_left[1] - hip_left[1]]
            left_dot_product = left_a[0] * left_b[0] + left_a[1] * left_b[1]
            left_magnitude_a = math.sqrt(left_a[0]**2 + left_a[1]**2)
            left_magnitude_b = math.sqrt(left_b[0]**2 + left_b[1]**2)
            left_cos_angle = left_dot_product / (left_magnitude_a * left_magnitude_b)
            # Left_Angle:
            left_angle = math.acos(left_cos_angle) * (180.0 / math.pi)
            
            
            right_a = [shoulder_right[0] - hip_right[0], shoulder_right[1] - hip_right[1]]
            right_b = [knee_right[0] - hip_right[0], knee_right[1] - hip_right[1]]
            right_dot_product = right_a[0] * right_b[0] + right_a[1] * right_b[1]
            right_magnitude_a = math.sqrt(right_a[0]**2 + right_a[1]**2)
            right_magnitude_b = math.sqrt(right_b[0]**2 + right_b[1]**2)
            right_cos_angle = right_dot_product / (right_magnitude_a * right_magnitude_b)
            # Right_Angle:
            right_angle = math.acos(right_cos_angle) * (180.0 / math.pi)

            # Determine the position (standing or sitting) based on Y-values and angles
            if avg_shoulder_y < avg_hip_y - Y_THRESHOLD and avg_hip_y < avg_ankle_y - Y_THRESHOLD and (left_angle > ANGLE_THRESHOLD and right_angle > ANGLE_THRESHOLD):
                self.POSITION = 1
            elif avg_ankle_y > avg_hip_y + Y_THRESHOLD > avg_shoulder_y and (left_angle < ANGLE_THRESHOLD or right_angle < ANGLE_THRESHOLD):
                self.POSITION = 2
            else:
                self.POSITION = None  # In case the posture is ambiguous or body parts are not well detected
                
            return self.POSITION
      
      
    def userBodyOrientation_whenSitting(self, nose, left_shoulder, right_shoulder, z_threshold=0.2):
       
        # Calculate the z-axis differences between the shoulders
        z_shoulder_diff = left_shoulder - right_shoulder

        # Calculate the z-axis difference between the nose and the center of the shoulders
        z_head_diff = nose - (left_shoulder + right_shoulder) / 2

        # If both shoulders are about the same depth, and head is aligned with shoulders, facing forward
        if abs(z_shoulder_diff) < z_threshold and abs(z_head_diff) < z_threshold:
            return 0
        # If left shoulder is closer (negative diff), user is turned to their right
        elif z_shoulder_diff < -z_threshold:
            return -1
        # If right shoulder is closer (positive diff), user is turned to their left
        elif z_shoulder_diff > z_threshold:
            return 1

        
