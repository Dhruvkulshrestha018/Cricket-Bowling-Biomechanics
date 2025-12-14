import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple
import json

class PoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5):
        """
        Initialize MediaPipe Pose estimator for cricket bowling analysis
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Keypoint indices for cricket bowling analysis
        self.keypoint_indices = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24
        }
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process video and extract pose keypoints frame by frame
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Process with MediaPipe
            results = self.pose.process(frame_rgb)
            
            frame_data = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'keypoints': {},
                'landmarks': None
            }
            
            if results.pose_landmarks:
                # Extract 2D keypoints
                landmarks = results.pose_landmarks.landmark
                frame_data['landmarks'] = landmarks
                
                h, w = frame.shape[:2]
                
                for name, idx in self.keypoint_indices.items():
                    landmark = landmarks[idx]
                    frame_data['keypoints'][name] = {
                        'x': landmark.x * w,
                        'y': landmark.y * h,
                        'z': landmark.z,  # Relative depth
                        'visibility': landmark.visibility
                    }
            
            frames_data.append(frame_data)
            frame_count += 1
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'video_dimensions': (int(cap.get(3)), int(cap.get(4))),
            'frames_data': frames_data
        }
    
    def get_wrist_elbow_trajectory(self, frames_data: List[Dict], bowling_arm: str = 'right') -> Dict:
        """
        Extract smooth trajectory of wrist and elbow
        """
        wrist_points = []
        elbow_points = []
        timestamps = []
        
        for frame in frames_data:
            if 'keypoints' in frame and frame['keypoints']:
                wrist_key = f'{bowling_arm}_wrist'
                elbow_key = f'{bowling_arm}_elbow'
                
                if wrist_key in frame['keypoints'] and elbow_key in frame['keypoints']:
                    wrist = frame['keypoints'][wrist_key]
                    elbow = frame['keypoints'][elbow_key]
                    
                    wrist_points.append([wrist['x'], wrist['y']])
                    elbow_points.append([elbow['x'], elbow['y']])
                    timestamps.append(frame['timestamp'])
        
        return {
            'wrist_trajectory': np.array(wrist_points),
            'elbow_trajectory': np.array(elbow_points),
            'timestamps': np.array(timestamps),
            'bowling_arm': bowling_arm
        }