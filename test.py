import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from src.pose_estimator import PoseEstimator
from utils.video_utils import read_video, save_video
from src.trajectory_traker import TrajectoryAnalyzer

def draw_trajectory_with_path(frame, wrist_points, current_idx, history_length=30):
    """
    Draw trajectory path with fading effect and current position marker
    """
    frame_with_traj = frame.copy()
    
    # Draw connecting lines for recent history
    start_idx = max(0, current_idx - history_length)
    
    for i in range(start_idx, current_idx - 1):
        if i < len(wrist_points) and (i + 1) < len(wrist_points):
            pt1 = (int(wrist_points[i][0]), int(wrist_points[i][1]))
            pt2 = (int(wrist_points[i + 1][0]), int(wrist_points[i + 1][1]))
            
            # Fade older points (alpha blending)
            alpha = (i - start_idx) / history_length
            color = (0, int(255 * (1 - alpha)), int(255 * alpha))  # Blue to yellow
            
            cv2.line(frame_with_traj, pt1, pt2, color, 2)
    
    # Draw current position
    if current_idx < len(wrist_points):
        x, y = int(wrist_points[current_idx][0]), int(wrist_points[current_idx][1])
        
        # Current position marker
        cv2.circle(frame_with_traj, (x, y), 8, (0, 0, 255), -1)  # Red circle
        cv2.circle(frame_with_traj, (x, y), 10, (255, 255, 255), 2)  # White border
        
        # Add coordinate text
        cv2.putText(frame_with_traj, f"({x},{y})", (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame_with_traj

def main():
    video_path = "/Users/dhruvkulshrestha/Desktop/Cricket_bowling_analysis/input_video/cut_video.mp4"
    output_path = "/Users/dhruvkulshrestha/Desktop/Cricket_bowling_analysis/output_video/wrist_trajectory.mp4"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        return
    
    print(f"🎥 Processing video: {os.path.basename(video_path)}")
    
    try:
        # 1️⃣ Initialize pose estimator
        pose_estimator = PoseEstimator(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )

        # 2️⃣ Process video
        print("🔄 Extracting pose data...")
        pose_data = pose_estimator.process_video(video_path)
        
        print(f"✅ FPS: {pose_data['fps']:.2f}")
        print(f"✅ Total frames: {pose_data['frame_count']}")
        
        frames_data = pose_data["frames_data"]
        
        # Count frames with pose
        frames_with_pose = sum(1 for f in frames_data if f.get('keypoints'))
        print(f"✅ Frames with pose detected: {frames_with_pose}/{pose_data['frame_count']}")

        # 3️⃣ Extract trajectory
        trajectory = pose_estimator.get_wrist_elbow_trajectory(
            frames_data=frames_data,
            bowling_arm="right"
        )

        wrist_traj = trajectory["wrist_trajectory"]
        timestamps = trajectory["timestamps"]
        
        print(f"✅ Wrist trajectory shape: {wrist_traj.shape}")
        
        if wrist_traj.shape[0] == 0:
            print("❌ No wrist points detected!")
            return

        # 4️⃣ Read raw frames
        print("📖 Reading video frames...")
        frames = read_video(video_path)
        
        if len(frames) == 0:
            print("❌ No frames read from video!")
            return

        # 5️⃣ Draw trajectory on frames
        print("🎨 Drawing trajectory...")
        output_frames = []
        
        for i, frame in enumerate(frames):
            if i < len(wrist_traj):
                # Draw trajectory with path
                frame_with_traj = draw_trajectory_with_path(
                    frame, wrist_traj, i, history_length=20
                )
                
                # Add frame info
                h, w = frame.shape[:2]
                cv2.putText(frame_with_traj, f"Frame: {i}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_with_traj, f"Points: {len(wrist_traj)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                output_frames.append(frame_with_traj)
            else:
                # Just use original frame if no trajectory data
                output_frames.append(frame)
        
        # 6️⃣ Save output video
        print("💾 Saving output video...")
        fps = pose_data["fps"]
        save_video(output_frames, output_path, fps=fps)
        
        print(f"✅ Output video saved to: {output_path}")
        print(f"✅ Output frames: {len(output_frames)}")
        
        # 7️⃣ Create a simple plot of trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(wrist_traj[:, 0], wrist_traj[:, 1], 'b-', linewidth=2, label='Wrist Path')
        plt.scatter(wrist_traj[0, 0], wrist_traj[0, 1], color='green', s=100, label='Start')
        plt.scatter(wrist_traj[-1, 0], wrist_traj[-1, 1], color='red', s=100, label='End')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Wrist Trajectory Path')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Match video coordinates
        
        plot_path = output_path.replace('.mp4', '_trajectory_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Trajectory plot saved to: {plot_path}")
        
        # Show stats
        print("\n📊 TRAJECTORY STATS:")
        print(f"   Total points: {len(wrist_traj)}")
        print(f"   X range: [{wrist_traj[:, 0].min():.1f}, {wrist_traj[:, 0].max():.1f}]")
        print(f"   Y range: [{wrist_traj[:, 1].min():.1f}, {wrist_traj[:, 1].max():.1f}]")
        print(f"   Duration: {timestamps[-1]:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()