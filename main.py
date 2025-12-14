import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from src.pose_estimator import PoseEstimator
from src.trajectory_traker import TrajectoryAnalyzer  # <-- New import
from utils.video_utils import read_video, save_video

def draw_trajectory_with_path(frame, wrist_points, current_idx, history_length=30):
    """Draw trajectory path with fading effect"""
    frame_with_traj = frame.copy()
    
    start_idx = max(0, current_idx - history_length)
    
    for i in range(start_idx, current_idx - 1):
        if i < len(wrist_points) and (i + 1) < len(wrist_points):
            pt1 = (int(wrist_points[i][0]), int(wrist_points[i][1]))
            pt2 = (int(wrist_points[i + 1][0]), int(wrist_points[i + 1][1]))
            
            # Fade older points
            alpha = (i - start_idx) / history_length
            color = (0, int(255 * (1 - alpha)), int(255 * alpha))
            
            cv2.line(frame_with_traj, pt1, pt2, color, 2)
    
    # Current position marker
    if current_idx < len(wrist_points):
        x, y = int(wrist_points[current_idx][0]), int(wrist_points[current_idx][1])
        cv2.circle(frame_with_traj, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(frame_with_traj, (x, y), 10, (255, 255, 255), 2)
    
    return frame_with_traj

def add_biomechanics_overlay(frame, metrics):
    """Add biomechanics metrics to video frame"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Metrics box background
    cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (300, 160), (255, 255, 255), 2)
    
    y_offset = 40
    for label, value in metrics.items():
        if value is not None:
            text = f"{label}: {value}"
            cv2.putText(overlay, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
    
    return overlay

def main():
    video_path = "/Users/dhruvkulshrestha/Desktop/Cricket_bowling_analysis/input_video/cut_video.mp4"
    output_path = "/Users/dhruvkulshrestha/Desktop/Cricket_bowling_analysis/output_video/analysis_output.mp4"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    print(f"Processing: {os.path.basename(video_path)}")
    
    try:
        # 1️⃣ Initialize components
        pose_estimator = PoseEstimator(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )

        # 2️⃣ Process video with pose estimation
        print("Extracting pose data...")
        pose_data = pose_estimator.process_video(video_path)
        fps = pose_data["fps"]
        
        print(f"FPS: {fps:.2f}")
        print(f"Total frames: {pose_data['frame_count']}")
        
        frames_data = pose_data["frames_data"]
        
        # 3️⃣ Extract trajectory
        trajectory = pose_estimator.get_wrist_elbow_trajectory(
            frames_data=frames_data,
            bowling_arm="right"
        )

        wrist_traj_raw = trajectory["wrist_trajectory"]
        elbow_traj_raw = trajectory["elbow_trajectory"]
        timestamps = trajectory["timestamps"]
        
        if wrist_traj_raw.shape[0] == 0:
            print("No wrist points detected!")
            return
        
        print(f"Raw wrist points: {wrist_traj_raw.shape[0]}")

        # 4️⃣ Initialize TrajectoryAnalyzer
        trajectory_analyzer = TrajectoryAnalyzer(fps=fps)
        
        # 5️⃣ Smooth trajectories
        print("Smoothing trajectories...")
        wrist_traj_smooth = trajectory_analyzer.smooth_trajectory(
            wrist_traj_raw, window_size=7, polyorder=2
        )
        elbow_traj_smooth = trajectory_analyzer.smooth_trajectory(
            elbow_traj_raw, window_size=7, polyorder=2
        )
        
        # 6️⃣ Calculate velocities
        print("Calculating velocities...")
        wrist_velocity = trajectory_analyzer.calculate_velocity(wrist_traj_smooth)
        wrist_speed = trajectory_analyzer.calculate_speed(wrist_velocity)
        
        elbow_velocity = trajectory_analyzer.calculate_velocity(elbow_traj_smooth)
        elbow_speed = trajectory_analyzer.calculate_speed(elbow_velocity)
        
        # 7️⃣ Find ball release frame
        release_frame = trajectory_analyzer.find_ball_release_frame(wrist_speed)
        print(f"Ball release detected at frame: {release_frame}")
        print(f"Release timestamp: {timestamps[release_frame]:.3f}s")
        print(f"Max wrist speed: {wrist_speed.max():.1f} pixels/sec")
        print(f"Wrist speed at release: {wrist_speed[release_frame]:.1f} pixels/sec")
        
        #  Segment phases
        phases = trajectory_analyzer.segment_phases(wrist_traj_smooth, release_frame)
        print("\nAction Phases:")
        for phase_name, (start, end) in phases.items():
            if start < len(timestamps):
                duration = timestamps[min(end, len(timestamps)-1)] - timestamps[start]
                print(f"   {phase_name.replace('_', ' ').title():20} {duration:.3f}s")
        
        # 9️⃣ Read video frames
        print("\nReading video frames...")
        frames = read_video(video_path)
        
        if len(frames) == 0:
            print("❌ No frames read from video!")
            return
        
        # 🔟 Create annotated output
        print("Creating annotated video...")
        output_frames = []
        
        for i, frame in enumerate(frames):
            if i < len(wrist_traj_smooth):
                # Draw trajectory
                frame_with_traj = draw_trajectory_with_path(
                    frame, wrist_traj_smooth, i, history_length=20
                )
                
                # Determine current phase
                current_phase = "unknown"
                for phase_name, (start, end) in phases.items():
                    if start <= i < end:
                        current_phase = phase_name
                        break
                
                # Prepare metrics for this frame
                metrics = {
                    "Frame": i,
                    "Time": f"{timestamps[i]:.3f}s",
                    "Phase": current_phase.replace("_", " ").title(),
                    "Wrist Speed": f"{wrist_speed[i]:.1f} px/s" if i < len(wrist_speed) else "N/A",
                    "Elbow Speed": f"{elbow_speed[i]:.1f} px/s" if i < len(elbow_speed) else "N/A",
                }
                
                # Highlight release frame
                if i == release_frame:
                    metrics["RELEASE!"] = "BALL RELEASED"
                    cv2.circle(frame_with_traj, 
                              (int(wrist_traj_smooth[i][0]), int(wrist_traj_smooth[i][1])),
                              15, (0, 255, 0), 3)
                
                # Add metrics overlay
                frame_final = add_biomechanics_overlay(frame_with_traj, metrics)
                output_frames.append(frame_final)
            else:
                output_frames.append(frame)
        
        #  Save output video
        print("\nSaving output video...")
        save_video(output_frames, output_path, fps=fps)
        print(f"Output video saved to: {output_path}")
        
        # 📈 Create analysis plots
        print("Generating analysis plots...")
        
        # Plot 1: Trajectory
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(wrist_traj_smooth[:, 0], wrist_traj_smooth[:, 1], 'b-', linewidth=2, label='Wrist Path')
        plt.scatter(wrist_traj_smooth[release_frame, 0], wrist_traj_smooth[release_frame, 1], 
                   color='red', s=200, zorder=5, label='Ball Release')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.title('Wrist Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        
        # Plot 2: Speed profile
        plt.subplot(2, 2, 2)
        plt.plot(timestamps, wrist_speed, 'g-', linewidth=2, label='Wrist Speed')
        plt.axvline(x=timestamps[release_frame], color='r', linestyle='--', label='Ball Release')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (pixels/sec)')
        plt.title('Wrist Speed Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Phase visualization
        plt.subplot(2, 2, 3)
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'yellow', 'violet']
        phase_names = list(phases.keys())
        
        for idx, (phase_name, (start, end)) in enumerate(phases.items()):
            if start < len(timestamps):
                phase_duration = timestamps[min(end, len(timestamps)-1)] - timestamps[start]
                plt.barh(idx, phase_duration, color=colors[idx], edgecolor='black')
                plt.text(phase_duration/2, idx, phase_name.replace('_', ' ').title(), 
                        ha='center', va='center', fontweight='bold')
        
        plt.xlabel('Duration (seconds)')
        plt.title('Bowling Action Phases')
        plt.yticks([])
        
        # Plot 4: Summary metrics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        summary_text = f"""
        BIOMECHANICS ANALYSIS SUMMARY
        =============================
        
        Video Info:
        • FPS: {fps:.2f}
        • Total Frames: {len(frames)}
        • Duration: {timestamps[-1]:.3f}s
        
        Key Metrics:
        • Ball Release Frame: {release_frame}
        • Release Time: {timestamps[release_frame]:.3f}s
        • Max Wrist Speed: {wrist_speed.max():.1f} px/s
        • Release Wrist Speed: {wrist_speed[release_frame]:.1f} px/s
        
        Phase Durations:
        """
        
        for phase_name, (start, end) in phases.items():
            if start < len(timestamps):
                duration = timestamps[min(end, len(timestamps)-1)] - timestamps[start]
                summary_text += f"• {phase_name.replace('_', ' ').title()}: {duration:.3f}s\n"
        
        plt.text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
                verticalalignment='top', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        plot_path = output_path.replace('.mp4', '_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Analysis plot saved to: {plot_path}")
        
        # 📊 Print final summary
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print(f"Output files created:")
        print(f"  • Video: {output_path}")
        print(f"  • Plot: {plot_path}")
        print("\nKey findings:")
        print(f"  • Ball released at frame {release_frame}")
        print(f"  • Peak wrist speed: {wrist_speed.max():.1f} px/sec")
        print(f"  • Action duration: {timestamps[-1]:.3f} seconds")
        print("="*50)
        
        plt.show()
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()