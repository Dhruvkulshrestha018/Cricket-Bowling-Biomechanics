import numpy as np
from scipy import signal, interpolate
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings

class TrajectoryAnalyzer:
    def __init__(self, fps=30):
        self.fps = fps
        self.dt = 1 / fps
    
    def smooth_trajectory(self, trajectory: np.ndarray, window_size: int = 5, polyorder: int = 2):
        """
        Apply Savitzky-Golay filter for smooth trajectory
        """
        if len(trajectory) > window_size:
            x_smooth = signal.savgol_filter(trajectory[:, 0], window_size, polyorder)
            y_smooth = signal.savgol_filter(trajectory[:, 1], window_size, polyorder)
            return np.column_stack((x_smooth, y_smooth))
        return trajectory
    
    def calculate_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate velocity from position data
        """
        velocities = np.zeros_like(positions)
        for i in range(1, len(positions)):
            dx = positions[i, 0] - positions[i-1, 0]
            dy = positions[i, 1] - positions[i-1, 1]
            velocities[i] = [dx / self.dt, dy / self.dt]
        return velocities
    
    def calculate_speed(self, velocities: np.ndarray) -> np.ndarray:
        """
        Calculate speed magnitude from velocity vectors
        """
        return np.sqrt(np.sum(velocities**2, axis=1))
    
    def find_ball_release_frame(self, wrist_speed: np.ndarray, threshold_ratio: float = 0.8):
        """
        Identify ball release moment using peak wrist speed
        Based on biomechanics: ball release occurs near peak wrist velocity
        """
        if len(wrist_speed) == 0:
            return -1
        
        max_speed = np.max(wrist_speed)
        threshold = max_speed * threshold_ratio
        
        # Find first peak above threshold
        peaks, _ = signal.find_peaks(wrist_speed, height=threshold)
        
        if len(peaks) > 0:
            return peaks[0]  # First major peak
        else:
            # Fallback: frame with maximum speed
            return np.argmax(wrist_speed)
    
    def segment_phases(self, wrist_trajectory: np.ndarray, release_frame: int):
        """
        Segment bowling action into phases:
        1. Run-up
        2. Jump/Coiling
        3. Delivery stride
        4. Release
        5. Follow-through
        """
        total_frames = len(wrist_trajectory)
        
        phases = {
            'run_up': (0, int(release_frame * 0.3)),
            'coiling': (int(release_frame * 0.3), int(release_frame * 0.6)),
            'delivery_stride': (int(release_frame * 0.6), release_frame),
            'release': (release_frame, min(release_frame + int(0.1 * self.fps), total_frames)),
            'follow_through': (min(release_frame + int(0.1 * self.fps), total_frames), total_frames)
        }
        
        return phases