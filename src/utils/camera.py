"""
Camera Utility Module
Handles webcam and iPhone camera inputs for the Contour system
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class Camera:
    """Camera interface for webcam and iPhone inputs"""
    
    def __init__(self, config: Dict):
        """Initialize camera"""
        self.config = config
        self.camera_type = config.get('type', 'webcam')
        self.device_id = config.get('device_id', 0)
        self.resolution = tuple(config.get('resolution', [640, 480]))
        self.fps = config.get('fps', 30)
        
        # Camera object
        self.cap = None
        
        # Camera parameters
        self.fx = config.get('fx', 525.0)
        self.fy = config.get('fy', 525.0)
        self.cx = config.get('cx', 319.5)
        self.cy = config.get('cy', 239.5)
        
        # Initialize camera
        self._initialize_camera()
        
        logger.info(f"Camera initialized: {self.camera_type}")
    
    def _initialize_camera(self):
        """Initialize camera based on type"""
        if self.camera_type == 'webcam':
            self._initialize_webcam()
        elif self.camera_type == 'iphone':
            self._initialize_iphone()
        else:
            raise ValueError(f"Unsupported camera type: {self.camera_type}")
    
    def _initialize_webcam(self):
        """Initialize webcam"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open webcam at device {self.device_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Webcam initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
        except Exception as e:
            logger.error(f"Error initializing webcam: {e}")
            raise
    
    def _initialize_iphone(self):
        """Initialize iPhone camera (simulated for now)"""
        # For now, we'll simulate iPhone camera with webcam
        # In practice, you would use AirPlay or similar for iPhone streaming
        logger.info("iPhone camera mode - using webcam as fallback")
        self._initialize_webcam()
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera"""
        if self.cap is None:
            logger.error("Camera not initialized")
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to capture frame")
                return None
            
            # Resize frame if needed
            if frame.shape[:2] != self.resolution[::-1]:
                frame = cv2.resize(frame, self.resolution)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def capture_batch(self, num_frames: int) -> List[np.ndarray]:
        """Capture multiple frames"""
        frames = []
        
        for _ in range(num_frames):
            frame = self.capture()
            if frame is not None:
                frames.append(frame)
            else:
                break
        
        return frames
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get camera intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def get_distortion_coeffs(self) -> np.ndarray:
        """Get camera distortion coefficients"""
        # Default to no distortion
        return np.zeros(5, dtype=np.float32)
    
    def calibrate(self, calibration_images: List[str]) -> bool:
        """Calibrate camera using calibration images"""
        try:
            # Prepare calibration data
            objpoints = []  # 3D points in real world space
            imgpoints = []  # 2D points in image plane
            
            # Chessboard parameters
            CHECKERBOARD = (9, 6)  # Number of internal corners
            
            # Prepare object points
            objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
            
            for image_path in calibration_images:
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
                
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
            
            if len(objpoints) < 5:
                logger.warning("Not enough calibration images")
                return False
            
            # Calibrate camera
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            if ret:
                # Update camera parameters
                self.fx = mtx[0, 0]
                self.fy = mtx[1, 1]
                self.cx = mtx[0, 2]
                self.cy = mtx[1, 2]
                
                logger.info("Camera calibration successful")
                return True
            else:
                logger.error("Camera calibration failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during camera calibration: {e}")
            return False
    
    def save_calibration(self, filepath: str):
        """Save camera calibration parameters"""
        try:
            calibration_data = {
                'fx': self.fx,
                'fy': self.fy,
                'cx': self.cx,
                'cy': self.cy,
                'resolution': self.resolution,
                'camera_matrix': self.get_camera_matrix().tolist(),
                'distortion_coeffs': self.get_distortion_coeffs().tolist()
            }
            
            np.save(filepath, calibration_data)
            logger.info(f"Camera calibration saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
    
    def load_calibration(self, filepath: str):
        """Load camera calibration parameters"""
        try:
            calibration_data = np.load(filepath, allow_pickle=True).item()
            
            self.fx = calibration_data['fx']
            self.fy = calibration_data['fy']
            self.cx = calibration_data['cx']
            self.cy = calibration_data['cy']
            
            logger.info(f"Camera calibration loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
    
    def get_frame_info(self) -> Dict:
        """Get current frame information"""
        if self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION)
        }
    
    def set_property(self, prop_id: int, value: float):
        """Set camera property"""
        if self.cap is not None:
            self.cap.set(prop_id, value)
    
    def get_property(self, prop_id: int) -> float:
        """Get camera property"""
        if self.cap is not None:
            return self.cap.get(prop_id)
        return 0.0
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")


class CameraRecorder:
    """Camera recorder for saving video streams"""
    
    def __init__(self, output_path: str, fps: int = 30, 
                 codec: str = 'mp4v', resolution: Tuple[int, int] = (640, 480)):
        """Initialize camera recorder"""
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.resolution = resolution
        
        # Video writer
        self.writer = None
        
        # Initialize writer
        self._initialize_writer()
    
    def _initialize_writer(self):
        """Initialize video writer"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, self.resolution
            )
            
            if not self.writer.isOpened():
                raise RuntimeError("Could not open video writer")
            
            logger.info(f"Video recorder initialized: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error initializing video recorder: {e}")
            raise
    
    def record_frame(self, frame: np.ndarray):
        """Record a frame"""
        if self.writer is not None:
            # Resize frame if needed
            if frame.shape[:2] != self.resolution[::-1]:
                frame = cv2.resize(frame, self.resolution)
            
            self.writer.write(frame)
    
    def release(self):
        """Release video writer"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.info("Video recorder released")


class CameraStream:
    """Camera stream for real-time processing"""
    
    def __init__(self, camera: Camera, buffer_size: int = 10):
        """Initialize camera stream"""
        self.camera = camera
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.is_running = False
        
    def start(self):
        """Start camera stream"""
        self.is_running = True
        logger.info("Camera stream started")
    
    def stop(self):
        """Stop camera stream"""
        self.is_running = False
        logger.info("Camera stream stopped")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the stream"""
        if not self.is_running:
            return None
        
        frame = self.camera.capture()
        if frame is not None:
            # Update buffer
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
        
        return frame
    
    def get_frame_buffer(self) -> List[np.ndarray]:
        """Get the current frame buffer"""
        return self.frame_buffer.copy()
    
    def get_average_frame(self) -> Optional[np.ndarray]:
        """Get average frame from buffer (for noise reduction)"""
        if not self.frame_buffer:
            return None
        
        # Convert to float for averaging
        frames_float = [frame.astype(np.float32) for frame in self.frame_buffer]
        average_frame = np.mean(frames_float, axis=0)
        
        return average_frame.astype(np.uint8) 