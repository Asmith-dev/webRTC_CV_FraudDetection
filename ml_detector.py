#!/usr/bin/env python3
"""
Enhanced ML Detection Module for WebRTC Video Analysis
Handles face detection, head pose estimation, and other ML features using MediaPipe and advanced models
"""
import cv2
import numpy as np
import logging
import mediapipe as mp
import math
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple       
from dataclasses import dataclass
import face_recognition 

logger = logging.getLogger("ml-detector")

@dataclass
class FaceDetection:
    """Data class for face detection results"""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0
    landmarks: Optional[List[Tuple[float, float]]] = None

@dataclass
class HeadPose:
    """Data class for head pose estimation results"""
    pitch: float  # Up/down rotation (degrees)
    yaw: float    # Left/right rotation (degrees)
    roll: float   # Tilt rotation (degrees)
    confidence: float = 0.0
    translation_vector: Optional[Tuple[float, float, float]] = None
    rotation_vector: Optional[Tuple[float, float, float]] = None

@dataclass
class AnalysisResult:
    """Data class for complete analysis results"""
    timestamp: str
    faces_detected: int
    face_positions: List[Dict[str, Any]]
    head_poses: List[Dict[str, float]]
    frame_size: Dict[str, int]
    processing_time_ms: float
    features_enabled: Dict[str, bool]
    identity_changed: bool = False
    movement_px: int = 0  # Total bbox movement since last encode for visualization

class MLDetector: 
    """Enhanced ML detector class with MediaPipe and advanced models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ML detector with configuration
        
        Args:
            config: Configuration dictionary with ML settings
        """
        self.config = config or self._get_default_config()
        self.frame_count = 0
        self.analysis_interval = self.config.get('analysis_interval', 1)  # Analyze every frame by default
        
        # Initialize MediaPipe components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize ML models
        self._initialize_models()
        
        # 3D model points for head pose estimation
        self._setup_3d_model_points()
        
        self._baseline_encoding: Optional[np.ndarray] = None
        
        # Identity-tracking state
        self._identity_frame_counter: int = 0
        self._last_face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        
        logger.info("Enhanced ML Detector initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ML detector"""
        return {
            'analysis_interval': 1,  # Analyze every frame
            'face_detection': {
                'enabled': True,
                'model': 'mediapipe',  # Options: 'mediapipe', 'yunet', 'haarcascade'
                'min_detection_confidence': 0.5,
                'model_selection': 0  # 0 for short-range, 1 for full-range
            },
            'head_pose': {
                'enabled': True,
                'model': 'mediapipe_mesh',  # Options: 'mediapipe_mesh', 'pnp_solver'
                'confidence_threshold': 0.5,
                'use_face_mesh': True,
                'refine_landmarks': True
            },
            # Identity-tracking (face re-identification)
            'identity_tracking': {
                'enabled': True,
                'interval': 20,       # Encode once every N analysed frames
                'model': 'small',     # 'small' is faster; 'large' is default dlib resnet
                'num_jitters': 0,     # 0-1 for speed; higher improves robustness
                'tolerance': 0.6,     # Distance threshold for match
                'movement_threshold': 40  # Min-pixel movement of bbox to trigger re-encode
            },
            'performance': {
                'max_fps': 30,
                'resize_factor': 1.0  # Resize factor for processing (1.0 = original size)
            }
        }
    
    def _initialize_models(self):
        """Initialize all ML models"""
        try:
            # Initialize face detection
            if self.config['face_detection']['enabled']:
                self._initialize_face_detection()
            
            # Initialize head pose estimation
            if self.config['head_pose']['enabled']:
                self._initialize_head_pose()
                
            logger.info("All ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            raise
    
    def _initialize_face_detection(self):
        """Initialize face detection model"""
        try:
            model_type = self.config['face_detection']['model']
            
            if model_type == 'mediapipe':
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=self.config['face_detection']['model_selection'],
                    min_detection_confidence=self.config['face_detection']['min_detection_confidence']
                ) 
                logger.info("MediaPipe Face Detection model loaded successfully") 
                
            elif model_type == 'yunet':
                # YuNet model from OpenCV DNN
                model_path = cv2.dnn.readNetFromONNX("yunet.onnx")  # You need to download this
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model_path, "", (320, 320), 0.6, 0.3, 5000
                )
                logger.info("YuNet face detection model loaded successfully")
                
            elif model_type == 'haarcascade':
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
                if self.face_cascade.empty():
                    raise ValueError("Failed to load Haar cascade model")
                    
                logger.info("Haar Cascade face detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
            # Fallback to Haar cascade
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                self.config['face_detection']['model'] = 'haarcascade'
                logger.info("Fallback to Haar Cascade face detection")
            except:
                raise
    
    def _initialize_head_pose(self):
        """Initialize head pose estimation model"""
        try:
            if self.config['head_pose']['model'] == 'mediapipe_mesh':
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=5,
                    refine_landmarks=self.config['head_pose']['refine_landmarks'],
                    min_detection_confidence=self.config['head_pose']['confidence_threshold'],
                    min_tracking_confidence=0.5
                ) 
                logger.info("MediaPipe Face Mesh for head pose estimation loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize head pose estimation: {e}")
            raise  
    
    def _setup_3d_model_points(self):
        """Setup 3D model points for head pose estimation"""
        # 3D model points of a generic face (based on anthropometric data)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Camera internals (approximate values - should be calibrated for best results)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion 
    
    def _get_camera_matrix(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Get camera matrix for the current frame size"""
        if self.camera_matrix is None or self.camera_matrix.shape != (3, 3):
            focal_length = frame_width
            center = (frame_width / 2, frame_height / 2) 
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]   
            ], dtype=np.float64)
        return self.camera_matrix 
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[AnalysisResult]:
        """
        Analyze a video frame for all enabled ML features
        
        Args:
            frame: Input video frame as numpy array (BGR format)
            
        Returns:
            AnalysisResult object with all detection results
        """
        start_time = datetime.now()
        
        # Increment frame counter
        self.frame_count += 1
        
        # Skip frames based on analysis interval
        if self.frame_count % self.analysis_interval != 0:
            # Skip processing; caller will treat None as "no analysis for this frame"
            return None
        
        try:
            # Resize frame if needed for performance
            resize_factor = self.config['performance']['resize_factor']
            if resize_factor != 1.0:
                new_width = int(frame.shape[1] * resize_factor)
                new_height = int(frame.shape[0] * resize_factor)
                processing_frame = cv2.resize(frame, (new_width, new_height))
            else:
                processing_frame = frame.copy() 
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)    
            
            # Perform face detection
            faces = []
            if self.config['face_detection']['enabled']:
                faces = self._detect_faces_advanced(rgb_frame, processing_frame)
            
            # Perform head pose estimation
            head_poses = []
            if self.config['head_pose']['enabled'] and len(faces) > 0:
                head_poses = self._estimate_head_poses_advanced(rgb_frame, processing_frame, faces)
            
            # Scale results back to original frame size if resized
            if resize_factor != 1.0:
                faces = self._scale_detections(faces, resize_factor)
            
            # Identity tracking ---------------------------------
            identity_cfg = self.config.get('identity_tracking', {})
            identity_changed = False
            movement_px = 0  # initialize

            if identity_cfg.get('enabled', True) and len(faces) == 1: 
                f = faces[0]
                # Check movement – skip if face hasn't moved enough since last encode
                if self._last_face_bbox is not None:
                    dx = abs(f.x - self._last_face_bbox[0])
                    dy = abs(f.y - self._last_face_bbox[1])
                    dw = abs(f.w - self._last_face_bbox[2])
                    dh = abs(f.h - self._last_face_bbox[3])
                    movement_px = dx + dy + dw + dh
                    if movement_px < identity_cfg.get('movement_threshold', 40):
                        # insignificant movement – keep previous identity status
                        pass
                    else:
                        self._identity_frame_counter += 1
                else:
                    self._identity_frame_counter += 1

                if self._identity_frame_counter % identity_cfg.get('interval', 20) == 0:
                    # Crop and resize face to speed up encoding (expects RGB)
                    top = max(f.y, 0)
                    bottom = max(min(f.y + f.h, rgb_frame.shape[0]), top + 1)
                    left = max(f.x, 0)
                    right = max(min(f.x + f.w, rgb_frame.shape[1]), left + 1)
                    face_crop = rgb_frame[top:bottom, left:right]

                    try:
                        # Downscale to ~150 px width while preserving aspect ratio
                        if face_crop.size > 0:
                            h_c, w_c = face_crop.shape[:2]
                            scale = 150 / max(h_c, w_c)
                            if scale < 1.0:
                                face_crop = cv2.resize(face_crop, (int(w_c * scale), int(h_c * scale)))

                        encodings = face_recognition.face_encodings(
                            face_crop,
                            known_face_locations=[(0, face_crop.shape[1], face_crop.shape[0], 0)],
                            num_jitters=identity_cfg.get('num_jitters', 0),
                            model=identity_cfg.get('model', 'small')
                        )

                        if encodings:
                            encoding = encodings[0]
                            if self._baseline_encoding is None:
                                self._baseline_encoding = encoding
                            else:
                                same = face_recognition.compare_faces(
                                    [self._baseline_encoding],
                                    encoding,
                                    tolerance=identity_cfg.get('tolerance', 0.6)
                                )[0]
                                identity_changed = not same
                    except Exception as e:
                        logger.debug(f"Face encoding failed: {e}") 

                # Update bbox every analysed frame (outside encoding interval check)
                self._last_face_bbox = (f.x, f.y, f.w, f.h)
            elif len(faces) > 1 and identity_cfg.get('enabled', True):
                identity_changed = True  # multiple faces => uncertain identity
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = AnalysisResult(
                timestamp=datetime.now().isoformat(),
                faces_detected=len(faces),
                face_positions=[self._face_to_dict(face) for face in faces],
                head_poses=[self._head_pose_to_dict(pose) for pose in head_poses],
                frame_size={'width': frame.shape[1], 'height': frame.shape[0]},
                processing_time_ms=processing_time,
                features_enabled={
                    'face_detection': self.config['face_detection']['enabled'],
                    'head_pose': self.config['head_pose']['enabled'],
                    'identity_tracking': self.config.get('identity_tracking', {}).get('enabled', True)
                },
                identity_changed=identity_changed,
                movement_px=movement_px
            )
            
            logger.debug(f"Frame analyzed: {len(faces)} faces detected in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return None
    
    def _detect_faces_advanced(self, rgb_frame: np.ndarray, bgr_frame: np.ndarray) -> List[FaceDetection]:
        """Advanced face detection using selected model"""
        try:
            model_type = self.config['face_detection']['model']
            
            if model_type == 'mediapipe':
                return self._detect_faces_mediapipe(rgb_frame)
            elif model_type == 'yunet':
                return self._detect_faces_yunet(bgr_frame)
            elif model_type == 'haarcascade':
                return self._detect_faces_haar(bgr_frame) 
            
            return []
            
        except Exception as e:
            logger.error(f"Error in advanced face detection: {e}")
            return []
    
    def _detect_faces_mediapipe(self, rgb_frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using MediaPipe"""
        results = self.face_detection.process(rgb_frame)
        face_detections = []
        
        if results.detections:
            h, w, _ = rgb_frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Extract key landmarks if available
                landmarks = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        landmarks.append((keypoint.x * w, keypoint.y * h))
                
                face_detections.append(FaceDetection(
                    x=x, y=y, w=width, h=height,
                    confidence=detection.score[0],
                    landmarks=landmarks
                ))
        
        return face_detections
    
    def _detect_faces_yunet(self, bgr_frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using YuNet model"""
        if not hasattr(self, 'yunet_detector'):
            return []
            
        h, w = bgr_frame.shape[:2]
        self.yunet_detector.setInputSize((w, h))
        
        _, faces = self.yunet_detector.detect(bgr_frame)
        face_detections = []
        
        if faces is not None:
            for face in faces:
                x, y, w, h, conf = face[:5].astype(int)
                landmarks = face[5:15].reshape(5, 2)  # 5 landmarks
                
                face_detections.append(FaceDetection(
                    x=x, y=y, w=w, h=h,
                    confidence=conf,
                    landmarks=[(lm[0], lm[1]) for lm in landmarks]
                ))  
        
        return face_detections
    
    def _detect_faces_haar(self, bgr_frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_detections = []
        for (x, y, w, h) in faces:
            face_detections.append(FaceDetection(
                x=int(x), y=int(y), w=int(w), h=int(h),
                confidence=0.8  # Haar cascade doesn't provide confidence
            ))
        
        return face_detections
    
    def _estimate_head_poses_advanced(self, rgb_frame: np.ndarray, bgr_frame: np.ndarray, 
                                    faces: List[FaceDetection]) -> List[HeadPose]:
        """Advanced head pose estimation using MediaPipe Face Mesh"""
        try:
            if self.config['head_pose']['model'] == 'mediapipe_mesh':
                return self._estimate_head_poses_mediapipe(rgb_frame, bgr_frame)
            
            return []
            
        except Exception as e:
            logger.error(f"Error in head pose estimation: {e}")
            return []
    
    def _estimate_head_poses_mediapipe(self, rgb_frame: np.ndarray, bgr_frame: np.ndarray) -> List[HeadPose]:
        """Estimate head pose using MediaPipe Face Mesh"""
        results = self.face_mesh.process(rgb_frame)
        head_poses = []
        
        if results.multi_face_landmarks:
            h, w, _ = rgb_frame.shape
            camera_matrix = self._get_camera_matrix(w, h)     
            
            for face_landmarks in results.multi_face_landmarks:
                # Get specific landmark points for pose estimation
                landmarks = face_landmarks.landmark
                
                # Key facial points (in image coordinates)
                image_points = np.array([
                    (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
                    (landmarks[152].x * w, landmarks[152].y * h), # Chin
                    (landmarks[226].x * w, landmarks[226].y * h), # Left eye left corner
                    (landmarks[446].x * w, landmarks[446].y * h), # Right eye right corner
                    (landmarks[57].x * w, landmarks[57].y * h),   # Left mouth corner
                    (landmarks[287].x * w, landmarks[287].y * h)  # Right mouth corner
                ], dtype=np.float64)
                
                # Solve PnP to get rotation and translation vectors
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points,
                    image_points,
                    camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )   
                
                if success:
                    # Convert rotation vector to Euler angles
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    
                    # Calculate Euler angles (in degrees)
                    pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
                    
                    # unwrap pitch
                    if pitch < -90:
                        pitch += 180
                    elif pitch > 90:
                        pitch -= 180
                    
                    # unwrap roll
                    if roll < -90:
                        roll += 180
                    elif roll > 90:
                        roll -= 180
                    
                    head_poses.append(HeadPose(
                        pitch=pitch,
                        yaw=yaw,
                        roll=roll,
                        confidence=0.9,  # High confidence for MediaPipe
                        translation_vector=(
                            float(translation_vector[0][0]),
                            float(translation_vector[1][0]),
                            float(translation_vector[2][0])
                        ),
                        rotation_vector=(
                            float(rotation_vector[0][0]),
                            float(rotation_vector[1][0]),
                            float(rotation_vector[2][0])
                        )
                    ))
        
        return head_poses
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees"""
        # Calculate Euler angles from rotation matrix
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        # Convert from radians to degrees
        pitch = math.degrees(x)
        yaw = math.degrees(y)
        roll = math.degrees(z)
        
        return pitch, yaw, roll
    
    def _scale_detections(self, faces: List[FaceDetection], scale_factor: float) -> List[FaceDetection]:
        """Scale detection results back to original frame size"""
        inverse_scale = 1.0 / scale_factor
        
        scaled_faces = []
        for face in faces:
            scaled_face = FaceDetection(
                x=int(face.x * inverse_scale),
                y=int(face.y * inverse_scale),
                w=int(face.w * inverse_scale),
                h=int(face.h * inverse_scale),
                confidence=face.confidence,
                landmarks=[(lm[0] * inverse_scale, lm[1] * inverse_scale) for lm in face.landmarks] if face.landmarks else None
            )
            scaled_faces.append(scaled_face)
        
        return scaled_faces
    
    def _create_empty_result(self, frame_shape: Tuple[int, int, int]) -> AnalysisResult:
        """Create empty analysis result for skipped frames"""
        return AnalysisResult(
            timestamp=datetime.now().isoformat(),
            faces_detected=0,
            face_positions=[],
            head_poses=[],
            frame_size={'width': frame_shape[1], 'height': frame_shape[0]},
            processing_time_ms=0.0,
            features_enabled={
                'face_detection': self.config['face_detection']['enabled'],
                'head_pose': self.config['head_pose']['enabled'],
                'identity_tracking': self.config.get('identity_tracking', {}).get('enabled', True)
            },
            identity_changed=False,
            movement_px=0
        )
    
    def _face_to_dict(self, face: FaceDetection) -> Dict[str, Any]:
        """Convert FaceDetection object to dictionary"""
        result = {
            'x': face.x,
            'y': face.y,
            'w': face.w,
            'h': face.h,
            'confidence': face.confidence
        }
        
        if face.landmarks:
            result['landmarks'] = [{'x': lm[0], 'y': lm[1]} for lm in face.landmarks]
        
        return result
    
    def _head_pose_to_dict(self, pose: HeadPose) -> Dict[str, float]:
        """Convert HeadPose object to dictionary"""
        result = {
            'pitch': pose.pitch,
            'yaw': pose.yaw,
            'roll': pose.roll,
            'confidence': pose.confidence
        }
        
        if pose.translation_vector:
            result['translation'] = {
                'x': pose.translation_vector[0],
                'y': pose.translation_vector[1],
                'z': pose.translation_vector[2]
            }
        
        if pose.rotation_vector:
            result['rotation'] = {
                'x': pose.rotation_vector[0],
                'y': pose.rotation_vector[1],
                'z': pose.rotation_vector[2]
            }
        
        return result
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update detector configuration and reinitialize if needed"""
        old_config = self.config.copy()
        self.config.update(new_config)
        
        # Check if models need to be reinitialized
        models_changed = (
            old_config.get('face_detection', {}).get('model') != self.config.get('face_detection', {}).get('model') or
            old_config.get('head_pose', {}).get('model') != self.config.get('head_pose', {}).get('model')
        )
        
        if models_changed:
            logger.info("Model configuration changed, reinitializing...")
            self._initialize_models()
        
        logger.info("ML detector configuration updated")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current detector statistics"""
        return {
            'frames_processed': self.frame_count,
            'analysis_interval': self.analysis_interval,
            'features_enabled': {
                'face_detection': self.config['face_detection']['enabled'],
                'head_pose': self.config['head_pose']['enabled'],
                'identity_tracking': self.config.get('identity_tracking', {}).get('enabled', True)
            },
            'models_loaded': {
                'face_detection': hasattr(self, 'face_detection') or hasattr(self, 'face_cascade') or hasattr(self, 'yunet_detector'),
                'head_pose': hasattr(self, 'face_mesh')
            },
            'current_models': {
                'face_detection': self.config['face_detection']['model'],
                'head_pose': self.config['head_pose']['model']
            }
        }
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
        except:
            pass