#!/usr/bin/env python3
"""
Enhanced WebRTC Video Analysis Server with Advanced Face Detection and Head Pose Estimation
Uses aiortc for WebRTC communication and enhanced ML module for analysis
"""
import asyncio
import websockets
import json
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from datetime import datetime
import numpy as np
import time  # local import to avoid adding at top
import uuid

# Import our enhanced ML detector module
from en_ml import MLDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-server")

class EnhancedVideoAnalyzer:
    def __init__(self, websocket):
        self.websocket = websocket     
        
        # Initialize enhanced ML detector with configuration
        ml_config = {
            'analysis_interval': 10,  # Analyze every frame for real-time performance
            'face_detection': {
                'enabled': True,
                'model': 'mediapipe',  # Use MediaPipe for better accuracy
                'min_detection_confidence': 0.6,
                'model_selection': 0  # 0 for short-range (webcam), 1 for full-range
            },
            'head_pose': {
                'enabled': True,
                'model': 'mediapipe_mesh',  # Use MediaPipe Face Mesh for accurate pose
                'confidence_threshold': 0.6,
                'use_face_mesh': True,
                'refine_landmarks': True
            }, 
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
                'resize_factor': 0.75  # Reduce resolution for better performance
            }
        }
        
        self.ml_detector = MLDetector(ml_config)
        self.frame_skip_counter = 0
        self.performance_stats = {
            'frames_processed': 0,
            'avg_processing_time': 0.0,
            'max_processing_time': 0.0,
            'min_processing_time': float('inf')
        } 
        
        logger.info(f"Enhanced VideoAnalyzer initialized for {websocket.remote_address}")
        
    async def analyze_frame(self, frame: VideoFrame):
        """Analyze video frame using enhanced ML detector"""
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Off-load heavy ML analysis to a background thread so event-loop stays responsive
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.ml_detector.analyze_frame, img)   
            
            if result is None:
                return  # skip if analysis failed
            
            # Update performance statistics
            self._update_performance_stats(result.processing_time_ms)
            
            # Prepare enhanced analysis result
            analysis_data = {
                "type": "analysis",
                "data": {
                    "timestamp": result.timestamp,
                    "faces_detected": result.faces_detected,
                    "face_positions": result.face_positions,
                    "head_poses": result.head_poses,
                    "frame_size": result.frame_size,
                    "processing_time_ms": result.processing_time_ms,
                    "features_enabled": result.features_enabled,
                    "performance_stats": self.performance_stats.copy(),
                    "movement_px": getattr(result, "movement_px", 0)
                }
            }
            
            # Add detailed head pose information if available
            if result.head_poses:
                enhanced_poses = []
                for pose in result.head_poses:
                    enhanced_pose = pose.copy()
                    
                    # Add interpretable pose descriptions
                    enhanced_pose['pose_description'] = self._interpret_head_pose(pose)
                    enhanced_pose['attention_status'] = self._analyze_attention(pose)
                    
                    enhanced_poses.append(enhanced_pose)
                
                analysis_data["data"]["enhanced_head_poses"] = enhanced_poses
            
            # Add face quality metrics
            if result.face_positions:
                enhanced_faces = []
                for face in result.face_positions:
                    enhanced_face = face.copy()
                    
                    # Calculate face quality metrics
                    enhanced_face['quality_metrics'] = self._calculate_face_quality(face, img)
                    enhanced_face['size_category'] = self._categorize_face_size(face, result.frame_size)
                    
                    enhanced_faces.append(enhanced_face)
                
                analysis_data["data"]["enhanced_face_positions"] = enhanced_faces
            
            # Add identity alert flag if the person changed
            if getattr(result, "identity_changed", False):
                analysis_data["data"]["identity_alert"] = True
            
            # Send analysis result to client
            await self.websocket.send(json.dumps(analysis_data))
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            # Send error notification to client
            await self.websocket.send(json.dumps({
                "type": "error",
                "message": f"Analysis error: {str(e)}"
            }))

    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.performance_stats['frames_processed'] += 1
        
        # Update timing statistics
        if processing_time > self.performance_stats['max_processing_time']:
            self.performance_stats['max_processing_time'] = processing_time
        
        if processing_time < self.performance_stats['min_processing_time']:
            self.performance_stats['min_processing_time'] = processing_time
        
        # Calculate rolling average
        frames = self.performance_stats['frames_processed']
        current_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (current_avg * (frames - 1) + processing_time) / frames

    def _interpret_head_pose(self, pose: dict) -> dict:
        """Interpret head pose angles into human-readable descriptions"""
        pitch = pose.get('pitch', 0)
        yaw = pose.get('yaw', 0)
        roll = pose.get('roll', 0)
        
        # bring pitch into [-90, 90] by un-wrapping 180° flips
        if pitch < -90:
            pitch += 180
        elif pitch > 90:
            pitch -= 180
        
        # Interpret pitch (up/down)
        if pitch > 15:
            pitch_desc = "looking up"
        elif pitch < -15:
            pitch_desc = "looking down"
        else:
            pitch_desc = "level"
        
        # Interpret yaw (left/right)
        if yaw > 20:
            yaw_desc = "turned right"
        elif yaw < -20:
            yaw_desc = "turned left"
        else:
            yaw_desc = "facing forward"
        
        # Interpret roll (tilt)
        if roll > 15:
            roll_desc = "tilted right"
        elif roll < -15:
            roll_desc = "tilted left"
        else:
            roll_desc = "upright"
        
        return {
            "pitch_description": pitch_desc,
            "yaw_description": yaw_desc,
            "roll_description": roll_desc,
            "overall_pose": f"{yaw_desc}, {pitch_desc}, {roll_desc}"
        }

    def _analyze_attention(self, pose: dict) -> dict:
        """Analyze attention/engagement based on head pose"""
        pitch = pose.get('pitch', 0)
        yaw = pose.get('yaw', 0)
        
        # bring pitch into [-90, 90] by un-wrapping 180° flips
        if pitch < -90:
            pitch += 180
        elif pitch > 90:
            pitch -= 180
        
        # Calculate attention score (0-100)
        attention_score = 100
        
        # Reduce score based on head turn
        attention_score -= min(abs(yaw) * 2, 60)  # Max 60 point reduction for extreme turns
        
        # Reduce score based on up/down look
        attention_score -= min(abs(pitch) * 1.5, 30)  # Max 30 point reduction for extreme pitch
        
        attention_score = max(0, attention_score)
        
        # Categorize attention level
        if attention_score >= 80:
            attention_level = "high"
            attention_desc = "fully engaged"
        elif attention_score >= 60:
            attention_level = "moderate"
            attention_desc = "moderately engaged"
        elif attention_score >= 40:
            attention_level = "low"
            attention_desc = "distracted"
        else:
            attention_level = "very_low"
            attention_desc = "highly distracted"
        
        return {
            "attention_score": round(attention_score, 2),
            "attention_level": attention_level,
            "attention_description": attention_desc,
            "is_focused": attention_score >= 70
        }

    def _calculate_face_quality(self, face: dict, frame: np.ndarray) -> dict:
        """Calculate face quality metrics"""
        x, y, w, h = face['x'], face['y'], face['w'], face['h']
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return {"quality_score": 0, "issues": ["invalid_region"]}
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        quality_score = 100
        issues = []
        
        # Check face size (larger is generally better for analysis)
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        face_ratio = face_area / frame_area
        
        if face_ratio < 0.01:  # Face is less than 1% of frame
            quality_score -= 30
            issues.append("face_too_small")
        elif face_ratio > 0.5:  # Face is more than 50% of frame
            quality_score -= 10
            issues.append("face_too_large")
        
        # Check brightness
        mean_brightness = np.mean(gray_face)
        if mean_brightness < 80:
            quality_score -= 20
            issues.append("too_dark")
        elif mean_brightness > 200:
            quality_score -= 15
            issues.append("too_bright")
        
        # Check contrast (using standard deviation as proxy)
        contrast = np.std(gray_face)
        if contrast < 20:
            quality_score -= 15
            issues.append("low_contrast")
        
        # Check for blur (using Laplacian variance)
        try:
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if blur_score < 100:
                quality_score -= 25
                issues.append("blurry")
        except:
            pass
        
        quality_score = max(0, min(100, quality_score))
        
        return {
            "quality_score": round(quality_score, 2),
            "face_area_ratio": round(face_ratio * 100, 2),
            "brightness": round(mean_brightness, 2),
            "contrast": round(contrast, 2),
            "issues": issues
        }

    def _categorize_face_size(self, face: dict, frame_size: dict) -> str:
        """Categorize face size relative to frame"""
        face_area = face['w'] * face['h']
        frame_area = frame_size['width'] * frame_size['height']
        ratio = face_area / frame_area
        
        if ratio > 0.3:
            return "very_large"
        elif ratio > 0.15:
            return "large"
        elif ratio > 0.05:
            return "medium"
        elif ratio > 0.01:
            return "small"
        else:
            return "very_small"

class VideoTransformTrack:
    """
    Enhanced video track that receives frames and processes them
    """
    def __init__(self, track, analyzer):
        self.track = track
        self.analyzer = analyzer
        
        # --- FPS measurement state ---
        self._fps_window_start = time.time()
        self._fps_window_frames = 0
        
    async def recv(self): 
        frame = await self.track.recv()

        # Kick off analysis without blocking the RTP pipeline
        asyncio.create_task(self.analyzer.analyze_frame(frame)) 

        # ---------------- FPS tracking ----------------
        # Count every raw frame that enters the backend – independent of
        # analysis_interval – to know the true incoming frame rate.
        self._fps_window_frames += 1
        now = time.time()
        elapsed = now - self._fps_window_start

        if elapsed >= 1.0:  # log roughly once per second
            fps = self._fps_window_frames / elapsed
            logger.info(f"Incoming video FPS: {fps:.2f}")
            # Reset window
            self._fps_window_start = now
            self._fps_window_frames = 0
        # ------------------------------------------------

        return frame

async def handle_webrtc_connection(websocket, path):
    """Handle WebRTC signaling and media processing with enhanced features"""
    logger.info(f"New WebSocket connection from {websocket.remote_address}")
    
    # Create RTCPeerConnection with ICE servers
    ice_servers = [
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302"),
        RTCIceServer(urls="stun:stun2.l.google.com:19302"),
    ]
    config = RTCConfiguration(iceServers=ice_servers)
    pc = RTCPeerConnection(configuration=config) 
    
    # Create enhanced video analyzer
    analyzer = EnhancedVideoAnalyzer(websocket)
    
    # Store ICE candidates that arrive before remote description is set
    pending_candidates = []
    remote_description_set = False
   
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "connected":
            logger.info("WebRTC connection established successfully")
            # Send connection confirmation with capabilities
            await websocket.send(json.dumps({
                "type": "connection_established",
                "capabilities": {
                    "face_detection": True,
                    "head_pose_estimation": True,
                    "attention_analysis": True,
                    "face_quality_metrics": True
                }
            }))
        elif pc.connectionState == "failed":
            logger.warning("WebRTC connection failed")
            
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")
        
    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info(f"ICE gathering state is {pc.iceGatheringState}")
    
    @pc.on("track")
    async def on_track(track):
        logger.info(f"Received {track.kind} track")
        
        if track.kind == "video":
            # Create video transform track for analysis
            transform_track = VideoTransformTrack(track, analyzer)
            
            # Start processing frames
            asyncio.create_task(process_video_track(transform_track)) 
    
    async def process_video_track(track):
        """Process incoming video frames"""
        try:
            while True:
                await track.recv()
        except Exception as e:
            logger.error(f"Error processing video track: {e}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type == "offer":
                    logger.info("Received offer")
                    
                    # Set remote description
                    offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                    await pc.setRemoteDescription(offer)
                    remote_description_set = True
                    
                    # Process any pending ICE candidates
                    for candidate_data in pending_candidates:
                        await process_ice_candidate(pc, candidate_data)
                    
                    pending_candidates.clear()
                    
                    # Create and send answer
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    
                    await websocket.send(json.dumps({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    })) 
                    logger.info("Sent answer") 
                    
                elif message_type == "ice-candidate":
                    candidate_data = data.get("candidate")
                    if candidate_data and candidate_data.get("candidate"):
                        if remote_description_set:
                            await process_ice_candidate(pc, candidate_data)
                        else:
                            # Store candidate for later processing
                            pending_candidates.append(candidate_data)
                            logger.info("Stored ICE candidate for later processing")
                
                elif message_type == "get_stats":
                    # Return enhanced ML detector statistics
                    stats = analyzer.ml_detector.get_stats()
                    stats.update(analyzer.performance_stats)
                    await websocket.send(json.dumps({
                        "type": "stats",
                        "data": stats
                    }))
                
                elif message_type == "update_config":
                    # Update ML detector configuration
                    new_config = data.get("config", {})
                    analyzer.ml_detector.update_config(new_config)
                    await websocket.send(json.dumps({
                        "type": "config_updated",
                        "message": "Configuration updated successfully"
                    }))
                
                elif message_type == "get_capabilities":
                    # Return server capabilities
                    await websocket.send(json.dumps({
                        "type": "capabilities",
                        "data": {
                            "face_detection_models": ["mediapipe", "haarcascade"],
                            "head_pose_estimation": True,
                            "attention_analysis": True,
                            "face_quality_metrics": True,
                            "real_time_processing": True,
                            "performance_monitoring": True
                        }
                    }))
                
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Cleaning up connection")
        try:
            await pc.close()
        except:
            pass

async def process_ice_candidate(pc, candidate_data):
    """Process ICE candidate"""
    try:
        # Parse the candidate string
        candidate_str = candidate_data["candidate"]
        candidate_parts = candidate_str.split()
        
        # Extract components from candidate string
        foundation = candidate_parts[0].split(':')[1] if ':' in candidate_parts[0] else candidate_parts[0]
        component = int(candidate_parts[1])
        protocol = candidate_parts[2]
        priority = int(candidate_parts[3])
        ip = candidate_parts[4]
        port = int(candidate_parts[5])
        candidate_type = candidate_parts[7] if len(candidate_parts) > 7 else "host"
        
        candidate = RTCIceCandidate(
            component=component,
            foundation=foundation,
            ip=ip,
            port=port,
            priority=priority,
            protocol=protocol,
            type=candidate_type,
            sdpMid=candidate_data.get("sdpMid"),
            sdpMLineIndex=candidate_data.get("sdpMLineIndex")
        )
        await pc.addIceCandidate(candidate)
        logger.info("Added ICE candidate")
    except Exception as e:
        logger.error(f"Error adding ICE candidate: {e}")

async def main():
    """Start the enhanced WebRTC signaling server"""
    logger.info("Starting Enhanced WebRTC Video Analysis Server")
    logger.info("Features: Advanced Face Detection, Head Pose Estimation, Attention Analysis")
    logger.info("Server will listen on ws://0.0.0.0:8765")
    
    # Start WebSocket server
    async with websockets.serve(
        handle_webrtc_connection, 
        "0.0.0.0", 
        8765,
        max_size=2**20,  # 1MB max message size
        ping_interval=20,
        ping_timeout=10
    ):
        logger.info("Enhanced WebRTC server is running...")
        logger.info("Connect your WebRTC client to ws://localhost:8765")
        logger.info("Available endpoints:")
        logger.info("  - Face Detection: MediaPipe BlazeFace")
        logger.info("  - Head Pose: MediaPipe Face Mesh with PnP solver")
        logger.info("  - Features: Attention analysis, face quality metrics")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        # Import cv2 here to ensure it's available
        import cv2
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        logger.error("Please install: pip install opencv-python mediapipe")
    except Exception as e:
        logger.error(f"Server error: {e}")