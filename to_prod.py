import os
import json
import pika
# Initialize Google Cloud Storage client
from google.cloud import storage
from google.oauth2 import service_account
# from config import CONSUMER_QUEUE, DEAD_LETTER_QUEUE, DEAD_LETTER_EXCHANGE

ML_QUEUE = "TEST_PRE_TRANSCRIPTION_QUEUE_LOCAL" 
DEAD_LETTER_QUEUE = "TEST_TRANSCRIPTION_DEAD_LETTER_QUEUE_LOCAL"
DEAD_LETTER_EXCHANGE = "TEST_TRANSCRIPTION_DEAD_LETTER_EXCHANGE_LOCAL"

BROKER_URL = 'amqp://rabbit:84sxOIf4V0k8Rwp@34.170.213.229:5672'
print(f"DEAD_LETTER_QUEUE: {bool(DEAD_LETTER_QUEUE)}")
print(f"DEAD_LETTER_EXCHANGE: {bool(DEAD_LETTER_EXCHANGE)}")

GOOGLE_CLOUD_BUCKET_NAME = "stag_metantz"


def initialize_gcs_client():
    """Initialize and return Google Cloud Storage client and bucket"""
    try:
        credentials = service_account.Credentials.from_service_account_file('keyfile.json') 
        client = storage.Client(credentials=credentials)
        bucket = client.get_bucket(GOOGLE_CLOUD_BUCKET_NAME)
        print('GCS client initialized:', bool(client))
        return client, bucket
    except Exception as e:
        print(f"Error initializing GCS client: {e}")
        raise

# Initialize GCS client globally
gcs_client, gcs_bucket = initialize_gcs_client()

# Import ML analysis components
import cv2
import tempfile
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video-analyzer")

# Import the ML detector (assuming it's in the same directory)
try:
    from en_ml import MLDetector
    ML_AVAILABLE = True
except ImportError:
    logger.warning("ML detector not available. Install required packages: pip install opencv-python mediapipe face-recognition")
    ML_AVAILABLE = False

def download_video_from_gcs(interview_id: str, bucket_name: str = GOOGLE_CLOUD_BUCKET_NAME) -> str:
    """
    Download video from GCS and save to temporary file
    
    Args:
        interview_id: The interview ID to download
        bucket_name: GCS bucket name
        
    Returns:
        Path to the downloaded video file
    """
    try:
        # Construct GCS path
        gcs_path = f"interviews_stitched/{interview_id}/stitched_video.mp4"
        
        # Get blob from bucket
        blob = gcs_bucket.get_blob(gcs_path)
        
        if blob is None:
            raise FileNotFoundError(f"Video not found at path: {gcs_path}")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()
        
        # Download video to temporary file
        blob.download_to_filename(temp_path)
        
        logger.info(f"Downloaded video for interview {interview_id} to {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Error downloading video for interview {interview_id}: {e}")
        raise

def analyze_video_file(video_path: str, interview_id: str) -> dict:
    """
    Analyze video file using ML pipeline
    
    Args:
        video_path: Path to the video file
        interview_id: Interview ID for logging
        
    Returns:
        Analysis results dictionary
    """
    if not ML_AVAILABLE:
        logger.error("ML analysis not available - missing dependencies")
        return {"error": "ML analysis not available"}
    
    try:
        # Initialize ML detector with configuration for post-interview analysis
        ml_config = {
            'analysis_interval': 5,  # Analyze every 5th frame for efficiency
            'face_detection': {
                'enabled': True,
                'model': 'mediapipe',
                'min_detection_confidence': 0.6,
                'model_selection': 0
            },
            'head_pose': {
                'enabled': True,
                'model': 'mediapipe_mesh',
                'confidence_threshold': 0.6,
                'use_face_mesh': True,
                'refine_landmarks': True
            },
            'eye_gaze': {
                'enabled': True,
                'model': 'mediapipe_iris',
                'confidence_threshold': 0.6,
                'gaze_smoothing': 0.3
            },
            'identity_tracking': {
                'enabled': True,
                'interval': 30,  # Check identity every 30 frames
                'model': 'small',
                'num_jitters': 0,
                'tolerance': 0.6,
                'movement_threshold': 40
            },
            'cheating_detection': {
                'enabled': True,
                'gaze_away_threshold': 15,
                'head_turn_threshold': 25,
                'pupil_displacement_threshold': 0.3,
                'time_window': 30,
                'suspicious_ratio_threshold': 0.3,
                'combine_head_gaze': True,
                'track_eye_patterns': True,
                'subtle_cheating_detection': True
            },
            'performance': {
                'max_fps': 30,
                'resize_factor': 0.75
            }
        }
        
        ml_detector = MLDetector(ml_config)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Analyzing video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Analysis results storage
        analysis_results = {
            'interview_id': interview_id,
            'video_info': {
                'total_frames': total_frames,
                'fps': fps,
                'duration_seconds': duration,
                'analysis_start_time': datetime.now().isoformat()
            },
            'frame_analysis': [],
            'summary': {
                'total_faces_detected': 0,
                'suspicious_frames': 0,
                'identity_changes': 0,
                'average_attention_score': 0.0,
                'cheating_risk_level': 'low',
                'recommendations': []
            },
            'cheating_indicators': {
                'suspicious_behaviors': set(),
                'gaze_away_count': 0,
                'head_turn_count': 0,
                'subtle_eye_movement_count': 0
            }
        }
        
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze frame using ML detector
            result = ml_detector.analyze_frame(frame)
            
            if result is not None:
                processed_frames += 1
                
                # Store frame analysis
                frame_data = {
                    'frame_number': frame_count,
                    'timestamp': result.timestamp,
                    'faces_detected': result.faces_detected,
                    'face_positions': result.face_positions,
                    'head_poses': result.head_poses,
                    'eye_gazes': result.eye_gazes,
                    'processing_time_ms': result.processing_time_ms,
                    'identity_changed': result.identity_changed,
                    'cheating_indicators': result.cheating_indicators
                }
                
                analysis_results['frame_analysis'].append(frame_data)
                
                # Update summary statistics
                if result.faces_detected > 0:
                    analysis_results['summary']['total_faces_detected'] += result.faces_detected
                
                if result.identity_changed:
                    analysis_results['summary']['identity_changes'] += 1
                
                # Process cheating indicators
                if result.cheating_indicators:
                    indicators = result.cheating_indicators
                    
                    if indicators.get('suspicion_level') in ['medium', 'high']:
                        analysis_results['summary']['suspicious_frames'] += 1
                    
                    # Collect suspicious behaviors
                    for behavior in indicators.get('suspicious_behaviors', []):
                        analysis_results['cheating_indicators']['suspicious_behaviors'].add(behavior)
                        
                        if behavior == 'looking_away':
                            analysis_results['cheating_indicators']['gaze_away_count'] += 1
                        elif behavior == 'excessive_head_turn':
                            analysis_results['cheating_indicators']['head_turn_count'] += 1
                        elif behavior == 'subtle_eye_movement':
                            analysis_results['cheating_indicators']['subtle_eye_movement_count'] += 1
                
                # Log progress every 100 frames
                if processed_frames % 100 == 0:
                    logger.info(f"Processed {processed_frames}/{total_frames} frames for interview {interview_id}")
        
        # Close video
        cap.release()
        
        # Calculate final summary
        if processed_frames > 0:
            suspicious_ratio = analysis_results['summary']['suspicious_frames'] / processed_frames
            
            # Determine cheating risk level
            if suspicious_ratio > 0.4:
                analysis_results['summary']['cheating_risk_level'] = 'high'
                analysis_results['summary']['recommendations'].append('High risk of cheating detected - manual review recommended')
            elif suspicious_ratio > 0.2:
                analysis_results['summary']['cheating_risk_level'] = 'medium'
                analysis_results['summary']['recommendations'].append('Moderate risk - consider additional verification')
            else:
                analysis_results['summary']['cheating_risk_level'] = 'low'
                analysis_results['summary']['recommendations'].append('Low risk - standard processing')
            
            # Convert sets to lists for JSON serialization
            analysis_results['cheating_indicators']['suspicious_behaviors'] = list(
                analysis_results['cheating_indicators']['suspicious_behaviors']
            )
        
        analysis_results['video_info']['analysis_end_time'] = datetime.now().isoformat()
        analysis_results['summary']['processed_frames'] = processed_frames
        
        logger.info(f"Analysis completed for interview {interview_id}: {processed_frames} frames processed")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error analyzing video for interview {interview_id}: {e}")
        return {"error": str(e), "interview_id": interview_id}

def process_interview_message(ch, method, properties, body):
    """
    Process a message from the queue containing interview_id
    
    Args:
        ch: Channel
        method: Delivery method
        properties: Message properties
        body: Message body (JSON string)
    """
    try:
        # Parse message
        message = json.loads(body)
        interview_id = message.get('interview_id')
        
        if not interview_id:
            logger.error("Message missing interview_id")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        
        logger.info(f"Processing interview: {interview_id}")
        
        # Download video from GCS
        video_path = download_video_from_gcs(interview_id)
        
        try:
            # Analyze video
            analysis_results = analyze_video_file(video_path, interview_id)
            
            # Save analysis results (you can modify this to save to database, file, etc.)
            results_file = f"analysis_results_{interview_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            logger.info(f"Analysis results saved to {results_file}")
            
            # Log summary
            if 'error' not in analysis_results:
                summary = analysis_results['summary']
                logger.info(f"Interview {interview_id} analysis summary:")
                logger.info(f"  - Risk Level: {summary['cheating_risk_level']}")
                logger.info(f"  - Suspicious Frames: {summary['suspicious_frames']}/{summary['processed_frames']}")
                logger.info(f"  - Identity Changes: {summary['identity_changes']}")
                logger.info(f"  - Recommendations: {summary['recommendations']}")
            else:
                logger.error(f"Analysis failed for interview {interview_id}: {analysis_results['error']}")
        
        finally:
            # Clean up temporary video file
            try:
                os.unlink(video_path)
                logger.info(f"Cleaned up temporary file: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {video_path}: {e}")
        
        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in message: {e}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        # Reject message and requeue
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def start_message_consumer():
    """Start consuming messages from the queue"""
    try:
        logger.info("Starting message consumer...")
        logger.info(f"Connecting to RabbitMQ at: {BROKER_URL}")
        logger.info(f"Consuming from queue: {ML_QUEUE}")
        
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(pika.URLParameters(BROKER_URL))
        channel = connection.channel()
        
        # Declare queue with dead letter exchange
        arguments = {
            "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE,
            "x-dead-letter-routing-key": DEAD_LETTER_QUEUE,
        }
        channel.queue_declare(queue=ML_QUEUE, durable=True, arguments=arguments)
        
        # Declare dead letter queue and exchange
        channel.exchange_declare(exchange=DEAD_LETTER_EXCHANGE, exchange_type='direct', durable=True)
        channel.queue_declare(queue=DEAD_LETTER_QUEUE, durable=True)
        channel.queue_bind(exchange=DEAD_LETTER_EXCHANGE, queue=DEAD_LETTER_QUEUE, routing_key=DEAD_LETTER_QUEUE)
        
        # Set up consumer
        channel.basic_qos(prefetch_count=1)  # Process one message at a time
        channel.basic_consume(
            queue=ML_QUEUE,
            on_message_callback=process_interview_message,
            auto_ack=False
        )
        
        logger.info("Consumer started. Waiting for messages...")
        logger.info("Press Ctrl+C to stop")
        
        # Start consuming
        channel.start_consuming()
        
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user")
    except Exception as e:
        logger.error(f"Consumer error: {e}")
    finally:
        try:
            connection.close()
            logger.info("Connection closed")
        except:
            pass

def publish_test_message():
    """Publish a test message to the queue"""
    try: 
        print(f"Connecting to RabbitMQ at: {bool(BROKER_URL)}")
        print(f"Publishing to queue: {bool(ML_QUEUE)}") 

        # Connect to RabbitMQ
        connection = pika.BlockingConnection(pika.URLParameters(BROKER_URL))
        channel = connection.channel() 

        # Add the DLX argument to match the consumer's queue declaration
        arguments = {
            "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE,
            "x-dead-letter-routing-key": DEAD_LETTER_QUEUE,
        }
        channel.queue_declare(queue=ML_QUEUE, durable=True, arguments=arguments)

        # Create a test message
        message = {"interview_id": "685a900953a0a77cf89dd3d2"}  

        # Publish the message
        channel.basic_publish(
            exchange="",  # Use default exchange
            routing_key=ML_QUEUE,  # Queue name as routing key
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type="application/json",
            ),
        )

        print(f"Published test message: {message}")
        connection.close()
        print("Connection closed")

    except Exception as e:
        print(f"Error publishing test message: {e}")

#need to creare a script for consuming the message from the queue and then processing it through our ml pipeline

if __name__ == "__main__":
    # Uncomment one of these based on what you want to do:
    
    # 1. Publish a test message
    publish_test_message()
        
    # 2. Start the consumer to process messages
    start_message_consumer()