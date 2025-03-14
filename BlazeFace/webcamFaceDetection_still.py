# # Configuration parameters
# scoreThreshold = 0.7
# iouThreshold = 0.3      
# #modelType = "front"  # model_path="models/face_detection_front.tflite"
# #modelType = "back"  # model_path="models/face_detection_back.tflite"
# # modelType = "front_u8"  # model_path="models/face_detection_front_128x128_full_integer_quant.tflite"
# # modelType = "back_u8"  # model_path="models/face_detection_back_256x256_full_integer_quant.tflite"
# #modelType = "back_i8"  # model_path="models/face_detection_back_256x256_integer_quant.tflite"
# # modelType = "front_i8"  # model_path="models/face_detection_front_128x128_integer_quant.tflite"
# # modelType = "back_me_u8"  # model_path="models/blaze_face_back_me.tflite" 
# modelType = "front_me_u8"  # model_path="models/blaze_face_front.tflite"

import cv2
from .BlazeFaceDetection.blazeFaceDetectorQ import blazeFaceDetector
import imageio

#if you run this script directly, remove the . from above import line and then go to "blazeFaceDetectorQ.py"


def init_face_detector(model_type, score_threshold, iou_threshold):
    """Initialize the BlazeFace detector with the specified parameters"""
    return blazeFaceDetector(model_type, score_threshold, iou_threshold)

def init_camera(camera_id=0):
    """Initialize the webcam and return camera object with properties"""
    camera = cv2.VideoCapture(camera_id)
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    
    if not camera.isOpened():
        raise RuntimeError("Error: Could not open webcam")
        
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    
    return camera, frame_width, frame_height, fps

def init_video_writer(record_video, model_type, frame_width, frame_height, fps):
    """Initialize video writer if recording is enabled"""
    if not record_video:
        return None
        
    output_filename = f"{model_type}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = (frame_width, frame_width) if frame_width < frame_height else (frame_height, frame_height)
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, size)
    print(f"Recording to {output_filename}")
    return video_writer

def crop_to_square(img, frame_width, frame_height):
    """Crop the input frame to a square from the center"""
    if frame_width > frame_height:
        offset = (frame_width - frame_height) // 2
        return img[:, offset:offset + frame_height]
    else:
        offset = (frame_height - frame_width) // 2
        return img[offset:offset + frame_width, :]

def process_frame(img, face_detector):
    """Process a single frame to detect and visualize faces"""
    detection_results = face_detector.detectFaces(img)
    return face_detector.drawDetections(img, detection_results)

def cleanup_resources(camera, video_writer=None, model_type=None, record_video=False):
    """Clean up all resources properly"""
    camera.release()
    if record_video and video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print("Program terminated successfully.")
    if record_video:
        print(f"Video saved as {model_type}.mp4")

def main():
    # Configuration parameters
    score_threshold = 0.7
    iou_threshold = 0.3
    model_type = "front_me_u8"  # model_path="models/blaze_face_front.tflite"
    record_video = True
    
    # Initialize components
    face_detector = init_face_detector(model_type, score_threshold, iou_threshold)
    camera, frame_width, frame_height, fps = init_camera()
    #video_writer = init_video_writer(record_video, model_type, frame_width, frame_height, fps)
    
    try:
        while True:
            # Read frame from the webcam
            ret, img = camera.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            # Crop the frame to a square
            img = crop_to_square(img, frame_width, frame_height)
            
            # Detect faces and draw results
            img_plot = process_frame(img, face_detector)
            
            # Write frame to video file if recording is enabled
            #if record_video and video_writer is not None:
                #video_writer.write(img_plot)
            
            # Display the result
            cv2.imshow("Face Detection", img_plot)
            
            # Press key q to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Always clean up resources properly
        cleanup_resources(camera, model_type, record_video)
        #cleanup_resources(camera, video_writer, model_type, record_video)

        

def process_video_file(input_video_path, model_name, output_video_path=None, score_threshold=0.7, iou_threshold=0.3):
    """
    Process a video file with face detection and save the annotated result as a video
    
    Args:
        input_video_path (str): Path to input video file
        model_name (str): Name of face detection model to use
        output_video_path (str, optional): Path to save output video (defaults to "[model_name]_output.mp4")
        score_threshold (float, optional): Confidence threshold for detections (default: 0.7)
        iou_threshold (float, optional): IOU threshold for non-max suppression (default: 0.3)
    
    Returns:
        str: Path to the saved output video
    """
    # Initialize the face detector
    face_detector = init_face_detector(model_name, score_threshold, iou_threshold)
    
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open input video file: {input_video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine cropped dimensions
    square_size = min(frame_width, frame_height)
    
    # Set output path
    if output_video_path is None:
        output_video_path = f"{model_name}_output.mp4"
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        fps,
        (square_size, square_size)
    )
    
    print(f"Processing video: {input_video_path}")
    print(f"Output will be saved to: {output_video_path}")
    print(f"Total frames: {total_frames}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Crop the frame to a square
            frame = crop_to_square(frame, frame_width, frame_height)
            
            # Process the frame
            detection_results = face_detector.detectFaces(frame)
            annotated_frame = face_detector.drawDetections(frame, detection_results)
            
            # Write the frame to the output video
            video_writer.write(annotated_frame)
            
            # Update progress
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
    
    finally:
        # Release resources
        cap.release()
        video_writer.release()
        
        if frame_count > 0:
            print(f"\nVideo processing completed: {frame_count} frames processed")
            print(f"Output saved to: {output_video_path}")
        else:
            print("No frames were processed, video was not created")
        
    return output_video_path



if __name__ == "__main__":
    

    main()
