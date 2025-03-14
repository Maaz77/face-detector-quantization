"""
This scripts, captures a video from the webcam camera and performs face detection using the "BlazeFace" and 
"CenterFace" INT8 quantized models. Outputs, three videos: one for the original video, one for the BlazeFace, and 
the other one for CenterFace.
"""

from BlazeFace.webcamFaceDetection_still import process_video_file
from CenterFace.demo_tflite import process_video
import cv2
import time
import os

blazeface_model = "front_me_u8"
centerface_model = "CenterFace/Models/centerface_1x3xHxW_integer_quant.tflite"

def capture_webcam_video(output_filename="DetectedVideos/originalvideo.mp4", width=640, height=480, duration=60):
    """
    Captures video from webcam and saves it as MP4 file.
    
    Args:
        output_filename: Name of the output video file
        width: Width of the video frame
        height: Height of the video frame
        duration: Duration of the recording in seconds (default: 10)
    
    Returns:
        The path to the saved video file
    """
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))
    
    start_time = time.time()
    print(f"Recording video for {duration} seconds...")
    
    # Record video
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Write the frame to the output file
        out.write(frame)
        
        # Display the frame
        cv2.imshow('Recording...', frame)
        
        # Break the loop if 'q' is pressed or duration is reached
        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > duration:
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved as {output_filename}")
    return output_filename


capture_webcam_video()

print("Detecting for BlazeFace...")
process_video_file(input_video_path="DetectedVideos/originalvideo.mp4", model_name=blazeface_model, output_video_path="DetectedVideos/blazeface.mp4")
print("BlazeFace detection completed.")

print("##############################################")

print("Detecting for CenterFace...")
process_video(model_path=centerface_model, video_path="DetectedVideos/originalvideo.mp4", detection_result_path="DetectedVideos/centerface.mp4")
print("CenterFace detection completed.")

