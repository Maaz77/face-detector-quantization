# import cv2
# import glob
# import numpy as np

# image_directory = '/Users/maaz/Desktop/ST-face-monitoring/Face-Detectors-Exp/CenterFace/Images_for_Calibration'

# # load 1024 random images from the image_directory
# files = glob.glob(f'{image_directory}/*.jpg')
# np.random.shuffle(files)
# files = files[:256]
# img_datas = []
# for idx, file in enumerate(files):
#     bgr_img = cv2.imread(file)
#     rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
#     resized_img = cv2.resize(rgb_img, dsize=(128,128))
#     extend_batch_size_img = resized_img[np.newaxis, :]
#     #normalized_img = extend_batch_size_img / 255.0 
#     img_datas.append(extend_batch_size_img)
# calib_datas = np.vstack(img_datas)
# # print the mean and std per channel for the calibration dataset
# mean_per_channel = np.mean(calib_datas, axis=0)
# std_per_channel = np.std(calib_datas, axis=0)

# calib_datas = (calib_datas - mean_per_channel) / std_per_channel

# new_mean_per_channel = np.mean(calib_datas, axis=(0, 1, 2))
# new_std_per_channel = np.std(calib_datas, axis=(0, 1, 2))


# print(f'Mean per channel: {new_mean_per_channel}')
# print(f'Std per channel: {new_std_per_channel}')
# calib_datas = calib_datas.transpose(0,3,1,2)
# print(f'calib_datas.shape: {calib_datas.shape}') 
# np.save(file='maaz_calibdata.npy', arr=calib_datas)


import cv2
import glob
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt



def get_image_files(directory: str, sample_size: int = 256) -> List[str]:
    """Get a random sample of image files from directory."""
    files = glob.glob(f'{directory}/*.jpg')
    np.random.shuffle(files)
    return files[:sample_size]


def process_image(file: str) -> np.ndarray:
    """Process a single image file into the required format."""
    bgr_img = cv2.imread(file)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img[np.newaxis, :]  # Add batch dimension


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data with mean and std."""
    mean_per_channel = np.mean(data, axis=0)
    std_per_channel = np.std(data, axis=0)
    normalized_data = (data - mean_per_channel) / std_per_channel
    return normalized_data, mean_per_channel, std_per_channel


def print_stats(data: np.ndarray) -> None:
    """Print mean and std statistics per channel."""
    mean_per_channel = np.mean(data, axis=(0, 1, 2))
    std_per_channel = np.std(data, axis=(0, 1, 2))
    print(f'Mean per channel: {mean_per_channel}')
    print(f'Std per channel: {std_per_channel}')

def display_random_images(data: np.ndarray = None, file_path: str = None, percentage: float = 10, is_nchw: bool = False) -> None:
    """
    Display a random percentage of images from the calibration data.
    
    Args:
        data: Numpy array of images in either NHWC or NCHW format
        file_path: Path to the numpy file containing the images
        percentage: Percentage of images to display (0-100)
        is_nchw: Whether the data is in NCHW format (vs. NHWC)
    """
    
    # Load data from file if file_path is provided
    if file_path is not None:
        data = np.load(file_path)
    
    if data is None:
        raise ValueError("Either data or file_path must be provided.")
    
    # Convert from NCHW to NHWC if necessary
    if is_nchw:
        data = data.transpose(0, 2, 3, 1)
    
    n_images = data.shape[0]
    n_display = max(1, int(n_images * percentage / 100))
    
    # Randomly select indices
    indices = np.random.choice(n_images, n_display, replace=False)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_display)))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Display each selected image
    for i, idx in enumerate(indices):
        if i >= n_display:
            break
            
        img = data[idx]
        
        # Add subplot
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ax.imshow(img)
        ax.set_title(f"Image {idx}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()


def gen_calibdata_NCHW_Standardized(image_directory: str =  '/Users/maaz/Desktop/ST-face-monitoring/Face-Detectors-Exp/CenterFace/Images_for_Calibration' , output_file_path : str = 'maaz_calibdata.npy') -> None:
    
    """Generate and save the calibration data in NCHW format."""
    
    # Get image files
    files = get_image_files(image_directory)
    
    # Process all images
    img_datas = list(map(process_image, files))
    
    # Stack and normalize data
    calib_datas = np.vstack(img_datas)
    calib_datas, _, _ = normalize_data(calib_datas)
    
    # Print statistics
    print_stats(calib_datas)
    
    # Rearrange dimensions for model input (NCHW format)
    calib_datas = calib_datas.transpose(0, 3, 1, 2)
    print(f'calib_datas.shape: {calib_datas.shape}')
    
    # Save data
    np.save(file= output_file_path, arr=calib_datas)
    

def original_model_preprocess(frame):
    """
    Preprocessing procedure on a captured frame for the CenterFace original .onnx model.
    
    Args:
        frame: The raw frame from the camera
        
    Returns:
        processed_frame: The frame ready for model inference
    """

    # crop a square patch from the center of the frame with size min(H, W)
    h, w = frame.shape[:2]
    s = min(h, w)
    y1 = (h - s) // 2
    y2 = y1 + s
    x1 = (w - s) // 2
    x2 = x1 + s
    frame = frame[y1:y2, x1:x2]

    img_h_new, img_w_new = 128, 128  # Currently using original dimensions

    processed_frame = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,           # No scaling
        size=(img_w_new, img_h_new),  # Target size (same as original in this case)
        mean=(0, 0, 0),            # No mean subtraction
        swapRB=True,               # Swap Red and Blue channels (BGR to RGB)
        crop=False                 # No cropping
    )
    
    return processed_frame


def gen_calibdata_from_webcam(
    output_file_path: str = 'webcam_calibdata.npy',
    sample_size: int = 256,
    capture_frequency_ms: float = 200,
    camera_id: int = 0,
    display_percentage: float = 10,
    normalize: bool = True
) -> None:
    """
    Generate and save calibration data from webcam captures in NCHW format.
    
    Args:
        output_file_path: Path to save the calibration data
        sample_size: Number of frames to capture (default: 256)
        capture_frequency_ms: Time between captures in milliseconds (default: 0.5ms)
        camera_id: Camera ID to use (default: 0 for default webcam)
        display_percentage: Percentage of images to display after capture
        normalize: Whether to normalize the data (default: True)
    """
    import time
    import os
    
    # Update output filename based on normalization setting
    base_name, ext = os.path.splitext(output_file_path)
    if "normalized" not in base_name and "raw" not in base_name:
        output_file_path = f"{base_name}_{'normalized' if normalize else 'raw'}{ext}"
    
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera with ID {camera_id}")
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Capturing {sample_size} frames at {capture_frequency_ms}ms intervals...")
    
    # Convert ms to seconds for time.sleep()
    capture_interval = capture_frequency_ms / 1000.0
    
    processed_frames = []
    frames_captured = 0
    
    try:
        while frames_captured < sample_size:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error capturing frame, retrying...")
                continue
            
            # Process the frame using the original model preprocessing
            processed_frame = original_model_preprocess(frame)
            
            # original_model_preprocess returns data in shape (1, 3, H, W)
            # We need to squeeze the batch dimension for proper stacking
            processed_frames.append(processed_frame.squeeze(0))
            
            frames_captured += 1
            
            # Display progress
            if frames_captured % 10 == 0:
                print(f"Captured {frames_captured}/{sample_size} frames")
            
            # Wait for the next capture interval
            time.sleep(capture_interval)
    
    finally:
        # Release the webcam
        cap.release()
        cv2.destroyAllWindows()
    
    # Stack all processed frames - already in NCHW format (3, H, W)
    calib_datas = np.stack(processed_frames)
    
    if normalize:
        # Transpose to NHWC for normalization since normalize_data expects this format
        calib_datas_nhwc = calib_datas.transpose(0, 2, 3, 1)
        
        # Normalize the data
        calib_datas_normalized, mean, std = normalize_data(calib_datas_nhwc)
        
        # Print statistics
        print("Statistics after normalization:")
        print_stats(calib_datas_normalized)
        
        # Transpose back to NCHW format for model input
        final_data = calib_datas_normalized.transpose(0, 3, 1, 2)/255.0
    else:
        # Use raw data, just print statistics
        print("Statistics (no normalization applied):")
        calib_datas_nhwc = calib_datas.transpose(0, 2, 3, 1)
        print_stats(calib_datas_nhwc)
        
        # Keep the original NCHW format
        final_data = calib_datas
    
    print(f'Calibration data shape: {final_data.shape}')
    
    # Save data
    np.save(file=output_file_path, arr=final_data)
    print(f"Calibration data saved to {output_file_path}")
    
    # Display some random images from the calibration data
    if display_percentage > 0:
        display_random_images(
            data=final_data, 
            percentage=display_percentage,
            is_nchw=True
        )


if __name__ == "__main__":
    
    # gen_calibdata_from_webcam(
    #     output_file_path='webcam_calibdata.npy',
    #     sample_size=256,
    #     capture_frequency_ms=200,
    #     camera_id=0,
    #     display_percentage=100,
    #     normalize=False
    # )
    
    





#use the images in folder to calibrate, keep the range 

