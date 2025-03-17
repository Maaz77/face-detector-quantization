from . import common
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tensorflow.lite.python.interpreter import Interpreter


def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(interpreter, input_blob, output_blob, image, threshold=0.4, nms_iou=0.5):
    

    interpreter.set_tensor(input_blob[0]['index'], image)
    interpreter.invoke()

    # hm, box, landmark = outputs['1028'], outputs['1029'], outputs['1027']
    if not indexswap:
        lm = interpreter.get_tensor(output_blob[0]['index']).transpose((0,3,1,2)) # 1,h,w,10
        box = interpreter.get_tensor(output_blob[1]['index']).transpose((0,3,1,2)) # 1,h,w,4
        hm = interpreter.get_tensor(output_blob[2]['index']).transpose((0,3,1,2)) # 1,1,h,w
    else:
        lm = interpreter.get_tensor(output_blob[1]['index']).transpose((0,3,1,2)) # 1,h,w,10
        box = interpreter.get_tensor(output_blob[0]['index']).transpose((0,3,1,2)) # 1,h,w,4
        hm = interpreter.get_tensor(output_blob[2]['index']).transpose((0,3,1,2)) # 1,1,h,w
        
    
    hm = torch.from_numpy(hm).clone()
    box = torch.from_numpy(box).clone()
    landmark = torch.from_numpy(lm).clone()
    
    # print('hm.shape:', hm.shape)
    # print('hm:', hm.dtype)
    # print('box.shape:', box.shape)
    # print('landmark.shape:', landmark.shape)
    

    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    # Convert linear indices to 2D coordinates considering both dimensions
    ys = list((indices // hm_width).int().data.numpy())  # Row indices
    xs = list((indices % hm_width).int().data.numpy())   # Column indices
    # Validate coordinates are within bounds
    assert all(0 <= y < hm_height for y in ys), "Y coordinates out of bounds"
    assert all(0 <= x < hm_width for x in xs), "X coordinates out of bounds"
    
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    
    # Get input image dimensions
    input_height, input_width = image.shape[1:3]
    
    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        # Scale coordinates to feature map size
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        
        # Scale coordinates back to input image size
        scale_x = input_width / (hm_width * stride)
        scale_y = input_height / (hm_height * stride)
        xyrb = xyrb * [scale_x, scale_y, scale_x, scale_y]
        
        # Scale landmarks
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        x5y5_scaled = np.array(x5y5)
        x5y5_scaled[:5] *= scale_x  # Scale x coordinates
        x5y5_scaled[5:] *= scale_y  # Scale y coordinates
        
        box_landmark = list(zip(x5y5_scaled[:5], x5y5_scaled[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))

    return nms(objs, iou=nms_iou)



def camera_demo(MODEL_PATH):
    
    global indexswap  
    
    if ("480x640" in MODEL_PATH):
        inputheight = 480
        inputwidth = 640
    elif ("256x256" in MODEL_PATH):
        inputheight = 256 
        inputwidth = 256
    elif ("512x512" in MODEL_PATH):
        inputheight = 512
        inputwidth = 512
    
    if ("nhwc" in MODEL_PATH):
        dotranspose  = False
    else:
        dotranspose = True
    
    if ("me" in MODEL_PATH):
        indexswap = True
    else:
        indexswap = False
    
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=5)
    interpreter.allocate_tensors()
    input_blob = interpreter.get_input_details()
    output_blob = interpreter.get_output_details()
        
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, inputwidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, inputheight)

    for i in range(1000):
        ok, frame = cap.read()
        frameheight = frame.shape[0]
        framewidth = frame.shape[1]
        if not ok:
            continue
        img = cv2.resize(frame, (inputwidth,inputheight))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, :]

        

        img = ((img  ) / 255.0) 
        #img =  (img - np.min(img))/(np.max(img) - np.min(img))
        if dotranspose:
            img = img.transpose((0,3,1,2))
            
        #img = img.transpose((0,3,1,2))

        objs = detect(interpreter, input_blob, output_blob, img)
        

        for obj in objs:
            
            
            
            
            if inputheight != frameheight or inputwidth != framewidth:
                scale_x = framewidth / inputwidth
                scale_y = frameheight / inputheight
                obj.x *= scale_x
                obj.y *= scale_y
                obj.r *= scale_x
                obj.b *= scale_y
                if obj.landmark is not None:
                    obj.landmark = [(x * scale_x, y * scale_y) for x, y in obj.landmark]
                    
            common.drawbbox(frame, obj)


        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def video_detection(input_video_path, model_path, output_video_path):
    
    global indexswap  
    
    # Model input size determination based on model path
    if ("480x640" in model_path):
        inputheight = 480
        inputwidth = 640
    elif ("256x256" in model_path):
        inputheight = 256 
        inputwidth = 256
    elif ("512x512" in model_path):
        inputheight = 512
        inputwidth = 512
    
    # Determine if transpose is needed
    if ("nhwc" in model_path):
        dotranspose = False
    else:
        dotranspose = True
    
    # Determine if index swap is needed
    if ("me" in model_path):
        indexswap = True
    else:
        indexswap = False
    
    # Load the model
    interpreter = Interpreter(model_path=model_path, num_threads=5)
    interpreter.allocate_tensors()
    input_blob = interpreter.get_input_details()
    output_blob = interpreter.get_output_details()
        
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You could also use 'mp4v' for .mp4 output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (framewidth, frameheight))
    
    # Process each frame
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        frame_index += 1
        print(f"Processing frame {frame_index}/{frame_count}")
        
        # Resize frame to model input dimensions
        img = cv2.resize(frame, (inputwidth, inputheight))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, :]
        
        # Normalize
        img = ((img) / 255.0) 
        
        # Transpose if needed
        if dotranspose:
            img = img.transpose((0, 3, 1, 2))
            
        # Perform detection
        objs = detect(interpreter, input_blob, output_blob, img)
        
        # Draw bounding boxes and scale coordinates if needed
        for obj in objs:
            if inputheight != frameheight or inputwidth != framewidth:
                scale_x = framewidth / inputwidth
                scale_y = frameheight / inputheight
                obj.x *= scale_x
                obj.y *= scale_y
                obj.r *= scale_x
                obj.b *= scale_y
                if obj.landmark is not None:
                    obj.landmark = [(x * scale_x, y * scale_y) for x, y in obj.landmark]
                    
            common.drawbbox(frame, obj)
        
        # Write the frame to the output video
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video processing completed. Output saved to {output_video_path}")

if __name__ == "__main__":
    
    MODEL_PATH_1 = "Models/dbface_keras_480x640_weight_quant_nhwc.tflite"
    MODEL_PATH_2 = "Models/dbface_keras_480x640_integer_quant_nhwc.tflite"
    MODEL_PATH_3 = "Models/dbface_keras_256x256_weight_quant_nhwc.tflite"
    MODEL_PATH_4 = "Models/dbface_keras_256x256_integer_quant_nhwc.tflite"
    
    MODEL_PATH_5 = "Models/dbface_480x640_INT8_nhwc_full_integer_quant_me.tflite" #my qunatization with my calib dataset. does not work because of the int8 input type brings a lot of conflicts
    MODEL_PATH_6 = "Models/dbface_480x640_INT8_nhwc_integer_quant_me.tflite"  #my qunatization with my calib dataset
    
    
    # models 2 and 6 are both INT8 and works well but i think the one i quantized is better that is the model 6 
    camera_demo(MODEL_PATH=MODEL_PATH_6)
