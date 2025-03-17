import common
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import onnxruntime as ort
import time  # Add time import for FPS calculation


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


def detect(session, image, threshold=0.4, nms_iou=0.5):
    # Run inference using ONNX Runtime
    outputs = session.run(None, {'input:0': image})
    
    # Parse outputs - adjust indices based on your ONNX model's output order
    # Typically in ONNX models the order might be different from TFLite
    lm = outputs[0]  # landmark
    box = outputs[1]  # box
    hm = outputs[2]   # heatmap
    
    # If the ONNX model outputs are in NHWC format, transpose them to NCHW
    if lm.shape[1] != 10:  # Check if channels dimension is not in the right place
        lm = lm.transpose((0, 3, 1, 2))  # 1,h,w,10 -> 1,10,h,w
        box = box.transpose((0, 3, 1, 2))  # 1,h,w,4 -> 1,4,h,w
        hm = hm.transpose((0, 3, 1, 2))  # 1,h,w,1 -> 1,1,h,w
    
    hm = torch.from_numpy(hm).clone()
    box = torch.from_numpy(box).clone()
    landmark = torch.from_numpy(lm).clone()
    
    # Rest of the processing is identical
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices // hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
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
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        
        scale_x = input_width / (hm_width * stride)
        scale_y = input_height / (hm_height * stride)
        xyrb = xyrb * [scale_x, scale_y, scale_x, scale_y]
        
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        x5y5_scaled = np.array(x5y5)
        x5y5_scaled[:5] *= scale_x
        x5y5_scaled[5:] *= scale_y
        
        box_landmark = list(zip(x5y5_scaled[:5], x5y5_scaled[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))

    return nms(objs, iou=nms_iou)


def camera_demo(MODEL_PATH):
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
        dotranspose = False
    else:
        dotranspose = True
    
    # Initialize ONNX Runtime session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 5
    session = ort.InferenceSession(MODEL_PATH, sess_options=session_options, providers=['CPUExecutionProvider'])
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, inputwidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, inputheight)

    # FPS calculation variables
    prev_frame_time = 0
    curr_frame_time = 0
    fps_values = []
    fps_smoothing = 10  # Number of frames to average FPS over

    while True:
        # Start timing for FPS calculation
        curr_frame_time = time.time()
        
        ok, frame = cap.read()
        frameheight = frame.shape[0]
        framewidth = frame.shape[1]
        if not ok:
            continue
        
        img = cv2.resize(frame, (inputwidth, inputheight))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, :]

        img = ((img) / 255.0) 
        if dotranspose:
            img = img.transpose((0, 3, 1, 2))

        objs = detect(session, img)

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
        
        # Calculate FPS
        if prev_frame_time > 0:
            fps = 1 / (curr_frame_time - prev_frame_time)
            fps_values.append(fps)
            # Keep only the last fps_smoothing values
            if len(fps_values) > fps_smoothing:
                fps_values.pop(0)
            # Calculate average FPS
            avg_fps = sum(fps_values) / len(fps_values)
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        prev_frame_time = curr_frame_time

        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Update with your ONNX model paths
    MODEL_PATH_1 = "Models/dbface_keras_480x640_float32_nhwc.onnx"
    camera_demo(MODEL_PATH=MODEL_PATH_1)
