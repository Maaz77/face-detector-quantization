<!-- onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt  -ois input:1,3,480,640 # this works without any problem and also with 128*128




- onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt    -ois input:1,3,128,128 -cind "input" "maaz_calibdata.npy" "[[[0 ,0 ,0]]]" "[[[1, 1, 1]]]" 
    - this is the final command that worked, it is important to not transpose the calibration dataset, so the input shape is (1,3,128,128) but the calibration dataset shape is (BatchSize,128,128,3) as well.
    - the new permutation os axis of the new input of the model in the command must be same structured as the original structure of the model, but structure of channles for calibration dataset could be different from the model. 
    - onnx2tf automatically does this which is moving the channle axis to the last axis 


- onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt    -ois input:1,3,128,128 -cind "input" "maaz_calibdata.npy" "[[[[0]] ,[[0]] ,[[0]]]]" "[[[[1]],[[1]], [[1]]]]" -kat "input"
    - This command preserves the original input shape structure whichi is compatible with the calibration data shape structure as well. 

- offset could be negative, but scale was mostly from 0 to 5 in the original .onnx model

- the input image values to .onnx original model are range 0-255 -->

- `onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt    -ois input:1,3,128,128 -cind "input" "webcam_calibdata_raw.npy" "[[[[0]] ,[[0]] ,[[0]]]]" "[[[[1]],[[1]], [[1]]]]" -kat "input"`
    - This command must be executed in the this directory PATH = `CenterFace/onnx2tf-cli-docker`
    - This is the command used to quantize the first version of the centerface.
    - The "webcam_calibdata_raw.npy" dataset contains images captured from myselft.
    - The images in this dataset, are not normalized. 
    - The tweak is to give the mean 0 and std 1 to the convertor command.
    - The original `.onnx` model given to converter command has spatial resolution of input `H*w`, not `128*128`.
    - The range of image values in the calib dataset (`webcam_calibdata_raw.npy`) are from 0 - 255
    - `-kat` is to preserve the original order of axis of the input of the model, otherwise, it will be changed to `NHWC`





# Comparison of the BlazeFace and CenterFace models both in INT8 precision (.tflite)

![Original image](DetectedVideos/originalvideo.gif)

![CenterFace](DetectedVideos/centerface.gif)

![BlazeFace](DetectedVideos/blazeface.gif)

<video src="DetectedVideos/centerface.mp4" controls></video>
<video src="DetectedVideos/blazeface.mp4" controls></video>