<!-- onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt  -ois input:1,3,480,640 # this works without any problem and also with 128*128




- onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt    -ois input:1,3,128,128 -cind "input" "maaz_calibdata.npy" "[[[0 ,0 ,0]]]" "[[[1, 1, 1]]]" 
    - this is the final command that worked, it is important to not transpose the calibration dataset, so the input shape is (1,3,128,128) but the calibration dataset shape is (BatchSize,128,128,3) as well.
    - the new permutation os axis of the new input of the model in the command must be same structured as the original structure of the model, but structure of channles for calibration dataset could be different from the model. 
    - onnx2tf automatically does this which is moving the channle axis to the last axis 


- onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt    -ois input:1,3,128,128 -cind "input" "maaz_calibdata.npy" "[[[[0]] ,[[0]] ,[[0]]]]" "[[[[1]],[[1]], [[1]]]]" -kat "input"
    - This command preserves the original input shape structure whichi is compatible with the calibration data shape structure as well. 

- offset could be negative, but scale was mostly from 0 to 5 in the original .onnx model

- the input image values to .onnx original model are range 0-255 -->

# BlazeFace

- `onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt    -ois input:1,3,128,128 -cind "input" "webcam_calibdata_raw.npy" "[[[[0]] ,[[0]] ,[[0]]]]" "[[[[1]],[[1]], [[1]]]]" -kat "input"`
    - This command must be executed in the this directory PATH = `CenterFace/onnx2tf-cli-docker`
    - This is the command used to quantize the first version of the centerface.
    - The "webcam_calibdata_raw.npy" dataset contains images captured from myselft.
    - The images in this dataset, are not normalized. 
    - The tweak is to give the mean 0 and std 1 to the convertor command.
    - The original `.onnx` model given to converter command has spatial resolution of input `H*w`, not `128*128`.
    - The range of image values in the calib dataset (`webcam_calibdata_raw.npy`) are from 0 - 255
    - `-kat` is to preserve the original order of axis of the input of the model, otherwise, it will be changed to `NHWC`



# DBFace 

- `onnx2tf -i dbface_keras_480x640_float32_nhwc.onnx -o quantization_rslts_dbface/ -oiqt   -cind "input__0" "calibdata_NHWC_480_640_DBFace.npy" "[[[[0,0,0]]]]" "[[[[1,1,1]]]]" -kat "input__0"`
    - The command used to quantize the network.
    - Calibration dataset is not on git because the file is big size.



- The videos of the detection results are in the folder `DetectedVideos`

<br><br>

## BalzeFace

![](DetectedVideos/blazeface.gif)

<br> <br>

## CenterFace
![](DetectedVideos/centerface.gif)

## DBFace

![](DetectedVideos/dbface.gif)


# Deployment of the Models on the STM32N6 device

| Model Name   | Extention | MACC         | Flash Size (Total) | RAM Size (Total) | Inference Time | Input Size | Landmarks | Inference Program |         
|--------------|-----------|--------------|--------------------|------------------|----------------|------------|-----------|-------------------|
|BlazeFace     | tflite    |   31,849,356 |      309 KB        |    641 KB        |   4.982 ms     |   128x128  |  &#10004; |   &#10004;        | 
|CenterFace    | tflite    | 101,344,367  |     2.7 MB         |    734 KB        |   10.59ms      |   128x128  |  &#10004; |  &#10004;         |
|DBFace        | tflite    | 2,085,454,097|    3 MB            |    13.1 MB       |    479.2ms     |   480x640  |  &#10004; | &#10004;          |
