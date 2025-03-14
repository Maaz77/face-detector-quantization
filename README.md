onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt  -ois input:1,3,480,640 # this works without any problem and also with 128*128




- onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt    -ois input:1,3,128,128 -cind "input" "maaz_calibdata.npy" "[[[0 ,0 ,0]]]" "[[[1, 1, 1]]]" 
    - this is the final command that worked, it is important to not transpose the calibration dataset, so the input shape is (1,3,128,128) but the calibration dataset shape is (BatchSize,128,128,3) as well.
    - the new permutation os axis of the new input of the model in the command must be same structured as the original structure of the model, but structure of channles for calibration dataset could be different from the model. 
    - onnx2tf automatically does this which is moving the channle axis to the last axis 


- onnx2tf -i centerface_1x3xHxW.onnx -o quantization_rslts/ -oiqt    -ois input:1,3,128,128 -cind "input" "maaz_calibdata.npy" "[[[[0]] ,[[0]] ,[[0]]]]" "[[[[1]],[[1]], [[1]]]]" -kat "input"
    - This command preserves the original input shape structure whichi is compatible with the calibration data shape structure as well. 

- offset could be negative, but scale was mostly from 0 to 5 in the original .onnx model

- the input image values to .onnx original model are range 0-255

