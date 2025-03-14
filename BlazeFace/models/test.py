import os
import tensorflow as tf

def check_float32_weights_biases(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    
    for tensor in tensor_details:
        if tensor['dtype'] == tf.float32:
            return True
    return False

def main():
    directory = '/Users/maaz/Desktop/ST-face-monitoring/Face-Detectors-Exp/webcamFaceDetection_stillQ/models'
    for filename in os.listdir(directory):
        if filename.endswith(".tflite"):
            model_path = os.path.join(directory, filename)
            if check_float32_weights_biases(model_path):
                print(filename)

if __name__ == "__main__":
    main()



