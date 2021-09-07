# Object_detection_tf2
object_detection in tensorflow2

REQUIRED LIBRARIES

$ pip install tensorflow
$ pip install tensorflow-object-detection-api
$ pip install opencv-python

If you have a gpu go ahead and download the appropriate CUDA version for you python version


This code was written to simplify the object detection in tensorflow2 for beginners. It is composed of one simple python file which has a class with different methods to detect images or web cam directly.

you can applay different pre trained models with the code just change the model <path/name> and the labels <path/label> files.


you have to main methods which you can use:
1- web_cam_detect: to detect objects on web cam
2- detect_img: to detect object from <images\test> and export them into <images\trained>

