
# Required Libraries

import tensorflow as tf
import numpy as np
import os
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2
import glob

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# List of the labels for each box (Model labels) 

# Label model path variable
PATH_TO_LABELS = 'labels/mscoco_label_map.pbtxt'

# Convert the labels into a dict
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Model path/name
model_name = 'our_model/ssd_mobilenet_v1_coco_2017_11_17'

# Load model
detection_model = tf.saved_model.load(model_name)


class object_detection():
  def __init__(self):
    pass

  def run_inference_for_single_image(self, model, image):
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`. for example doesn't want a shape like this (480, 640, 3),
    # instead it needs a shape like this (1, 480, 640, 3)
    input_tensor = input_tensor[tf.newaxis,...]


    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))

    output_dict = {key:value[0, :num_detections].numpy() 
                   for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
      # Reframe the the bbox mask to the image size.
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                 image.shape[0], image.shape[1])      
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                         tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
      
    return output_dict

  def show_inference(self, model, img):
    
    # convert the img to numpy array
    image_np = np.array(img)

    # Actual detection.
    output_dict = self.run_inference_for_single_image(model, image_np)
    

    # Visualization of the results of a detection. 
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    tem = np.array(image_np)
    image = cv2.cvtColor(tem, cv2.COLOR_RGB2BGR)
    return image
    

  def web_cam_detect(self):

    # allocate a cam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    while True:
      # Capture frame-by-frame
      ret, frame = cap.read()

      # if frame is read correctly ret is True
      if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


      img = self.show_inference(detection_model, frame)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

      # play the video
      cv2.imshow('object_detection', img)

      # wait until the user presses q
      if cv2.waitKey(1) == ord('q'):

        break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


  def detect_img(self):
    # detect all the img
    imgs = glob.glob("images/test/*.jpg")
    # declate a counter
    c = 0

    for i in imgs:
      # read all the imgs
      img = cv2.imread(i)


      # detect the img
      img = self.show_inference(detection_model, img)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

      # save the img
      cv2.imwrite('images/detected/image'+str(c)+'.jpg', img)
      c+=1



if __name__=='__main__':
  # make an object of object_detection
  od = object_detection()

  # open an object detection on web cam 
  od.web_cam_detect()

  # detect object on images
  #od.detect_img()

 