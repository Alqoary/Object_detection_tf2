import tensorflow as tf
import numpy as np
import os
import sys

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2
import pathlib

import glob


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


  # List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'labels/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



model_name = 'our_model/ssd_mobilenet_v1_coco_2017_11_17'
detection_model = tf.saved_model.load(model_name)

#print(detection_model.signatures['serving_default'].inputs)


class object_detection():
  def __init__(self):
    pass
  def run_inference_for_single_image(self, model, image):
    #image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
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
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
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
    #print(type(image))
    #print(image)
    return image
    

#for image_path in TEST_IMAGE_PATHS:
#  show_inference(detection_model, image_path)


  def web_cam_detect(self):


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
      # Our operations on the frame come here
      # Display the resulting frame
      img = self.show_inference(detection_model, frame)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      cv2.imshow('object_detection', img)

      #cv.imshow('frame', gray)
      if cv2.waitKey(1) == ord('q'):
          break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




  def detect_img(self):
    imgs = glob.glob("images/test/*.jpg")
    c = 0
    for i in imgs:

      img = cv2.imread(i)
        # if frame is read correctly ret is True


      img = self.show_inference(detection_model, img)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      cv2.imwrite('images/detected/image'+str(c)+'.jpg', img)
      c+=1

    def get_imgs_path(self):
      pass

if __name__=='__main__':
  od = object_detection()
  #od.web_cam_detect()
  od.detect_img()

  print('nice')