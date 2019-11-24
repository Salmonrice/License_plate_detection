import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'test/562000004755601.jpg'
lists = IMAGE_NAME.split("/")


CWD_PATH = os.getcwd()


PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')


PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')


PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)


NUM_CLASSES = 1


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')


detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


num_detections = detection_graph.get_tensor_by_name('num_detections:0')



#for x in range(len(lists)):

image = cv2.imread(IMAGE_NAME)
img = image.copy()
print(lists[len(lists)-1])
image_expanded = np.expand_dims(image, axis=0)

(boxes, scores, classes, num) = sess.run(
    	[detection_boxes, detection_scores, detection_classes, num_detections],
    	feed_dict={image_tensor: image_expanded})


vis_util.visualize_boxes_and_labels_on_image_array(
    	image,
    	np.squeeze(boxes),
    	np.squeeze(classes).astype(np.int32),
    	np.squeeze(scores),
    	category_index,
    	use_normalized_coordinates=True,
    	line_thickness=8,
    	min_score_thresh=0.70)


index = 0;
int count
img_name = lists[len(lists)-1].split(".")
for index,value in enumerate(classes[0]):
	object_dir = {}
	if scores[0,index] > 0.7:
		ymin = boxes[0,index,0]
		xmin = boxes[0,index,1]
		ymax = boxes[0,index,2]
		xmax = boxes[0,index,3]
		im_height,im_width = img.shape[:2]
		(xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
		crop_img = img[int(yminn):int(ymaxx), int(xminn):int(xmaxx)]
			
		cv2.imwrite("ocr/"+str(img_name[0])+"_"+str(index)+'.jpg',crop_img)


cv2.imwrite("result/"+str(lists[len(lists)-1]),image)
cv2.imwrite("next/"+str(lists[len(lists)-1]),img)
