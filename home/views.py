import os , sys , re , base64 , json
from django.shortcuts import render_to_response , render
from django.views.generic import TemplateView
from imageai.Detection import ObjectDetection
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def processImage(input_file,output_file,fileName) :

    sys.path.append("..")


    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util


    MODEL_NAME = './object_detection/inference_graph'
    IMAGE_NAME = './object_detection/test/562000004755601.jpg'
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

    index = 0
    img_name = lists[len(lists)-1].split(".")
    image_list = []
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
            image_list.append(str(img_name[0])+"_"+str(index)+'.'+str(img_name[1]))
            cv2.imwrite("ocr/"+str(img_name[0])+"_"+str(index)+'.'+str(img_name[1]),crop_img)

    cv2.imwrite("result/"+str(lists[len(lists)-1]),image)
    cv2.imwrite("next/"+str(lists[len(lists)-1]),img)
    return image_list

def base64ToFilePath(base64Str , basePath = "./home/object_detection/test"):
    img_data = bytes(base64Str, 'utf-8')
    fileName = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + ".jpg"
    filePath = basePath + fileName
    with open(os.path.join(execution_path ,filePath ), "wb") as fh:
        fh.write(base64.decodebytes(img_data))
    return fileName

def filePathToBase64(filePath):
    base64Str = ''
    with open(os.path.join(execution_path ,filePath ), "rb") as image_file:
        base64Str = base64.b64encode(image_file.read())
    return 'data:image/png;base64,' + base64Str.decode("utf-8")

class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, '.\\home\\templates\\index.html', context=None)
    def post(self, request, **kwargs):
        image = request.POST['image']
        image = re.sub('^data:image\/[a-z]+;base64,','', image)
        #inputPath = "./home/object_detection/test/"
        #outputPath = "./home/object_detection/result/"
        inputPath = "./object_detection/test/"
        outputPath = "./object_detection/result/"
        fileName = base64ToFilePath(image)
        index = processImage(inputPath + fileName , outputPath + fileName ,fileName)
        base64Str = filePathToBase64(outputPath + fileName)
        print(json.dumps(result[1], separators=(',', ':')))
        return render(request, '.\\home\\templates\\index.html', { 'image' : base64Str } )