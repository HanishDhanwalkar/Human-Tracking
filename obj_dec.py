import cv2
import argparse
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img', default="./imgs/1.jpg", type=str, help='path to <image>.jpg')
parser.add_argument('--classes', default="./classes.txt", type=str, help='path to <classes>.txt containing all the classes')
parser.add_argument('--yolo_config', default="yolov3.cfg", type=str, help='path to yolo config file eg. yolov3.cfg')
parser.add_argument('--yolo_weights', default="yolov3.weights", type=str, help='path to yolo weights file eg. yolov3.weights')


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    # print('class_id: ',class_id)
    label = str(classes[class_id])
    conf = str(round(confidence, ndigits=3))
    # print(conf)
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, conf, (x-10,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, label, (x-10,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def crop_class(img, box):
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h) 
    crop_img = img[y:y+h, x:x+w]
    cv2.imshow("cropped", crop_img)
    return crop_img

def crop_class_person(img, boxes, labels):
    for i, (box, lab) in enumerate(zip(boxes, labels)):
        if lab == 'person':
            print(i)
            # crop_class(img=img, box=box)

if __name__ == '__main__':
    args = parser.parse_args()  # get arguments from command line
    
    img_path = args.img
    image = cv2.imread(img_path)
    Width = image.shape[1]
    Height = image.shape[0]
    
    with open(args.classes, 'r') as f:
        classes = [clas.strip() for clas in f.readlines()]


    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Load YOLO
    net = cv2.dnn.readNet(args.yolo_config, args.yolo_weights)
    
    # create input blob 
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)
    
    outs = net.forward(get_output_layers(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            # print(1)
            box = boxes[i]
            class_id = class_ids[i]
        except:
            # print(2)
            i = i[0]
            box = boxes[i]
            class_id = class_ids[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # draw_prediction(image, class_id, confidences[i], round(x), round(y), round(x+w), round(y+h))

    # cv2.imshow("object detection", image)
    crop_class_person(image, boxes, class_ids)
    cv2.imshow("object detection", crop_class_person(image, boxes, class_ids))
    cv2.waitKey()
        
    # cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()