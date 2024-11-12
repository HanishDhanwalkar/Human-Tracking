import time
import os
import torch
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class YoloDectector():
    def __init__(self) -> None:
        self.model = self.load_model()
        with open('classes.txt', 'r') as f:
            self.classes = [clas.strip() for clas in f.readlines()]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        
    def load_model(self):
        model = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')
        return model 
        
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def get_output_layers(self):
        layer_names = self.model.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

        return output_layers

    def plot_boxes(self, frame, height, width, conf):
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False) 
        
        self.model.setInput(blob)
        outs = self.model.forward(self.get_output_layers())
    
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = conf
        nms_threshold = 0.4
        
        detections = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, h, w])


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
            x, y, w, h = round(x), round(y), round(w), round(h)
            
            label = str(detector.class_to_label(class_id))

            conf = str(round(confidence, ndigits=3))
            # print(conf)
            if label == 'person':
                detections.append(([x, y, h, w], conf, label))
            
        return frame, detections
    

if __name__ == '__main__':
    if 'yolov3s.cfg' not in os.listdir():
        print('yolov3s.cfg not found. Download yolo weights')

    if 'yolov3.weights' not in os.listdir():
        print('yolov3.weights not found. Download yolo weights')
    
    if 'yolov3.weights' in os.listdir():
        print('Found yolo weights')
        detector = YoloDectector()
        
    object_tracker = DeepSort(
                max_age=100000, # allow tracker to miss up to this many frames before discarding bounding box/track
                n_init=5, # initialises 2 frames 
                nms_max_overlap=1.0,
                max_cosine_distance=0.2)
        
    
    print('\n\n\n\n****************')
    # cap = cv2.VideoCapture(video_path) 
    
    
    video_path = r"main/sample_vids/sample1_.mp4" 
    
    if os.path.exists(video_path):
        print('FOUND video file')
        cap = cv2.VideoCapture(video_path) 
    else:
        print('NOT FOUND video file: ', video_path)
        cap = cv2.VideoCapture(0)  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Press 'q' to quit.")
    
    num_characters = []
    
    while True:
        ret, img = cap.read()
        start = time.perf_counter()
        
        if not ret:
            break
        
        img, detections = detector.plot_boxes(img, img.shape[0], img.shape[1], 0.8)
        
        # for detection in detections:
        #     # detections.append(([x, y, h, w], conf, label))
        #     if detection[2] == 'person':
        #         x, y, w, h = detection[0]
        #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #         cv2.putText(img, detection[2], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        
        tracks = object_tracker.update_tracks(detections, frame=img)
        for i, track in enumerate(tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            if track_id not in num_characters:
                num_characters.append(track_id)
                
            bbox = ltrb
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
            cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        
        print(f'number of characterds so far: {len(num_characters)}')
            
            
        end = time.perf_counter()
        fps = 1 / (end - start)
        
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Human Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()