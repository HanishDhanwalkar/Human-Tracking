import time
import os
import torch
import cv2
import numpy as np

class YoloDectector():
    def __init__(self, model_name) -> None:
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        
    def load_model(self, model_name):
        if model_name: 
            model = torch.hub.load(r'C:/Users/Hanish/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path=model_name, force_reload=True, source='local') 
        else: 
            model = torch.hub.load('ultralytics/yolovs', 'yolov5s', pretrained=True) 
        return model 
        
    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))
        
        results = self.model(frame)
        # print(results)
        # print(results.xyxyn)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # print(labels, cord)
        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame, height, width, confidence):
        labels, cord = results
        detections = []
        
        n = len(labels)
        x_shape, y_shape = width, height
        
        for i in range(n):
            row = cord[i]
            if i == 1:
                print(row)
            if row[4] >= confidence:
                x1,  y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape),int(row[3] * y_shape)

                if self.class_to_label(labels[i]) == 'person':
                    x_center = x1 + (x2 - x1)/2
                    y_center = y1 + (y2 - y1)/2
                    
                    tlwh = np.array([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                    confidence = float(row[4].item())
                    feature = 'person'
                    
                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), 'person'))
                    
        return frame, detections
    

if __name__ == '__main__':
    if 'yolov5s.pt' not in os.listdir():
        print('Downloading yolo weights')
        detector = YoloDectector(model_name=None)
    else:
        print('Found yolo weights')
        detector = YoloDectector(model_name='yolov5s.pt')
        
    
    print('\n\n\n\n****************')
    video_path = r"main/sample_vids/3.mp4" 
    if os.path.exists(video_path):
        print('FOUND video file')
        cap = cv2.VideoCapture(video_path) 
    else:
        print('NOT FOUND video file: ', video_path)
        cap = cv2.VideoCapture(0)  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Press 'q' to quit.")
    
    while True:
        ret, img = cap.read()
        start = time.perf_counter()
        
        if not ret:
            break
        results = detector.score_frame(img)
        img, detections = detector.plot_boxes(results, img, img.shape[0], img.shape[1], 0.5)
        
        for detection in detections:
            track_id = detection[2]
            bbox = detection[0]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
            cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        
        # tracks = results[1]
        # for track in tracks:
        #     ltrb = track.to_ltrb()
            
        #     bbox = ltrb
        #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
        #     cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        
        
        end = time.perf_counter()
        fps = 1 / (end - start)
        
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Human Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()