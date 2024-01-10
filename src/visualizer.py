import os
import cv2
import time
import av
import numpy as np
import streamlit as st
from ultralytics import YOLO
from src.utils import *
from src.ball import Ball
from src.hoop import Hoop
from src.score_count import Score 
from src.deep_sort.application_util import preprocessing
from src.deep_sort.deep_sort.detection import Detection
from src.deep_sort.deep_sort.tracker import Tracker
from src.deep_sort.deep_sort import nn_matching
from src.deep_sort.tools import generate_detections as gdet
from streamlit_webrtc import webrtc_streamer

class Visualizer:
    def __init__(self,model_path: os.path,threshold: float=0.5):
        self.ball= Ball()
        self.hoop= Hoop()
        self.model_path=model_path
        self.threshold=threshold
        self.model=YOLO(self.model_path)  # load a custom model
        self.font=cv2.FONT_HERSHEY_SIMPLEX
        self.lineType=cv2.LINE_AA
        self.score=Score()
        self.side='left' #or 'right'
        self.wait=10 #ms
        self.max_cosine_distance=0.99     #max cosine distance for similarity
        self.nn_budget=None
        self.nms_max_overlap=0.8         #discard multiple bounding boxes
        self.model_filename='models/mars-small128.pb' #Model to be used in the encoder
        self.class_names={0:'Ball', 1:'Hoop', 2:'Person'}    
        self.BALL_COLOR=(0,0,255)


    def record(self):
        st.title("Video Capture")
        cap=webrtc_streamer(key="my-stream" ,video_frame_callback=self.callback)

    def callback(self, frame_):

        #Convert recieved image to nd
        frame = frame_.to_ndarray(format="bgr24")

        # Encoder to convert the image into vector
        encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', self.max_cosine_distance, self.nn_budget)
        tracker = Tracker(metric, self.max_cosine_distance)
        
        H, W, _ = frame.shape
        self.score.frame_width = W

        # Detect objects in the frame
        results = self.model(frame)

        detections = []
        for result in results:
                names = []
                converted_boxes = []
                features = []
                scores = []
                cls_ids = []

                for r in result.boxes.data.tolist():
                    
                    x1, y1, x2, y2, cls_score, class_id = r
                    names.append(self.class_names[int(class_id)])
                    converted_boxes.append(utils.convert_boxes(x1, y1, x2, y2))
                    scores.append(cls_score)
                    cls_ids.append(class_id)

                features = encoder(frame, converted_boxes)
                names = np.array(names)

                detections += [Detection(bbox, score, feature, cls_id) for bbox, score, feature, cls_id in
                               zip(converted_boxes, scores, features, cls_ids)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        cls_ids = np.array([d.class_id for d in detections])
        indices = preprocessing.non_max_suppression(boxs, self.nms_max_overlap, scores)

        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
                # if not track.is_confirmed() or track.time_since_update > 1:
                #     continue
                if track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_id = track.class_id

                # Score Write
                cv2.putText(frame, str(self.score.lcount), (int(W*0.2), int(H*0.8)), self.font, 2, (153, 255, 102), 1, self.lineType)
                cv2.putText(frame, str(self.score.rcount), (int(W*0.8), int(H*0.8)), self.font, 2, (153, 255, 102), 1, self.lineType)

                if class_id == 0:  # Ball
                    self.ball.x1 = int(bbox[0])
                    self.ball.y1 = int(bbox[1])
                    self.ball.x2 = int(bbox[2])
                    self.ball.y2 = int(bbox[3])
                    self.ball.update()
                    self.score.detect_score(self.ball.x1, self.ball.y1, self.ball.x2, self.ball.y2,
                                            self.hoop.x1, self.hoop.y1, self.hoop.x2, self.hoop.y2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.BALL_COLOR, 4)
                elif class_id == 1:  # Hoop
                    self.hoop.x1 = int(bbox[0])
                    self.hoop.y1 = int(bbox[1])
                    self.hoop.x2 = int(bbox[2])
                    self.hoop.y2 = int(bbox[3])
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  utils.hoop_color(self.ball, self.hoop), 4)
                    self.score.set_side(self.hoop.x1, self.hoop.x2)
                else:  # Person
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  utils.average_color(frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                                  4)
                    cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                                (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.waitKey(self.wait)            

        return av.VideoFrame.from_ndarray(frame, format="rgb24")
