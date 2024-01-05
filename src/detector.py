import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from src.ball import Ball
from src.hoop import Hoop
from src.person import Person
from src.score_count import Score 
from src.deep_sort.application_util import preprocessing
from src.deep_sort.deep_sort.detection import Detection
from src.deep_sort.deep_sort.tracker import Tracker
from src.deep_sort.deep_sort import nn_matching
from src.deep_sort.tools import generate_detections as gdet

#from src.tracker import Tracker

class Detector:
    def __init__(self,model_path: os.path=None,threshold: float=0.5):
        self.ball= Ball()
        self.hoop= Hoop()
        self.persons = []
        self.model_path=model_path
        self.threshold=threshold
        self.model=YOLO(self.model_path)  # load a custom model
        self.font=cv2.FONT_HERSHEY_SIMPLEX
        self.lineType=cv2.LINE_AA
        self.score=Score()
        #self.tracker=Tracker()
        self.wait=1 #ms
        self.max_cosine_distance=0.9     #max cosine distance for similarity
        self.nn_budget=None
        self.nms_max_overlap=0.8         #discard multiple bounding boxes
        self.model_filename='models/mars-small128.pb' #Model to be used in the encoder
        self.class_names={0:'Ball', 1:'Hoop', 2:'Person'}


    def cropper(self,frame,x1,y1,x2,y2):
        img=cv2.imread(frame)
        return img[x1:x2,y1:y2]
    
    def convert_boxes(self,x1,y1,x2,y2):
        return [x1,y1,np.absolute(x2-x1),np.absolute(y2-y1)]

    def detect(self,video_path: os.path=None):
        
        #Capture the video and find its parameters
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        H, W, _ = frame.shape
        img=frame
        
        #Encoder to convert the image into vector
        encoder=gdet.create_box_encoder(self.model_filename,batch_size=1)
        metric=nn_matching.NearestNeighborDistanceMetric('cosine',self.max_cosine_distance,self.nn_budget)
        tracker=Tracker(metric)
        #out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        while ret:
            #Time of frame
            t1 = time.time()
            results=self.model(frame)

            #Function to count the score
            self.score.detect_score(self.ball.x1,self.ball.y1,self.ball.x2,self.ball.y2,
                    self.hoop.x1,self.hoop.y1,self.hoop.x2,self.hoop.y2)
            #self.ball.x1_tracked,self.ball.y1_tracked,self.ball.x2_tracked, self.ball.y2_tracked=self.tracker.predict(self.wait)
            #Black box over the ball
            # cv2.rectangle(frame, (int(self.ball.x1), int(self.ball.y1)), (int(self.ball.x2), int(self.ball.y2)), (0, 0, 0), 4)
            #cv2.rectangle(frame, (int(self.ball.x1_tracked), int(self.ball.y1_tracked)), (int(self.ball.x2_tracked), int(self.ball.y2_tracked)), (255, 255, 255), 4)
            for result in results:
                #Code to find the number of people
                # num_persons=0
                # for r in result.boxes.data.tolist():
                #     _,_,_,_,_,class_id=r
                #     if class_id==2:
                #         num_persons+=1
                names=[]
                converted_boxes=[]
                features=[]
                scores=[]
                for r in result.boxes.data.tolist():
                    x1,y1,x2,y2,cls_score,class_id=r
                    if class_id==0: #Ball
                        self.ball.x1=x1
                        self.ball.x2=x2
                        self.ball.y1=y1
                        self.ball.y2=y2
                        #self.tracker.update(x1,y1,x2,y2)
                    if class_id==1: #Hoop
                        self.hoop.x1=x1
                        self.hoop.x2=x2
                        self.hoop.y1=y1
                        self.hoop.y2=y2

                    names.append(self.class_names[int(class_id)])
                    converted_boxes.append(self.convert_boxes(x1,y1,x2,y2))
                    scores.append(cls_score)

                features=encoder(img,converted_boxes)
                names=np.array(names)

                # detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                #   zip(converted_boxes, scores, names, features)]
                detections = [Detection(bbox, score, feature) for bbox, score, feature in
                  zip(converted_boxes, scores, features)]

                
                boxs = np.array([d.tlwh for d in detections])# x1,y1 ,l,h for detected objects
                scores = np.array([d.confidence for d in detections]) # confidence scores for detected objects
                #classes = np.array([d.class_name for d in detections])
                # indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                indices = preprocessing.non_max_suppression(boxs, self.nms_max_overlap, scores) 
                detections = [detections[i] for i in indices]   
                
                tracker.predict()
                tracker.update(detections)

                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

                for track in tracker.tracks:
                    print("************")
                    print(track.to_tlbr())
                    #print(track.features)
                    print("*****----------********")
                    if not track.is_confirmed() or track.time_since_update >1:
                        continue
                    bbox = track.to_tlbr()
                    # class_name= track.get_class()
                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]

                    cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),color, 4)
                    # cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    #             +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                                (255, 255, 255), 2)

            cv2.imshow('frame',frame)
            cv2.waitKey(self.wait)
            ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()














# for r in result.boxes.data.tolist():
                #     x1,y1,x2,y2,cls_score,class_id=r
                #     #features=encoder(img,[x1,y1,x2-x1,y2-y1])
                    #Location update for ball and hoop
                    # if class_id==0: #Ball
                    #     self.ball.x1=x1
                    #     self.ball.x2=x2
                    #     self.ball.y1=y1
                    #     self.ball.y2=y2
                    #     #self.tracker.update(x1,y1,x2,y2)
                    # if class_id==1: #Hoop
                    #     self.hoop.x1=x1
                    #     self.hoop.x2=x2
                    #     self.hoop.y1=y1
                    #     self.hoop.y2=y2

                    # if cls_score > self.threshold:
                    #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    #     cv2.putText(
                    #                 frame, 
                    #                 result.names[int(class_id)].upper(),
                    #                 (int(x1), int(y1 - 10)),
                    #                 self.font,
                    #                 1.3,
                    #                 (0, 255, 0),
                    #                 3,
                    #                 self.lineType
                    #                 )
                    #     cv2.putText(frame,str(self.score.count), (300,150), self.font, 5,(0, 0, 255),3,self.lineType)
                    #     cv2.putText(frame,str(self.score.reset), (1000,150), self.font, 5,(0, 0, 255),3,self.lineType)
                    #     cv2.putText(frame,str(num_persons), (300,600), self.font, 5,(0, 0, 255),3,self.lineType)
                #out.write(frame)