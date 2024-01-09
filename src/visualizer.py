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
#from src.person import Person
from src.score_count import Score 
from src.deep_sort.application_util import preprocessing
from src.deep_sort.deep_sort.detection import Detection
from src.deep_sort.deep_sort.tracker import Tracker
from src.deep_sort.deep_sort import nn_matching
from src.deep_sort.tools import generate_detections as gdet
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# class VideoTransformer(VideoTransformerBase):
#     def transform(self, frame):
#             # Process frame with OpenCV here
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
#             return frame



class Visualizer:
    def __init__(self,model_path: os.path=os.path.join('..','dataset','last_50_re.pt')
,threshold: float=0.5):
        self.ball= Ball()
        self.hoop= Hoop()
        #self.persons = []
        self.model_path=model_path
        self.threshold=threshold
        self.model=YOLO(self.model_path)  # load a custom model
        self.font=cv2.FONT_HERSHEY_SIMPLEX
        self.lineType=cv2.LINE_AA
        self.score=Score()
        self.side='left' #or 'right'
        self.wait=1 #ms
        self.max_cosine_distance=0.99     #max cosine distance for similarity
        self.nn_budget=None
        self.nms_max_overlap=0.8         #discard multiple bounding boxes
        self.model_filename='models/mars-small128.pb' #Model to be used in the encoder
        self.class_names={0:'Ball', 1:'Hoop', 2:'Person'}    
        self.BALL_COLOR=(0,0,255)


    def record(self):
        cap=webrtc_streamer(key="my-stream" ,video_frame_callback=self.callback)
        st.title("Video Capture")
        frame_placeholder=st.empty()
        stop_button_pressed=st.button("Stop")


    # def callback(frame):
    #         img = frame.to_ndarray(format="bgr24")

    #         #img = cv2.cvtColor(cv2.Canny(img, threshold1, threshold2), cv2.COLOR_GRAY2BGR)
    #         #st.video(av.VideoFrame.from_ndarray(img, format="bgr24"))

    #         #Capture the video and find its parameters
    #         # imge = camera_input_live()
    #         cap = cv2.VideoCapture(img)
        
    #     #while cap.read():
            
    #         #Encoder to convert the image into vector
    #         encoder=gdet.create_box_encoder(self.model_filename,batch_size=1)
    #         metric=nn_matching.NearestNeighborDistanceMetric('cosine',self.max_cosine_distance,self.nn_budget)
    #         tracker=Tracker(metric,self.max_cosine_distance)
            
    #         #Record
    #         #out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    #         while True:
    #             #Time of frame
    #             t1 = time.time()
    #             results=self.model(frame)

    #             for result in results:
    #                 names=[]
    #                 converted_boxes=[]
    #                 features=[]
    #                 scores=[]
    #                 cls_ids=[]
    #                 for r in result.boxes.data.tolist():
                        
    #                     x1,y1,x2,y2,cls_score,class_id=r
    #                     names.append(self.class_names[int(class_id)])
    #                     converted_boxes.append(utils.convert_boxes(x1,y1,x2,y2))
    #                     scores.append(cls_score)
    #                     cls_ids.append(class_id)

    #                 features=encoder(frame,converted_boxes)
    #                 names=np.array(names)

    #                 detections = [Detection(bbox, score, feature,cls_id) for bbox, score, feature,cls_id in
    #                 zip(converted_boxes, scores, features,cls_ids)]

                    
    #                 boxs = np.array([d.tlwh for d in detections])# x1,y1 ,l,h for detected objects
    #                 scores = np.array([d.confidence for d in detections]) # confidence scores for detected objects
    #                 cls_ids = np.array([d.class_id for d in detections])
    #                 indices = preprocessing.non_max_suppression(boxs, self.nms_max_overlap, scores) 
                    
    #                 #Only keep the indices filtered by non_max_suppression
    #                 detections = [detections[i] for i in indices]   
                    
    #                 tracker.predict()
    #                 tracker.update(detections)

    #             for track in tracker.tracks:
    #                 if not track.is_confirmed() or track.time_since_update >1:
    #                     continue
    #                 bbox = track.to_tlbr()
    #                 class_id = track.class_id 
                
    #                 #Score Write
    #                 cv2.putText(frame,str(self.score.lcount), (300,150), self.font, 5,(0, 0, 255),3,self.lineType)
    #                 cv2.putText(frame,str(self.score.rcount), (1000,150), self.font, 5,(0, 0, 255),3,self.lineType)
                        
    #                 if class_id==0:  #Ball
    #                     self.ball.x1=int(bbox[0])
    #                     self.ball.y1=int(bbox[1])
    #                     self.ball.x2=int(bbox[2])
    #                     self.ball.y2=int(bbox[3])
    #                     self.ball.update()
    #                     self.score.detect_score(self.ball.x1,self.ball.y1,self.ball.x2,self.ball.y2,
    #                     self.hoop.x1,self.hoop.y1,self.hoop.x2,self.hoop.y2)          
    #                     cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),self.BALL_COLOR, 4)
    #                 elif class_id==1: #Hoop
    #                     self.hoop.x1=int(bbox[0])
    #                     self.hoop.y1=int(bbox[1])
    #                     self.hoop.x2=int(bbox[2])
    #                     self.hoop.y2=int(bbox[3]) 
    #                     cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),utils.hoop_color(self.ball,self.hoop), 4)
    #                     self.score.set_side(self.hoop.x1,self.hoop.x2)
    #                 else:             #Person
    #                     cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),utils.average_color(frame,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])), 4)
    #                     cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
    #                                             (255, 255, 255), 2)

                
    #             frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #             frame_placeholder.image(frame, channels="RGB")
    #             cv2.waitKey(self.wait)
    #             # if len(self.ball.queue)>=3:
    #             #     pts=np.reshape(self.ball.queue,(len(self.ball.queue), 2))
    #             #     print('pppppppppppppppppppppp')
    #             #     print(pts)
    #             #     pts=np.array(pts)
    #             #     cv2.polylines(frame, np.int32([pts]), False, self.BALL_COLOR,4)

    #         # if not ret:
    #         #     st.write("Ended")
    #         #     break

    #         # cv2.imshow('frame',frame)
    #         # cv2.waitKey(self.wait)
    #         #ret, frame = cap.read()
    #         cap.release()
    #         cv2.destroyAllWindows()





    #         return av.VideoFrame.from_ndarray(frame, format="bgr24")

    def callback(self, frame_):
        print("callback executed")
        frame = frame_.to_ndarray(format="bgr24")
        #frame=frame_
        print("1 executed")
        frame_placeholder = st.empty()

        # Capture the video and find its parameters
        # imge = camera_input_live()
        #cap = cv2.VideoCapture(img)
        print("2 executed")
        # Encoder to convert the image into vector
        encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', self.max_cosine_distance, self.nn_budget)
        tracker = Tracker(metric, self.max_cosine_distance)
        print("3 executed")
        
        print("white true executed")

            # ret, frame = cap.read()

            # if not ret:
            #     print("if not ret executed")

            #     st.write("Ended")
            #     break

        H, W, _ = frame.shape
        print("H executed")
        print(H)


        self.score.frame_width = W

            # Detect objects in the frame
        results = self.model(frame)
        print("results executed")
        print(results)


        detections = []

        for result in results:
                print("results for loop")
                names = []
                converted_boxes = []
                features = []
                scores = []
                cls_ids = []

                for r in result.boxes.data.tolist():
                    print("results for loop 2")
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
                print("for track in trackes executed")

                # if not track.is_confirmed() or track.time_since_update > 1:
                #     continue
                if track.time_since_update > 1:
                    continue
                print("for track in trackes executed 1")
                bbox = track.to_tlbr()
                print("for track in trackes executed 2")
                class_id = track.class_id

                print("for track in trackes executed 3")
                # Score Write
                cv2.putText(frame, str(self.score.lcount), (300, 150), self.font, 5, (0, 0, 255), 3, self.lineType)
                cv2.putText(frame, str(self.score.rcount), (1000, 150), self.font, 5, (0, 0, 255), 3, self.lineType)

                print("score write executed")
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
                    print("rectangles executed")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("frame executed")
        
        # try:
        #     cv2.imshow("image",frame)
        # except Exception as e:
        #     print("Exception during image display:", str(e))

        print("placeholder executed")
        cv2.waitKey(self.wait)
        print("before release executed")
            
        #cap.release()
        print("release executed")
        #cv2.destroyAllWindows()
        print("before return executed")
        return av.VideoFrame.from_ndarray(frame, format="rgb24")



    def detect(self,video_path: os.path=None):
        pass
        #Capture the video and find its parameters
        # imge = camera_input_live()
        # cap = cv2.VideoCapture(imge)
        
        # while not stop_button_pressed:
        #     ret, frame = cap.read()
        #     H, W, _ = frame.shape
        #     self.score.frame_width=W
            
        #     #Encoder to convert the image into vector
        #     encoder=gdet.create_box_encoder(self.model_filename,batch_size=1)
        #     metric=nn_matching.NearestNeighborDistanceMetric('cosine',self.max_cosine_distance,self.nn_budget)
        #     tracker=Tracker(metric,self.max_cosine_distance)
            
        #     #Record
        #     #out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        #     while ret:
        #         #Time of frame
        #         t1 = time.time()
        #         results=self.model(frame)

        #         for result in results:
        #             names=[]
        #             converted_boxes=[]
        #             features=[]
        #             scores=[]
        #             cls_ids=[]
        #             for r in result.boxes.data.tolist():
                        
        #                 x1,y1,x2,y2,cls_score,class_id=r
        #                 names.append(self.class_names[int(class_id)])
        #                 converted_boxes.append(utils.convert_boxes(x1,y1,x2,y2))
        #                 scores.append(cls_score)
        #                 cls_ids.append(class_id)

        #             features=encoder(frame,converted_boxes)
        #             names=np.array(names)

        #             detections = [Detection(bbox, score, feature,cls_id) for bbox, score, feature,cls_id in
        #             zip(converted_boxes, scores, features,cls_ids)]

                    
        #             boxs = np.array([d.tlwh for d in detections])# x1,y1 ,l,h for detected objects
        #             scores = np.array([d.confidence for d in detections]) # confidence scores for detected objects
        #             cls_ids = np.array([d.class_id for d in detections])
        #             indices = preprocessing.non_max_suppression(boxs, self.nms_max_overlap, scores) 
                    
        #             #Only keep the indices filtered by non_max_suppression
        #             detections = [detections[i] for i in indices]   
                    
        #             tracker.predict()
        #             tracker.update(detections)

        #         for track in tracker.tracks:
        #             if not track.is_confirmed() or track.time_since_update >1:
        #                 continue
        #             bbox = track.to_tlbr()
        #             class_id = track.class_id 
                
        #             #Score Write
        #             cv2.putText(frame,str(self.score.lcount), (300,150), self.font, 5,(0, 0, 255),3,self.lineType)
        #             cv2.putText(frame,str(self.score.rcount), (1000,150), self.font, 5,(0, 0, 255),3,self.lineType)
                        
        #             if class_id==0:  #Ball
        #                 self.ball.x1=int(bbox[0])
        #                 self.ball.y1=int(bbox[1])
        #                 self.ball.x2=int(bbox[2])
        #                 self.ball.y2=int(bbox[3])
        #                 self.ball.update()
        #                 self.score.detect_score(self.ball.x1,self.ball.y1,self.ball.x2,self.ball.y2,
        #                 self.hoop.x1,self.hoop.y1,self.hoop.x2,self.hoop.y2)          
        #                 cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),self.BALL_COLOR, 4)
        #             elif class_id==1: #Hoop
        #                 self.hoop.x1=int(bbox[0])
        #                 self.hoop.y1=int(bbox[1])
        #                 self.hoop.x2=int(bbox[2])
        #                 self.hoop.y2=int(bbox[3]) 
        #                 cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),utils.hoop_color(self.ball,self.hoop), 4)
        #                 self.score.set_side(self.hoop.x1,self.hoop.x2)
        #             else:             #Person
        #                 cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),utils.average_color(frame,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])), 4)
        #                 cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
        #                                         (255, 255, 255), 2)

                
        #         frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #         frame_placeholder.image(frame, channels="RGB")
        #         cv2.waitKey(self.wait)
        #         # if len(self.ball.queue)>=3:
        #         #     pts=np.reshape(self.ball.queue,(len(self.ball.queue), 2))
        #         #     print('pppppppppppppppppppppp')
        #         #     print(pts)
        #         #     pts=np.array(pts)
        #         #     cv2.polylines(frame, np.int32([pts]), False, self.BALL_COLOR,4)

        #     if not ret:
        #         st.write("Ended")
        #         break

        #     # cv2.imshow('frame',frame)
        #     # cv2.waitKey(self.wait)
        #     #ret, frame = cap.read()
        # cap.release() 
        # # cv2.destroyAllWindows()
