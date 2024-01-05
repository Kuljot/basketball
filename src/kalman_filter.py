import cv2
import numpy as np

class KalmanFilter:
    def __init__(self, x1=0, y1=0, x2=0, y2=0, dT=1,gain=100):
        
        # 6 state variables (x, y, l, h, vx, vy)
        # 4 measurement variables (x, y, l, h)

        self.kf = cv2.KalmanFilter(6, 4)

        # State Transition Matrix
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0,dT, 0],
            [0, 1, 0, 0, 0,dT],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement Matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)

        # Process Noise Covariance Matrix
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32) * gain

        x,y,l,h=self.convert(x1, y1, x2, y2)
        self.kf.statePost = np.array([x, y, l, h, 0, 0], dtype=np.float32)

    def update(self,x1, y1, x2, y2):
        self.kf.correct(np.array(self.convert(x1, y1, x2, y2), dtype=np.float32))
        prediction = self.kf.predict()
        print(prediction)
        x,y,l,h,vx,vy=prediction
        x1,y1,x2,y2=self.inv_convert(x,y,l,h)
        return x1,y1,x2,y2,vx,vy
    
    def convert(self,x1, y1, x2, y2):
        x=float((x1+x2)/2.0)
        y=float((y1+y2)/2.0)
        l=np.absolute(x1-x2)
        h=np.absolute(y1-y2)
        return [x,y,l,h]
    
    def inv_convert(self,x,y,l,h):
        x1=float(x-l/2.0)
        x2=float(x+l/2.0)
        y1=float(y-h/2.0)
        y2=float(y+h/2.0)
        return x1,y1,x2,y2