import numpy as np
class Score():
    def __init__(self,lcount=0, rcount=0,frame_width=640):
        self.lcount=lcount
        self.rcount=rcount
        self.reset=0
        self.max_hoop_height=0
        self.frame_width=frame_width
        self.side='left' #'right'

    def set_max_hoop_height(self,hoop_y1,hoop_y2):
        if (hoop_y2-hoop_y1)>self.max_hoop_height:
            self.max_hoop_height=(hoop_y2-hoop_y1)
    
    def set_side(self,hoop_x1,hoop_x2):
        hoop_x=int((hoop_x1+hoop_x2)/2)
        if hoop_x > self.frame_width*(0.4):
            self.side='right'
        else:
            self.side='left'
    
    def detect_score(self,ball_x1,ball_y1,ball_x2,ball_y2,
                    hoop_x1,hoop_y1,hoop_x2,hoop_y2):

        self.set_max_hoop_height(hoop_y1,hoop_y2)
        
        if ball_x1>hoop_x1 and ball_x2<hoop_x2:
            if ball_y1>hoop_y1 and ball_y2<hoop_y2:
                if (self.reset != 1):
                    if self.side=='right':
                        self.rcount+=1 
                    else:
                        self.lcount+=1
                    self.reset=1

        if np.minimum(int(ball_y1),int(ball_y2))>(np.maximum(int(hoop_y1),int(hoop_y2))+2*self.max_hoop_height):
                self.reset=0
