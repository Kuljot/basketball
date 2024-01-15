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
    # def center_inside(self,ball_x1,ball_y1,ball_x2,ball_y2,
    #                 hoop_x1,hoop_y1,hoop_x2,hoop_y2):

    def correct_order(self,x1,x2):

        min_=np.minimum(x1,x2)
        max_=np.maximum(x1,x2)
        return min_,max_
        

    
    def detect_score(self,ball_x1,ball_y1,ball_x2,ball_y2,
                    hoop_x1,hoop_y1,hoop_x2,hoop_y2):
        
        ball_x1,ball_x2=self.correct_order(ball_x1,ball_x2)
        ball_y1,ball_y2=self.correct_order(ball_y1,ball_y2)
        hoop_x1,hoop_x2=self.correct_order(hoop_x1,hoop_x2)
        hoop_y1,hoop_y2=self.correct_order(hoop_y1,hoop_y2)

        self.set_max_hoop_height(hoop_y1,hoop_y2)
        
        # if ball_x1>hoop_x1 and ball_x2<hoop_x2:
        #     if ball_y1>hoop_y1 and ball_y2<hoop_y2:
        #         if (self.reset != 1):
        #             if self.side=='right':
        #                 self.rcount+=1 
        #             else:
        #                 self.lcount+=1
        #             self.reset=1

        # if np.minimum(int(ball_y1),int(ball_y2))>(np.maximum(int(hoop_y1),int(hoop_y2))+2*self.max_hoop_height):
        #         self.reset=0

        center_x=int((ball_x1+ball_x2)/2.0)
        center_y=int((ball_y1+ball_y2)/2.0)

        hoop_mid_y=int((hoop_y1+hoop_y2)/2.0)
        
        if center_x>hoop_x1 and center_x<hoop_x2:
            if center_y<hoop_mid_y and center_y>(hoop_mid_y-1.5*self.max_hoop_height):
                self.reset=2
            elif center_y>hoop_mid_y and center_y<(hoop_mid_y+1.5*self.max_hoop_height) and self.reset==2:
                if self.side=='right':
                    self.rcount+=1 
                    self.reset=0
                else:
                    self.lcount+=1
                    self.reset=0
            else:
                self.reset=3
        else: 
            self.reset=4