class Score():
    def __init__(self,score=0):
        self.count=0
        self.reset=0
        self.max_hoop_height=0

    def set_max_hoop_height(self,hoop_y1,hoop_y2):
        if (hoop_y2-hoop_y1)>self.max_hoop_height:
            self.max_hoop_height=(hoop_y2-hoop_y1)
    
    def detect_score(self,ball_x1,ball_y1,ball_x2,ball_y2,
                    hoop_x1,hoop_y1,hoop_x2,hoop_y2):

        self.set_max_hoop_height(hoop_y1,hoop_y2)
        
        if ball_x1>hoop_x1 and ball_x2<hoop_x2:
            if ball_y1>hoop_y1 and ball_y2<hoop_y2:
                if (self.reset != 1):
                    self.count+=1 
                    self.reset=1

        if ball_y1>(hoop_y2+self.max_hoop_height):
                self.reset=0
