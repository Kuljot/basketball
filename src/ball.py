class Ball:
    def __init__(self,x1=0,y1=0,x2=0,y2=0):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        self.queue=[]
        self.max_queue_len=3

    def center(self):
        center_x=int((self.x1+self.x2)/2)
        center_y=int((self.y1+self.y2)/2)
        return center_x,center_y

    def update(self):
        x,y=self.center()
        if len(self.queue)>self.max_queue_len:
            self.queue.pop(0)
        self.queue.append([x,y])


    
