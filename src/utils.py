import numpy as np
class utils:       
    def average_color(frame,x1,y1,x2,y2):
        img=frame[x1:x2,y1:y2]
        try:
            average_color_row = np.average(img, axis=0)
        except:
            average_color_row = (255,255,255)
        average_color = np.average(average_color_row, axis=0)
        return average_color
    
    def hoop_color(ball,hoop):
        ball_x1=ball.x1
        ball_y1=ball.y1
        ball_x2=ball.x2
        ball_y2=ball.y2
        hoop_x1=hoop.x1
        hoop_y1=hoop.y1
        hoop_x2=hoop.x2
        hoop_y2=hoop.y2
        color=(255,255,255)
        if ball_x1>hoop_x1 and ball_x2<hoop_x2 and ball_y1>hoop_y1 and ball_y2<hoop_y2:
            color = (0,255,0)
        else:
            color = (255,0,0)
        return color

    def convert_boxes(x1,y1,x2,y2):
        return [x1,y1,np.absolute(x2-x1),np.absolute(y2-y1)]