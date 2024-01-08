# from src.kalman_filter import KalmanFilter

# class Tracker():
#     def __init__(self,x1=0,y1=0,x2=0,y2=0,t0=0):
#         self.KF=KalmanFilter(dT=10)

#         #Last coordinates
#         self.x1=x1
#         self.y1=y1
#         self.x2=x2
#         self.y2=y2
#         self.vx=0
#         self.vy=0
#         self.t=t0 #ms
    
#     #Predict at time t
#     def predict(self,dt):
#         x1=self.x1+self.vx*dt
#         y1=self.y1+self.vy*dt
#         x2=self.x2+self.vx*dt
#         y2=self.y2+self.vy*dt
#         self.t+=dt
#         return x1,y1,x2,y2

#     def update(self,x1,y1,x2,y2):
#         self.x1,self.y1,self.x2,self.y2, self.vx, self.vy=self.KF.update(x1, y1, x2, y2)