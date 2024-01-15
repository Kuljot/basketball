import os
from src.visualizer import Visualizer

model_path = os.path.join('.','models','last.pt')

#Create the visualizer object
vis=Visualizer(model_path,0.2)
vis.record()





