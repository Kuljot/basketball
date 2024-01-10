import os
from src.visualizer import Visualizer

model_path = os.path.join('.','models','last_50_re.pt')

#Create the visualizer object
vis=Visualizer(model_path,0.2)
vis.record()





