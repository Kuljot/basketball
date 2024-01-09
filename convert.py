import os
from src.visualizer import Visualizer
VIDEOS_DIR = os.path.join('..','dataset')
model_path = os.path.join(VIDEOS_DIR,'last_50_re.pt')

vis=Visualizer(model_path,0.2)
vis.record()