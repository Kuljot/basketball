import os
from src.detector import Detector

VIDEOS_DIR = os.path.join('..','dataset')
video_path = os.path.join(VIDEOS_DIR, 'thunder.mp4')
video_path_out =os.path.join(VIDEOS_DIR,'{}_out.mp4'.format(video_path))
model_path = os.path.join(VIDEOS_DIR,'last.pt')

det=Detector(model_path,0.2)
det.detect(video_path)

