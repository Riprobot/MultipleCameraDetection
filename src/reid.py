from boxmot import BotSort
# from boxmot.utils.ops import yolox_preprocess
import cv2
import gdown
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from ultralytics.utils import ops

tracker = BotSort(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=torch.device('cuda:0'), half=False)

reid_model = tracker.model

def get_features(xyxys, img):
    return reid_model.get_features(xyxys, img)