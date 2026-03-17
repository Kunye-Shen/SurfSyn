import os
import torch
import numpy as np
from PIL import Image

def save_output(img_dir, pred, save_dir):
    img_name = img_dir.split('/')[-1][:-4]
    img = np.array(Image.open(img_dir).convert('RGB'))
    h, w, _ = img.shape

    pred = (pred*255).cpu().detach().numpy()
    pred = Image.fromarray(pred.astype(np.uint8))
    pred = pred.resize((w, h), resample=Image.BILINEAR)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    pred.save(f'{save_dir}{img_name}.png')
    
    return

def normPRED(x):
    MAX = torch.max(x)
    MIN = torch.min(x)

    out = (x - MIN) / (MAX - MIN)

    return out