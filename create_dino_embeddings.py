EMBEDDINGS_BATCH_SIZE = 350

import torch
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import skimage.io as io
from skimage.transform import resize as skresize
import os

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval().cuda()
dinov2.zero_grad(set_to_none=True)

flag = False

with torch.no_grad():
    batch = torch.zeros((EMBEDDINGS_BATCH_SIZE, 350, 350, 3), dtype=torch.float32, device="cuda")

    paths = glob.glob("train2014/*.jpg")
    print(paths[:10])

    ids = []

    i = 0
    batchnum = 0
    for path in tqdm(paths):
        img_rgb = io.imread(path)
        img_rgb = skresize(img_rgb, (350, 350, 3))
                
        if len(img_rgb.shape) < 3 or img_rgb.shape[2] != 3:
            continue
        
        img_rgb_norm = np.divide(img_rgb - np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])) # mean and sd for imagenet
        img_bgr = img_rgb_norm[:, :, ::-1]
        
        pad_x = int(np.abs(img_bgr.shape[0] - batch.shape[1]) // 2)
        pad_y = int(np.abs(img_bgr.shape[1] - batch.shape[2]) // 2)

        batch[i, pad_x:pad_x+img_bgr.shape[0], pad_y:pad_y+img_bgr.shape[1], :] = torch.from_numpy(img_bgr.copy()).to("cuda")
        
        if not flag:
            print(path)
            io.imsave("sample.png", np.multiply(255.0, img_rgb).astype(np.uint8))
            flag = True

        ids.append(path.split(".")[0].split("_")[-1])
        i += 1
        if i >= EMBEDDINGS_BATCH_SIZE:
            emb = dinov2(batch.transpose(1, 3)).cpu().detach().numpy()
            assert emb.shape == (EMBEDDINGS_BATCH_SIZE, 768)
            torch.save((emb, ids), f"dino_embeddings/emb_{batchnum:05d}.pt")
            ids = []
            batch[:, :, :, :] = 0.0
            batchnum += 1
            
            i = 0

    print(batch.shape)
    print(batchnum)
    print(i)