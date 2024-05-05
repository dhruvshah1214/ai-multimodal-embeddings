import glob
import torch
import numpy as np
from tqdm import tqdm

emb_paths = glob.glob("joint_embeddings/*.pt")
emb_groups = [emb_paths[i:i + 10] for i in range(0, len(emb_paths), 10)] 


for group_idx, group in enumerate(tqdm(emb_groups)):
    batch_img, batch_txt = np.zeros((350*len(group), 768)), np.zeros((350*len(group), 1536))
    batch_ids = []
    for path_idx, path in enumerate(group):
        minibatch_img, minibatch_txt, minibatch_ids = torch.load(path)
        minibatch_ids = [int(id) for id in minibatch_ids]

        batch_ids.extend(minibatch_ids)
        
        start_idx = path_idx * 350
        batch_img[start_idx:start_idx+350, :] = minibatch_img
        batch_txt[start_idx:start_idx+350, :] = minibatch_txt
    torch.save((batch_img, batch_txt, batch_ids), f"coco_embeddings_3500/batch_{group_idx:05d}.pt")