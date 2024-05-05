from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai = OpenAI(api_key=os.getenv("OAI_KEY"))

import torch
import torch.nn.functional as F

from train import Block, FusionAdapter
import glob
import argparse
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

def main(args):    
    # load dino image embeddings from training set
    all_img_emb = torch.zeros((82600, 768), dtype=torch.float32, device="cuda")
    all_txt_emb = torch.zeros((82600, 1536), dtype=torch.float32, device="cuda")
    all_ids = []

    emb_paths = glob.glob("coco_embeddings_3500/*.pt")

    for path_idx, path in enumerate(emb_paths):
        batch, text_batch, ids = torch.load(path, map_location="cuda")
        all_img_emb[path_idx*3500:path_idx*3500+batch.shape[0], :] = torch.tensor(batch).cuda()
        all_txt_emb[path_idx*3500:path_idx*3500+batch.shape[0], :] = torch.tensor(text_batch).cuda()
        all_ids.extend(ids)
        
    # create text embedding for query
    response = openai.embeddings.create(
        input=args.query,
        model="text-embedding-3-small"
    )
    sample_emb = torch.tensor(response.data[0].embedding).unsqueeze(0).cuda()
    
    # load model
    checkpoint = torch.load(args.ckpt_path)
    h_X, h_Y, t = checkpoint["model"]
    h_X.cuda().eval()
    h_Y.cuda().eval()
    t.cuda()
    
    sample_text_in_shared_space = F.normalize(h_Y(sample_emb), dim=-1)
    all_img_emb_in_shared_space = F.normalize(h_X(all_img_emb), dim=-1)

    xy = (sample_text_in_shared_space @ all_img_emb_in_shared_space.T)
    yx = (all_img_emb_in_shared_space @ sample_text_in_shared_space.T)
    similarity = (xy + yx.T)/2
    similarity = similarity.squeeze()
    
    cos, ind = torch.topk(similarity, 3)
    
    coco = COCO("annotations/captions_train2014.json")
    sample_imgs = coco.loadImgs([all_ids[a] for a in ind])
    sample_imgs = [img["coco_url"] for img in sample_imgs]
    
    for idx, (score, url) in enumerate(zip(cos, sample_imgs)):
        I = io.imread(url)
        print(f"COSINE: {score:0.05f}")
        io.imsave(f"ret_image_{idx}.png", I)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    parser.add_argument("--query", default="a cat lays rest on a desk with pens, and index cards, and blue notepad")
    parser.add_argument("--output", default="./image.png")
    args = parser.parse_args()

    main(args)
