from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

client = OpenAI(api_key=os.getenv("OAI_KEY"))

from pycocotools.coco import COCO
coco_captions = COCO("annotations/captions_train2014.json")

import glob
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import skimage.io as io

flag1 = False

dino_emb_files = glob.glob("dino_embeddings/*.pt")
for file in tqdm(dino_emb_files):
    batch, ids = torch.load(file)
    ids = [int(id) for id in ids]
    annIds = coco_captions.getAnnIds(imgIds=ids)
    anns = coco_captions.loadAnns(annIds)

    df = pd.DataFrame(anns).groupby(by="image_id", sort=False)
    captions = []

    if not flag1:
        print(df.head())
    
    for id, group in df:
        group_caps = [group["caption"].to_list()[0]]
        group_ids = [group["image_id"].to_list()[0]]
        # assert len(caps) == 5
        captions.append(list(zip(group_ids, group_caps)))
    
    flatcap = [y for x in captions for (_, y) in x]
    flatid = [y for x in captions for (y, _) in x]

    if not flag1:
        print(flatcap)
        print(flatid)
        print(ids)

    response = client.embeddings.create(
        input=flatcap,
        model="text-embedding-3-small"
    )
    embeddings = [np.array(x.embedding) for x in response.data]

    assert list(range(len(flatcap))) == [x.index for x in response.data] # check order of embeddings in response
    
    embeddings = np.vstack(embeddings)
    text_batch = embeddings
    
    mismatch = np.nonzero(np.array(flatid) != np.array(ids))[0]
    
    # print(mismatch)

    assert mismatch.size == 0
    assert all([a == b for (a, b) in zip(flatid, ids)])

    if not flag1:
        img_coco = coco_captions.loadImgs([flatid[0]])[0]
        img = io.imread(img_coco["coco_url"])
        io.imsave("image.png", img)
        print(flatcap[0])
        flag1 = True

    torch.save((batch, text_batch, ids), f"joint_embeddings/{os.path.basename(file)}")