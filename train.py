import torch
import glob
from datetime import datetime
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
import time
import os
import argparse
from tqdm import tqdm

## models
## copy-pasted from paper appendix A

# D x, D y: latent dimension of unimodal encoders
# D s: latent dimension of shared space
# depth x, depth y: number of blocks for each adapter
# expansion factor: expansion factor hyperparameter
# dropout: dropout hyperparameter

D_x = 768
D_y = 1536
D_s = 512

# block depth
depth_x, depth_y = 2, 2

class Block(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.2):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, int(expansion_factor * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(expansion_factor * dim), dim),
            
        )
        self.ln = nn.LayerNorm(dim)

        # self.apply(self.init_weights)
    
    def init_weights(self, m):
        pass
        
    def forward(self, x):
        return x + self.fn(self.ln(x))

class FusionAdapter(nn.Module):
    def __init__(self, dim, dim_out, depth, n_paths=1, expansion_factor=4, dropout=0.2):
        super().__init__()
        if depth > 0:
            self.enc = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                # nn.Linear(dim, dim),
                # nn.GELU(),
                nn.Dropout(dropout),
            )
            
            self.fns = nn.ModuleList([
                nn.Sequential(
                    *[Block(dim, expansion_factor=expansion_factor, dropout=dropout) for _ in range(depth)],
                    nn.GELU(),
                    nn.Linear(dim, dim_out),
                ) 
                for _ in range(n_paths)
            ])
            
            self.tail = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(dim_out),
                nn.Linear(dim_out, dim_out)
            )
            # self.apply(self.init_weights)
            
        
    def forward(self, x):
        if depth < 1:
            return x
        enc = self.enc(x)
        sum = torch.stack([fn(enc) for fn in self.fns], dim=0).sum(dim=0)
        return self.tail(sum)

def train(hhX, hhY, STEPS=1e5, B=1000, savedir=None):
    STEPS = int(STEPS)
    B = int(B)
    
    all_img_emb = torch.zeros((82600, 768), dtype=torch.float32, device="cuda")
    all_txt_emb = torch.zeros((82600, 1536), dtype=torch.float32, device="cuda")
    all_ids = []
    
    emb_paths = glob.glob("coco_embeddings_3500/*.pt")

    cum_ind = 0
    for path_idx, path in enumerate(emb_paths):
        batch, text_batch, ids = torch.load(path)
        all_img_emb[cum_ind:cum_ind+batch.shape[0], :] = torch.tensor(batch).cuda()
        all_txt_emb[cum_ind:cum_ind+batch.shape[0], :] = torch.tensor(text_batch).cuda()
        all_ids.extend(ids)
        cum_ind += batch.shape[0]
    
    print("UP TO", cum_ind)

    t = nn.Parameter(0.15 * torch.ones([], requires_grad=True).to("cuda"))

    params = list(hhX.parameters()) + list(hhY.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    
    print_every = STEPS // 100
    save_every = STEPS // 10
    last_time = time.time()
    losses = []

    # symmetric alignment loss
    labels = torch.arange(B, device="cuda", dtype=torch.long)
    
    for i in range(STEPS):
        optimizer.zero_grad()
        temp_optim.zero_grad()
        
        ind = torch.randint(0, 82600, (2*B,)).cuda()
        # text_ind = torch.randint(0, 5, (2*B,))
        
        z_x = all_img_emb[ind, :]
        z_y = all_txt_emb[ind, :]
        
        # print(z_x[0, 0], z_y[0, 0])
        
        z_x1, z_x2 = torch.chunk(z_x, 2) # B x D x
        z_y1, z_y2 = torch.chunk(z_y, 2) # B x D y
        
        lam = random.random()
        
        z_x = lam * z_x1 + (1 - lam) * z_x2
        z_y = lam * z_y1 + (1 - lam) * z_y2
        
        # joint space and normalize
        s_x = F.normalize(hhX(z_x), dim=-1) # B x D s
        s_y = F.normalize(hhY(z_y), dim=-1) # B x D s
        
        # pairwise cosine similarity w/ temperature
        logits_xy = (s_x @ s_y.T) / t # B x B
        logits_yx = (s_y @ s_x.T) / t # B x B
        
        loss_xy = F.cross_entropy(logits_xy, labels)
        loss_yx = F.cross_entropy(logits_yx, labels)

        loss = (loss_xy + loss_yx) / 2.0
        
        # optimize
        loss.backward()
        
        optimizer.step()

        losses.append(loss.item())

        if i % save_every == 0 and savedir:
            os.makedirs(savedir, exist_ok=True)
            torch.save({"model": (hhX, hhY, t), "loss_history": losses, "opt": optimizer}, os.path.join(savedir, f"step_{i}.pt"))
    
        if i % print_every == 0:
            print(i, loss.item(), time.time() - last_time)
            total_norm = 0
            max_norm = 0
            for p in params:
                if (p is None) or (p.grad is None):
                    continue
                param_norm = p.grad.detach().data.norm(2)
                max_norm = max(max_norm, param_norm)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"GRAD NORM: {total_norm:.08f}, MAX NORM: {max_norm:.08f}, LR: {optimizer.param_groups[0]['lr']:.08f}, T: {t.item():.04f}", flush=True)
            last_time = time.time()
        
        # ind = (ind + 2*B) % (2*B)
    return optimizer, losses, t


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1e5)
    parser.add_argument("--batch", type=int, default=1e3)
    parser.add_argument("--output")
    args = parser.parse_args()
    
    adapterx = FusionAdapter(D_x, D_s, depth_x, n_paths=1, dropout=0.4)
    adaptery = FusionAdapter(D_y, D_s, depth_y, n_paths=1, dropout=0.4)
    
    adapterx.cuda().train()
    adaptery.cuda().train()

    path = args.output if args.output else f"model_{datetime.utcnow().isoformat()}/"

    opt, losses, t = train(adapterx, adaptery, STEPS=args.steps, B=args.batch, savedir=path)

    torch.save({"model": (adapterx, adaptery, t), "loss_history": losses, "opt": opt}, os.path.join(path, "model.pt"))

if __name__ == "__main__":
    main()
