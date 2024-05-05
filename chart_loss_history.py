from matplotlib import pyplot as plt
import torch
import numpy as np
import argparse
from train import Block, FusionAdapter

def main(args):
    ckpt = torch.load(args.model_path)
    plt.plot(ckpt["loss_history"])
    plt.savefig(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--output", default="./image.png")
    args = parser.parse_args()

    main(args)
