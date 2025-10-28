import sys
import argparse

import torch

def set_remaining_args(args):

    args.patch_size = int(args.encoder.split("-")[-1])
    args.token_num = (args.resize_to[0] * args.resize_to[1]) // (args.patch_size ** 2)

def print_args(args):
    print("====== Arguments ======")
    print(f"training name: {args.model_save_path.split('/')[-1]}\n")
    print(f"seed: {args.seed}\n")

    print(f"dataset: {args.dataset}")
    print(f"resize_to: {args.resize_to}\n")

    print(f"encoder: {args.encoder}\n")

    print(f"num_slots: {args.num_slots}")
    print(f"slot_att_iter: {args.slot_att_iter}")
    print(f"slot_dim: {args.slot_dim}")

    print(f"learning_rate: {args.learning_rate}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_iters: {args.num_iters}")
    print("====== ======= ======\n")

def get_args():
    parser = argparse.ArgumentParser("Dinosaur++")

    # === Data Related Parameters ===
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="coco", choices=["coco"]) 
    parser.add_argument('--resize_to',  nargs='+', type=int, default=[224, 224])

    # === ViT Related Parameters ===
    parser.add_argument('--encoder', type=str, default="dino-vitb-16", choices=["dinov2-vitb-14", "dino-vitb-16", "dino-vitb-8", "sup-vitb-16"])

    # === Slot Attention Related Parameters ===
    parser.add_argument('--num_slots', type=int, default=7)
    parser.add_argument('--slot_att_iter', type=int, default=3)
    parser.add_argument('--slot_dim', type=int, default=256)
    parser.add_argument('--use_adaptive_slot_attention', action="store_true")
    parser.add_argument('--use_sparsity_loss', action="store_true")
    parser.add_argument('--sparsity_loss_weight', type=float, default=0.5)

    # === Training Related Parameters ===
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_iters', type=int, default=500000)
    parser.add_argument('--train_workers', type=int, default=5)
    parser.add_argument('--val_workers', type=int, default=5)
    parser.add_argument('--use_dinosaur_lr_schedule', action="store_true")

    # === Misc ===
    parser.add_argument('--warmstart_checkpoint', type=str, default=None)
    parser.add_argument('--val_qual_interval', type=int, default=500)

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--exp_id', type=str, default=None)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--log_wandb', action="store_true")

    args = parser.parse_args()

    set_remaining_args(args)

    return args