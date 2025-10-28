import os
import os.path as osp
import time
import math
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tqdm import tqdm 

from models.model import DINOSAURpp, Visual_Encoder

from read_args import get_args, print_args
import dinosaur_utils
from checkpoint_utils import TrainStateSaver
import dinosaur_writer as writer
from vis import vcat, hcat, add_label, add_box


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def val(args, vis_encoder, model, val_dataloader, evaluator_inst, evaluator_sem, iteration, qual_only=False):
    vis_encoder.eval()
    model.eval()

    loader = tqdm(val_dataloader)

    slot_num = args.num_slots
    ps = args.patch_size

    i_for_qual = random.randint(0, 20)

    for i, (model_input, instance_gt, semantic_gt) in enumerate(loader):

        model_input = model_input.cuda(non_blocking=True)                # (B, 3, H, W)
        instance_gt = instance_gt.cuda(non_blocking=True)                      # (B, *, H_t, W_t)
        semantic_gt = semantic_gt.cuda(non_blocking=True)                      # (B, *, H_t, W_t)

        H_t, W_t = instance_gt.shape[-2:]

        H, W = args.resize_to

        features = vis_encoder(model_input)                      # (B, token, 768)
        reconstruction, slots, masks, per_slot_reconstructions, hard_keep_decision, soft_keep_decision = model(features)  # (B, token, 768), (B, S, D_slot), (B, S, token), (B, S, token, 768), (B, S) or None, (B, S) or None
        
        features = features.view(-1, H // ps, W // ps, features.shape[-1])  # (B, H // ps, W // ps, 768)
        reconstruction = reconstruction.view(-1, H // ps, W // ps, reconstruction.shape[-1])  # (B, H // ps, W // ps, 768)
        masks = masks.view(-1, slot_num, H // ps, W // ps)                      # (B,  S, H // ps, W // ps)
        masks = F.interpolate(masks, size=(H_t, W_t), mode="bilinear")    # (B, S, H_t, W_t)
        predictions = torch.argmax(masks, dim=1)                          # (B, H_t, W_t)
        per_slot_reconstructions = per_slot_reconstructions.view(-1, slot_num, H // ps, W // ps, per_slot_reconstructions.shape[-1])  # (B, S, H // ps, W // ps, 768)

        if i == i_for_qual and qual_only:
            features = features[0]
            pca = dinosaur_utils.TorchPCA(dinosaur_utils.get_pca_from_feature_space(features)).to(features.device, dtype=features.dtype)

            features_pca = pca.transform(features, normalize=True)
            features_pca = features_pca.permute(2, 0, 1)
            features_pca = F.interpolate(features_pca.unsqueeze(0), size=(H, W), mode="nearest")

            reconstruction = reconstruction[0]
            reconstruction_pca = pca.transform(reconstruction, normalize=True)
            reconstruction_pca = reconstruction_pca.permute(2, 0, 1)
            reconstruction_pca = F.interpolate(reconstruction_pca.unsqueeze(0), size=(H, W), mode="nearest")

            per_slot_reconstructions = per_slot_reconstructions[0]
            per_slot_reconstructions_pca = pca.transform(per_slot_reconstructions, normalize=True)
            per_slot_reconstructions_pca = per_slot_reconstructions_pca.permute(0, 3, 1, 2)
            per_slot_reconstructions_pca = F.interpolate(per_slot_reconstructions_pca, size=(H, W), mode="nearest")  # S, 3, H, W

            hard_keep_decision = hard_keep_decision[0] if hard_keep_decision is not None else None
            slot_grids = []
            kept_slot_grids = [] if hard_keep_decision is not None else None
            for slot_idx in range(slot_num):
                per_slot_reconstruction_pca = per_slot_reconstructions_pca[slot_idx]
                slot_mask = masks[0, slot_idx]
                slot_grid = add_label(add_box(vcat(
                    add_label(slot_mask, "Alpha Mask", clipping=True, lower_clipping_thresh=0, upper_clipping_thresh=1),
                    add_label(per_slot_reconstruction_pca, "Reconstruction", clipping=True, lower_clipping_thresh=0, upper_clipping_thresh=1),
                    flatten_grids=False,
                ), border=4), f"Slot {slot_idx}", nest_grid=True)
                slot_grids.append(slot_grid)

                if hard_keep_decision is not None and hard_keep_decision[slot_idx] > 0:
                    kept_slot_grids.append(slot_grid)

            input_grid = add_label(add_box(vcat(
                add_label(model_input[0], "Image"),
                add_label(features_pca[0], "Features", clipping=True, lower_clipping_thresh=0, upper_clipping_thresh=1),
                flatten_grids=False,
            ), border=4), "Input", nest_grid=True)

            pred_grid = add_label(add_box(vcat(
                add_label(predictions[0], "Segmentation", clipping=True, lower_clipping_thresh=0, upper_clipping_thresh=slot_num-1, image_range_text_off=True),
                add_label(reconstruction_pca[0], "Reconstruction", clipping=True, lower_clipping_thresh=0, upper_clipping_thresh=1),
                flatten_grids=False,
            ), border=4), "Prediction", nest_grid=True)

            if kept_slot_grids is None or len(kept_slot_grids) == 0:
                grid = hcat(
                    input_grid,
                    pred_grid,
                    *slot_grids,
                    flatten_grids=True,
                )

                writer.put_grid(name="qual_overview", grid=grid, step=iteration)

            else:
                grid = hcat(
                    input_grid,
                    pred_grid,
                    *kept_slot_grids,
                    flatten_grids=True,
                )

                writer.put_grid(name="qual_overview", grid=grid, step=iteration)

                grid = hcat(
                    input_grid,
                    pred_grid,
                    *slot_grids,
                    flatten_grids=True,
                )

                writer.put_grid(name="qual_overview_all_slots", grid=grid, step=iteration)

            evaluator_inst.reset()
            evaluator_sem.reset()
            vis_encoder.eval()
            model.train()
            return

        # === Instance Segmentation Evaluation ===
        miou_i, mbo_i, fgari_i = evaluator_inst.update(predictions, instance_gt)
        miou_s, mbo_s, fgari_s = evaluator_sem.update(predictions, semantic_gt)
        loss_desc = f"mBO_i: {mbo_i:.2f} mBO_s: {mbo_s:.2f} FG-ARI_i: {fgari_i:.2f}"
        # === Logger ===
        loader.set_description(loss_desc)
        # === === ===

    # === Evaluation Results ====
    miou_i, mbo_i, fgari_i = evaluator_inst.get_results(reset=True)
    miou_s, mbo_s, fgari_s = evaluator_sem.get_results(reset=True)

    # === Logger ===
    print("\n=== Results ===")
    print(f"Iteration: {iteration}\n")
    print("Instance Segmentation")
    print(f"\tmIoU: {miou_i:.5f}")
    print(f"\tmBO: {mbo_i:.5f}")
    print(f"\tFG-ARI: {fgari_i:.5f}\n")

    print("Semantic Segmentation")
    print(f"\tmIoU: {miou_s:.5f}")
    print(f"\tmBO: {mbo_s:.5f}")
    print(f"\tFG-ARI: {fgari_s:.5f}")

    writer.put_scalar("object_discovery/mIoU_i", scalar=miou_i, step=iteration)
    writer.put_scalar("object_discovery/mBO_i", scalar=mbo_i, step=iteration)
    writer.put_scalar("object_discovery/FG-ARI_i", scalar=fgari_i, step=iteration)
    writer.put_scalar("val/instance_mask_ari", scalar=fgari_i / 100, step=iteration)
    writer.put_scalar("val/instance_abo", scalar=mbo_i / 100, step=iteration)

    writer.put_scalar("object_discovery/mIoU_s", scalar=miou_s, step=iteration)
    writer.put_scalar("object_discovery/mBO_s", scalar=mbo_s, step=iteration)
    writer.put_scalar("object_discovery/FG-ARI_s", scalar=fgari_s, step=iteration)

    vis_encoder.eval()
    model.train()
    return


def setup_saver(model, optimizer, scheduler, checkpoints_dir, checkpoints_name):
    max_checkpoints_to_keep = 2
    saver_all = TrainStateSaver(model=model, 
                                optim=optimizer,
                                scheduler=scheduler,
                                base_path=checkpoints_dir,
                                base_name=checkpoints_name,
                                max_to_keep=max_checkpoints_to_keep)
    
    return saver_all


def restore_weights(saver_all):
    all_checkpoints = sorted(saver_all.get_checkpoints(include_iteration=True))

    finished_iterations = 0
    if len(all_checkpoints) > 0:
        
        print("Existing checkpoints:")
        for step, checkpoint in all_checkpoints:
            print(f"\t{step}: {checkpoint}")
            
        newest_checkpoint = all_checkpoints[-1][1]
        print(f"Restoring training state from checkpoint {newest_checkpoint}.")
        saver_all.load(full_path=newest_checkpoint)
        finished_iterations = all_checkpoints[-1][0]

    return finished_iterations


def save_all(saver_all, finished_iterations):
    save_path = saver_all.save(iteration=finished_iterations)
    print(f"Saved training state checkpoint to {save_path}.")


def write_checkpoints(saver_all, finished_iterations, start_iteration):
    if finished_iterations > start_iteration:
        save_all(saver_all, finished_iterations)


def main_worker(args):
    dinosaur_utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    log_loss_interval = 10
    val_qual_interval = args.val_qual_interval
    val_interval = 10000
    save_checkpoint_interval_min = 20
    out_dir = args.model_save_path
    checkpoints_name = "dinosaur_training"

    tensorboard_logs_dir = osp.join(out_dir, "tensorboard_logs")
    wandb_logs_dir = osp.join(out_dir, "wandb_logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tensorboard_logs_dir, exist_ok=True)
    os.makedirs(wandb_logs_dir, exist_ok=True)
    writer.setup_writers(log_tensorboard=not args.log_wandb, 
                         log_wandb=args.log_wandb, 
                         tensorboard_logs_dir=tensorboard_logs_dir, 
                         wandb_logs_dir=wandb_logs_dir,
                         exp_id=args.exp_id,
                         comment=args.comment,)
    
    checkpoints_dir = osp.join(out_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print_args(args)

    # === Dataloaders ====
    train_dataloader, val_dataloader = dinosaur_utils.get_dataloaders(args)

    # === Model ===
    vis_encoder = Visual_Encoder(args).cuda()
    model = DINOSAURpp(args).cuda()

    # === Training Items ===
    optimizer = torch.optim.Adam(dinosaur_utils.get_params_groups(model), lr=args.learning_rate)
    scheduler = dinosaur_utils.get_scheduler(args, optimizer, train_dataloader)

    # === Misc ===
    evaluator_instance = dinosaur_utils.Evaluator()
    evaluator_semantic = dinosaur_utils.Evaluator()

    print(f"Loss, optimizer and schedulers ready.")

    # === Load from checkpoint ===
    max_iterations = args.num_iters
    writer.set_max_iterations(max_iterations)
    saver_all = setup_saver(model, optimizer, scheduler, checkpoints_dir, checkpoints_name)
    finished_iterations = restore_weights(saver_all)
    if finished_iterations == 0 and args.warmstart_checkpoint is not None:
        print(f"Warmstarting model from checkpoint: {args.warmstart_checkpoint}")
        checkpoint = torch.load(args.warmstart_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_iteration = finished_iterations

    print("Starting training!")

    should_continue = lambda: finished_iterations < max_iterations

    if not should_continue():
        print("No more training iterations. Exiting.")
        return
    
    should_log_loss = lambda: finished_iterations % log_loss_interval == 0
    should_val = lambda: finished_iterations % val_interval == 0
    should_val_qual = lambda: finished_iterations % val_qual_interval == 0
    
    last_checkpoint_time = time.time()
    
    while should_continue():
        total_loss = 0.0
        vis_encoder.eval()
        model.train()

        loader = tqdm(train_dataloader)

        for iter_in_epoch, (frames, _, _) in enumerate(loader):
            frames = frames.cuda(non_blocking=True)      # (B, 3, H, W)

            B = frames.shape[0]

            with torch.no_grad():
                features = vis_encoder(frames)                      # (B, token, 768)
            reconstruction, _, masks, per_slot_reconstructions, hard_keep_decision, soft_keep_decision = model(features)  # (B, token, 768)

            mse_loss = F.mse_loss(reconstruction, features.detach())

            if args.use_sparsity_loss:
                sparse_degree = torch.mean(hard_keep_decision)  # hard keep decision shape : [B, S]
                sparse_loss = args.sparsity_loss_weight * sparse_degree
                loss = mse_loss + sparse_loss
            else:
                sparse_loss = None
                loss = mse_loss

            total_loss += loss.item()

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            mean_loss = total_loss / (iter_in_epoch + 1)
            loader.set_description(f"lr: {lr:.6f} | loss: {loss:.5f} | mean_loss: {mean_loss:.5f}")

            if should_log_loss():
                writer.put_scalar("00_overview/00_total_loss", scalar=loss, step=finished_iterations)
                writer.put_scalar("00_overview/03_learning_rate", scalar=lr, step=finished_iterations)
                if sparse_loss is not None:
                    writer.put_scalar("00_overview/01_mse_loss", scalar=mse_loss, step=finished_iterations)
                    writer.put_scalar("00_overview/02_sparse_loss", scalar=sparse_loss, step=finished_iterations)
                if hard_keep_decision is not None:
                    writer.put_scalar("00_overview/04_num_slots", scalar=hard_keep_decision.sum(dim=1).mean() , step=finished_iterations)
                if soft_keep_decision is not None:
                    writer.put_scalar("00_overview/05_avg_slot_keep_prob", scalar=soft_keep_decision.mean() , step=finished_iterations)

                # just for comparison with AdaSlot repo:
                writer.put_scalar("train/mse", scalar=mse_loss, step=finished_iterations)
                if sparse_loss is not None:
                    writer.put_scalar("train/sparse_penalty", scalar=sparse_loss, step=finished_iterations)
                    writer.put_scalar("train/loss_total", scalar=loss, step=finished_iterations)
                writer.put_scalar("lr-Adam", scalar=lr, step=finished_iterations)
                if hard_keep_decision is not None:
                    writer.put_scalar("train/hard_keep_decision", scalar=hard_keep_decision.sum(dim=1).mean() , step=finished_iterations)
                if soft_keep_decision is not None:
                    writer.put_scalar("train/slots_keep_prob", scalar=soft_keep_decision.mean() , step=finished_iterations)
            
            finished_iterations += 1

            if should_val():
                val(args, vis_encoder, model, val_dataloader, evaluator_instance, evaluator_semantic, finished_iterations, qual_only=False)
            elif should_val_qual():
                val(args, vis_encoder, model, val_dataloader, evaluator_instance, evaluator_semantic, finished_iterations, qual_only=True)

            writer.write_out_storage()

            if start_iteration < finished_iterations < max_iterations:
                if time.time() - last_checkpoint_time > 60 * save_checkpoint_interval_min:
                    save_all(saver_all, finished_iterations)
                    last_checkpoint_time = time.time()

            if not should_continue():
                break

    val(args, vis_encoder, model, val_dataloader, evaluator_instance, evaluator_semantic, finished_iterations)
    write_checkpoints(saver_all, finished_iterations, start_iteration)


if __name__ == '__main__':
    args = get_args()
    main_worker(args)
