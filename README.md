# Adaptive Slot Attention
This is an **UNOFFICIAL** PyTorch implementation of *Adaptive Slot Attention: Object Discovery with Dynamic Slot Number* [1]. It builds on the unofficial DINOSAUR [2] implementation from https://github.com/gorkaydemir/DINOSAUR.

Currently, this implementation was only tested on the COCO dataset. 

This implementation compares to the official AdaSlot implementation (https://github.com/amazon-science/AdaSlot) as follows:
- I checked the training logic (model architecture, data, optimization, etc.) in detail and it should match exactly.
- This implementation is less generic than the official implementation.
- A comparison of the training logs of this implementation and the official AdaSlot implementation can be found here: TODO. Training speed and loss curves are very similar.
- This implementation uses the validation metrics (mIoU, mBO, FG-ARI) implementations from https://github.com/gorkaydemir/DINOSAUR. I did not compare them to the implementations in the official AdaSlot repository. 

## Setup
This implementation was tested on an H100 GPU with python 3.10.17, CUDA 11.8, torch `2.1.2+cu118`, numpy 1.26.4. Other GPUs and version probably work as well. 

## Training DINOSAUR
Install the COCO dataset to `/path/to/coco`.

AdaSlot is trained in two stages. First, a fixed-number-of-slots model (very similar to original DINOSAUR) is trained for 200k steps (as described in the AdaSlot paper in Appendix A). Second, the adaptive-number-of-slots model is warmstarted from the stage-1-model and trained for 500k steps. 

Stage 1:
```bash
python train.py \
--model_save_path /path/to/stage1/output/dir \
--log_wandb \  # optional
--exp_id ADASLOT_COCO_STAGE_1 \  # used as experiment ID in wandb
--comment "Training AdaSlot Stage 1 on COCO" \  # used as comment in wandb
--dataset coco \
--root /path/to/coco \
--encoder "dino-vitb-16" \
--num_iters 200000 \
--batch_size 64 \
--resize_to 224 224 \
--train_workers 16 \
--val_workers 2
```

Stage 2:
```bash
python train.py \
--model_save_path /path/to/stage2/output/dir \
--log_wandb \  # optional
--exp_id ADASLOT_COCO_STAGE_2 \  # used as experiment ID in wandb
--comment "Training AdaSlot Stage 2 on COCO" \  # used as comment in wandb
--dataset coco \
--root /path/to/coco \
--encoder "dino-vitb-16" \
--num_slots 33 \
--use_adaptive_slot_attention \
--use_sparsity_loss \
--num_iters 500000 \
--batch_size 64 \
--resize_to 224 224 \
--train_workers 16 \
--val_workers 2 \
--warmstart_checkpoint /path/to/stage1/output/dir/checkpoints/dinosaur_training-iter-000200000.pt
```

Note: Using enough workers (and having enough CPUs and a fast enough disk) has quite a big impact on training speed. As this is system-dependent, it is recommended to play with the number of workers to find the number for optimal training speed.

## Acknowledgments and Changelog

This work is built on the unofficial DINOSAUR implementation from https://github.com/gorkaydemir/DINOSAUR.

It was roughly changed as follows:
- remove multi-gpu capabilities
- adapt checkpointing and logging
- remove ISA (invariant slot attention)
- small fixes to match the original DINOSAUR/AdaSlot implementation: initialize biases of linear layers to 0, remove dropout from slot attention MLP
- adapt the learning rate schedule to the one from AdaSlot
- enable TF32 for faster training
- set number of workers as command line argument (in my case, increasing the number of training data workers made training quite a lot faster)
- add adaptive slot attention

Further speedups might be possible, for example by mixed precision training or using `torch.compile`.

## Appendix: comparison to the official DINOSAUR implementation

The stage 1 of AdaSlot is very similar to DINOSAUR. The only differences seem to be: shorter training (200k iterations instead of 500k) and a slightly different learning rate schedule.

The implementation in this repository compares to the official DINOSAUR implementation (https://github.com/amazon-science/object-centric-learning-framework) as follows:
- I roughly compared the training logic and it seems to match.
- A comparison of the training logs of this implementation and the official DINOSAUR implementation can be found here: TODO. Training speed and loss curves are very similar.
- This implementation is essentially the same as in https://github.com/gorkaydemir/DINOSAUR, except for fixing some small differences to the official DINOSAUR implementation (biases of linear layers, etc. ; see changelog).

Command for DINOSAUR Training:
```bash
python train.py \
--model_save_path /path/to/dinosaur/output/dir \
--log_wandb \  # optional
--exp_id DINOSAUR_COCO \  # used as experiment ID in wandb
--comment "Training DINOSAUR on COCO" \  # used as comment in wandb
--dataset coco \
--root /path/to/coco \
--encoder "dino-vitb-16" \
--num_iters 500000 \
--batch_size 64 \
--resize_to 224 224 \
--train_workers 16 \
--val_workers 2 \
--use_dinosaur_lr_schedule
```

## References

[1] Ke Fan et al. "Adaptive Slot Attention: Object Discovery with Dynamic Slot Number", CVPR 2024

[2] Maximilian Seitzer et al. "Bridging the gap to real-world object-centric learning.", ICLR 2023
