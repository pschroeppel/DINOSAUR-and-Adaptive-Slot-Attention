import warnings

import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.nn import init
import random
import timm


class Loss_Function(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")


        self.token_num = args.token_num
        self.num_slots = args.num_slots

        self.epsilon = 1e-8

    def forward(self, reconstruction, masks, target):
        # :args reconstruction: (B, token, 768)
        # :args masks: (B, S, token)
        # :args target: (B, token, 768)

        target = target.detach()
        loss = self.mse(reconstruction, target.detach()).mean()

        return loss

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, residual=False, layer_order="none"):
        super().__init__()
        self.residual = residual
        self.layer_order = layer_order
        if residual:
            assert input_dim == output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)

        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)

        if layer_order in ["pre", "post"]:
            self.norm = nn.LayerNorm(input_dim)
        else:
            assert layer_order == "none"

    def forward(self, x):
        input = x

        if self.layer_order == "pre":
            x = self.norm(x)

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        if self.residual:
            x = x + input
        if self.layer_order == "post":
            x = self.norm(x)

        return x
    
class Visual_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.resize_to = args.resize_to
        self.token_num = args.token_num

        self.encoder = args.encoder

        self.model = self.load_model(args)


    def load_model(self, args):
        assert args.resize_to[0] % args.patch_size == 0
        assert args.resize_to[1] % args.patch_size == 0
        
        if args.encoder == "dino-vitb-8":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        elif args.encoder == "dino-vitb-16":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        elif args.encoder == "dinov2-vitb-14":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        elif args.encoder == "sup-vitb-16":
            model = timm.create_model("vit_base_patch16_224", pretrained=True, img_size=(args.resize_to[0], args.resize_to[1]))
        else:
            assert False

        for p in model.parameters():
            p.requires_grad = False

        # wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
        # wget https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
        # wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
        
        return model
    
    @torch.no_grad()
    def forward(self, frames):
        # :arg frames:  (B, 3, H, W)
        #
        # :return x:  (B, token, 768)

        B = frames.shape[0]

        self.model.eval()

        if self.encoder.startswith("dinov2-"):
            x = self.model.prepare_tokens_with_masks(frames)
        elif self.encoder.startswith("sup-"):
            x = self.model.patch_embed(frames)
            x = self.model._pos_embed(x)
        else:
            x = self.model.prepare_tokens(frames)


        for blk in self.model.blocks:
            x = blk(x)
        x = x[:, 1:]

        assert x.shape[0] == B
        assert x.shape[1] == self.token_num
        assert x.shape[2] == 768

        return x



class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # === Token calculations ===
        slot_dim = args.slot_dim
        hidden_dim = 2048

        # === MLP Based Decoder ===
        self.layer1 = nn.Linear(slot_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 768 + 1)
        self.relu = nn.ReLU(inplace=True)

        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)
        nn.init.zeros_(self.layer4.bias)

    def forward(self, slot_maps):
        # :arg slot_maps: (B, S, token, D_slot)

        slot_maps = self.relu(self.layer1(slot_maps))    # (B, S, token, D_hidden)
        slot_maps = self.relu(self.layer2(slot_maps))    # (B, S, token, D_hidden)
        slot_maps = self.relu(self.layer3(slot_maps))    # (B, S, token, D_hidden)

        slot_maps = self.layer4(slot_maps)               # (B, S, token, 768 + 1)

        return slot_maps
    

class SA(nn.Module):
    def __init__(self, args, input_dim):
        
        super().__init__()
        self.num_slots = args.num_slots
        self.scale = args.slot_dim ** -0.5
        self.iters = args.slot_att_iter
        self.slot_dim = args.slot_dim
        self.use_adaptive_slot_attention = args.use_adaptive_slot_attention

        # === Slot related ===
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim))
        init.xavier_uniform_(self.slots_mu)
        init.xavier_uniform_(self.slots_logsigma)

        # === Slot Attention related ===
        self.Q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.norm = nn.LayerNorm(self.slot_dim)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP(self.slot_dim, 4 * self.slot_dim, self.slot_dim,
                       residual=True, layer_order="pre")
        # === === ===

        # === Query & Key & Value ===
        self.K = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.V = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

        # === === ===

        if self.use_adaptive_slot_attention:
            self.slot_gating_network = nn.Sequential(
                nn.LayerNorm(self.slot_dim),
                nn.Linear(self.slot_dim, 4 * self.slot_dim),
                nn.ReLU(inplace=True),
                nn.Linear(4 * self.slot_dim, 2)
            )
            nn.init.zeros_(self.slot_gating_network[1].bias)
            nn.init.zeros_(self.slot_gating_network[3].bias)

    def forward(self, inputs):
        # :arg inputs:              (B, token, D)
        #
        # :return slots:            (B, S, D_slot)

        B = inputs.shape[0]
        S = self.num_slots
        D_slot = self.slot_dim
        epsilon = 1e-8

        mu = self.slots_mu.expand(B, S, D_slot)
        sigma = self.slots_logsigma.exp().expand(B, S, D_slot)
        slots = mu + sigma * torch.randn(mu.shape, device=sigma.device, dtype=sigma.dtype)

        keys = self.K(inputs)                                # (B, token, D_slot)
        values = self.V(inputs)                              # (B, token, D_slot)
        
        for t in range(self.iters):
            assert torch.sum(torch.isnan(slots)) == 0, f"Iteration {t}"
            
            slots_prev = slots
            slots = self.norm(slots)
            queries = self.Q(slots)                                     # (B, S, D_slot)

            dots = torch.einsum('bsd,btd->bst', queries, keys)          # (B, S, token)
            dots *= self.scale                                          # (B, S, token)
            attn = dots.softmax(dim=1) + epsilon                        # (B, S, token)
            attn = attn / attn.sum(dim=-1, keepdim=True)                # (B, S, token)

            updates = torch.einsum('bst,btd->bsd', attn, values)        # (B, S, D_slot)

            slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim))

            slots = slots.reshape(B, -1, self.slot_dim)
            slots = self.mlp(slots)

        if self.use_adaptive_slot_attention:
            keep_drop_scores = self.slot_gating_network(slots) #  [B, S, 2]
            hard_keep_decision = F.gumbel_softmax(keep_drop_scores, hard=True, tau = 1)[...,1]  # [B, S]
            soft_keep_decision = F.softmax(keep_drop_scores, dim = -1)[...,1]  # only for logging purposes ; [B, S]
        else:
            hard_keep_decision = None
            soft_keep_decision = None

        return slots, hard_keep_decision, soft_keep_decision


class DINOSAURpp(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.slot_dim = args.slot_dim
        self.slot_num = args.num_slots
        self.token_num = args.token_num

        self.slot_encoder = nn.Sequential(nn.LayerNorm(768),
                                         nn.Linear(768, 768),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(768, self.slot_dim),
                                         nn.LayerNorm(self.slot_dim))

        nn.init.zeros_(self.slot_encoder[1].bias)
        nn.init.zeros_(self.slot_encoder[3].bias)

        self.slot_attention = SA(args, input_dim=self.slot_dim)

        self.slot_decoder = Decoder(args)

        self.pos_dec = nn.Parameter(torch.Tensor(1, self.token_num, self.slot_dim))
        init.normal_(self.pos_dec, mean=0., std=.02)

    def sbd_slots(self, slots):
        # :arg slots: (B, S, D_slot)
        # 
        # :return slots: (B, S, token, D_slot)

        B, S, D_slot = slots.shape

        slots = slots.view(-1, 1, D_slot)                   # (B * S, 1, D_slot)
        slots = slots.tile(1, self.token_num, 1)            # (B * S, token, D_slot)

        pos_embed = self.pos_dec.expand(slots.shape)
        slots = slots + pos_embed                          # (B * S, token, D_slot)
        slots = slots.view(B, S, self.token_num, D_slot)

        return slots
    
    
    def reconstruct_feature_map(self, slot_maps, hard_keep_decision=None):
        # :arg slot_maps: (B, S, token, 768 + 1)
        # :arg hard_keep_decision: (B, S) or None
        #
        # :return reconstruction: (B, token, 768)
        # :return masks: (B, S, token)

        B = slot_maps.shape[0]

        per_slot_reconstructions, masks = torch.split(slot_maps, [768, 1], dim=-1)  # (B, S, token, 768), (B, S, token, 1)
        masks = masks.softmax(dim=1)                                # (B, S, token, 1)

        if hard_keep_decision is not None:  # Adaptive Slot Attention
            masks = masks * hard_keep_decision.unsqueeze(-1).unsqueeze(-1)  # (B, S, token, 1)
            masks = masks / (masks.sum(dim=1, keepdim=True) + 1e-5)  # (B, S, token, 1) ; renormalize

        reconstruction = torch.sum(per_slot_reconstructions * masks, dim=1)         # (B, token, 768)
        masks = masks.squeeze(dim=-1)                               # (B, S, token)

        return reconstruction, masks, per_slot_reconstructions


    def forward(self, features):
        # :arg features: (B, token, 768)
        #
        # :return reconstruction: (B, token, 768)
        # :return slots: (B, S, D_slot)
        # :return masks: (B, S, token)

        features = self.slot_encoder(features)
        slots, hard_keep_decision, soft_keep_decision = self.slot_attention(features)  # (B, S, D_slot), (B, S) or None, (B, S) or None
        assert torch.sum(torch.isnan(slots)) == 0

        slot_maps = self.sbd_slots(slots)
        slot_maps = self.slot_decoder(slot_maps)                            # (B, S, token, 768 + 1)

        reconstruction, masks, per_slot_reconstructions = self.reconstruct_feature_map(slot_maps, hard_keep_decision)     # (B, token, 768), (B, S, token)

        return reconstruction, slots, masks, per_slot_reconstructions, hard_keep_decision, soft_keep_decision
