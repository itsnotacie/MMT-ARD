import sys

from train.datasets import COCOFlickrDataset, ImageNetDataset
from CLIP_eval.eval_utils import load_clip_model

sys.path.append("open_flamingo")
import os
os.environ["NCCL_DEBUG"] = "ERROR"
import shutil
import time
import string
import random

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from training.scheduler import cosine_lr
from CLIP_benchmark.clip_benchmark.metrics.linear_probe import cosine_lr
from torchvision import transforms
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
from train.pgd_train import pgd
from train.apgd_train import apgd_train as apgd
import wandb
from train.utils import init_wandb, AverageMeter#, LabelSmoothing
from train.sam_data import SamData
from open_flamingo.eval.models.utils import unwrap_model
from train.utils import str2bool

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import argparse

import torch.nn as nn
import torch
import torch.nn.functional as F
import transformers
import accelerate
import peft
from peft import LoraConfig, get_peft_model

"""


###### Basic Setups ######
(Batchsize:512 --- 2 Epochs (5000 Iterations))
(lr = 2e-5; Epsilon = 2/255)
###### Basic Setups ######

# Distillation
torchrun --nproc_per_node=8 --master_port=62502 train/adversarial_distill_clip_ddp.py --eps 2 --experiment_name Adv_Distill


# Every epoch number: 512 * 5000   (1 Epoch == 2500 Steps) 

"""

parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_name', type=str, default='ViT-L-14', help='ViT-L-14, ViT-B-32')
parser.add_argument('--pretrained', type=str, default='openai')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--template', type=str, default='std')
parser.add_argument('--imagenet_root', type=str, default='/mnt/bn/tiktok-mm-4/aiic/users/dongjunhao/datasets/ImageNet', help='Imagenet dataset root directory')
parser.add_argument('--output_normalize', type=str2bool, default=True, help='Whether the embedding is normalized')
parser.add_argument('--start_step', type=int, default=0, help='Start step for training')
parser.add_argument('--optimizer_state', type=str, default='', help='Optimizer state file path')
parser.add_argument('--steps', type=int, default=5000, help='Number of training steps --- 20000 (2 epochs)')
parser.add_argument('--warmup', type=int, default=350, help='Warmup steps')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--loss_clean', type=str, default='none', help='ce, l2')
parser.add_argument('--clean_weight', type=float, default=0., help='Weight for clean loss')
parser.add_argument('--trades', type=str2bool, default=False, help='Use TRADES')
parser.add_argument('--opt', type=str, default='adamw', help='Optimizer type; sgd, adamw')
parser.add_argument('--momentum_sgd', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--attack', type=str, default='pgd', help='Adversarial attack type')

parser.add_argument('--norm', type=str, default='linf', help='Norm for adversarial perturbation')
parser.add_argument('--eps', type=float, default=4, help='Epsilon for adversarial perturbation')
parser.add_argument('--iterations_adv', type=int, default=10, help='Iterations for adversarial attack')
parser.add_argument('--stepsize_adv', type=float, default=1., help='Step size for adversarial attack (no effect for apgd)')
parser.add_argument('--wandb', type=str2bool, default=False, help='Use Weights & Biases for logging')
parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--overwrite', type=str2bool, default=True, help='Overwrite existing directory')
parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency')
parser.add_argument('--eval_freq', type=int, default=10, help='Evaluation frequency')
parser.add_argument('--output_dir', type=str, default='../save_ckpts', help='Output directory')
parser.add_argument('--save_checkpoints', type=str2bool, default=True, help='Save 10 training checkpoints')
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')

# Adv_Distill
parser.add_argument('--teacher_type', type=str, default='robust_PMG', help='Teacher Type: robust_PMG')
parser.add_argument('--teacher_arch', type=str, default='ViT-L-14', help='Teacher backbone: ViT-L-14')
parser.add_argument('--student_type', type=str, default='vanilla', help='Student Type: vanilla')
parser.add_argument('--student_arch', type=str, default='ViT-B-32', help='Student backbone: ViT-B-32')

parser.add_argument('--loss', type=str, default='KL', help='KL')
parser.add_argument('--adv_gen_loss', type=str, default="KL", help="KL")


# Debug
parser.add_argument('--debug', type=str2bool, default=False, help='Run debug mode to check label alignment.')


class PlainMultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            dropout=0.,
            bias=True,
            kdim=None,
            vdim=None,
            batch_first=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            assert NotImplementedError
        else:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def init_weights(self):
        pass

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        E = query.size(-1)
        qkv = self.qkv(query)
        qkv = qkv.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None

    def set_parameters(self, torch_tgt_module):
        assert isinstance(torch_tgt_module, nn.MultiheadAttention)
        assert self.embed_dim == torch_tgt_module.embed_dim
        assert self.batch_first == torch_tgt_module.batch_first
        assert self.dropout == torch_tgt_module.dropout
        assert self.head_dim == torch_tgt_module.head_dim
        assert self.num_heads == torch_tgt_module.num_heads
        assert self.kdim == torch_tgt_module.kdim
        assert self.vdim == torch_tgt_module.vdim
        self.qkv.weight.data = torch_tgt_module.in_proj_weight.data
        self.qkv.bias.data = torch_tgt_module.in_proj_bias.data
        self.proj.weight.data = torch_tgt_module.out_proj.weight.data
        self.proj.bias.data = torch_tgt_module.out_proj.bias.data

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def main(args):
    # setup wandb
    rank = dist.get_rank()
    world_size = dist.get_world_size()


    if rank == 0:
        print(f"Distributed training: world_size={world_size}, local_rank={local_rank}")
        # print args
        print(f"Arguments:\n{'-' * 20}")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        print(f"{'-' * 20}")

        # setup dirs
        if args.overwrite:
            shutil.rmtree(args.output_dir, ignore_errors=True)
        os.makedirs(os.path.join(args.output_dir, 'ckpts'), exist_ok=False)

        # write args to file
        with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
            f.write(str(args))

    main_device = 0
    ################### get teacher model ###################
    model_teacher_origin, _, image_processor = open_clip.create_model_and_transforms(
        args.teacher_arch, pretrained='openai'
    )
    if args.teacher_type == 'robust_PMG':
        if args.teacher_arch == 'ViT-L-14':
            robust_teacher_PMG_pretrained = "../save_ckpts/ViT-L-14_PMG_Fast2/final.pt"
            model_teacher, _, _ = load_clip_model(args.teacher_arch, robust_teacher_PMG_pretrained)

            clean_teacher_pretrained = "output/ViT-L-14_openai_imagenet_ce_imagenet_NORMAL_TEACHER_u6dKf/checkpoints/final.pt"
            model_teacher_clean, _, _ = load_clip_model(args.teacher_arch, clean_teacher_pretrained)

    ################### get teacher model ###################
    if args.optimizer_state != '':
        assert args.start_step > 0
        assert str(args.start_step) in args.optimizer_state
        assert args.pretrained in ['', 'none']
        args.pretrained = args.optimizer_state.replace('_opt', '')
    
    ################### student model loading ###################
    if args.student_type == 'vanilla':
        if 'lora' in args.student_arch:
            model, _, _ = load_clip_model(args.student_arch.replace("-lora",""), "openai")
            #change module for lora
            print_trainable_parameters(model.visual)
            #trainable params: 87849216 || all params: 87849216 || trainable%: 100.00
            for module in model.visual.transformer.resblocks:
                new_module = PlainMultiHeadAttention()
                new_module.set_parameters(module.attn)
                module.attn = new_module
            print("update PlainMultiHeadAttention")
            print(model.visual)
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["qkv","proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[],
            )
            model.visual = get_peft_model(model.visual, config)
            print_trainable_parameters(model.visual)
            #trainable params: 884736 || all params: 88733952 || trainable%: 1.00
        else:
            model, _, _ = load_clip_model(args.student_arch, "openai")
    # Get Original Model:
    model_orig, _, image_processor = open_clip.create_model_and_transforms(
        args.student_arch.replace("-lora",""), pretrained='openai'
    )
    ################### student model loading ###################

    # Remove the Normalize transform by creating a new Compose object
    preprocessor_without_normalize = transforms.Compose(image_processor.transforms[:-1])
    normalize = image_processor.transforms[-1]
    del image_processor
    if rank == 0:
        print(f'[preprocessor_without_normalize] {preprocessor_without_normalize}')
        print(f'[normalize] {normalize}')
    # preprocessor_without_normalize contains following transforms:
    # - Resize(size=224, interpolation=bicubic, max_size=None, antialias=warn)
    # - CenterCrop(size=(224, 224))
    # - ToTensor()
    # normalize:
    # Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    # get data
    if args.dataset == 'imagenet':
        dataset = ImageNetDataset(
            root=args.imagenet_root + '/train',
            transform=preprocessor_without_normalize,
        )

    elif args.dataset == 'segment_anything':
        dataset = SamData('/data/naman_deep_singh/datasets/newSAM', transform=preprocessor_without_normalize)
        if rank == 0:
            print(dataset.__len__())
    elif args.dataset == 'coco':
        if os.path.exists('/mnt/datasets/coco'):
            image_dir_path = '/mnt/datasets/coco/train2017'
            annotations_path = '/mnt/datasets/coco/annotations/captions_train2017.json'
        elif os.path.exists('/mnt/lustre'):
            image_dir_path = '/mnt/lustre/hein/cschlarmann37/datasets/coco/train2017'
            annotations_path = '/mnt/lustre/hein/cschlarmann37/datasets/coco/annotations/captions_train2017.json'
        else:
            raise ValueError('COCO dataset not found')
        dataset = COCOFlickrDataset(
            image_dir_path=image_dir_path,
            annotations_path=annotations_path,
            transform=preprocessor_without_normalize
        )
    dataset_eval = ImageNetDataset(
        root=args.imagenet_root + '/val',
        transform=preprocessor_without_normalize,
    )


    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    # dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    local_batch_size = args.batch_size // world_size

    train_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=local_batch_size, num_workers=32, pin_memory=True, drop_last=True, sampler=train_sampler) 
    val_sampler = DistributedSampler(dataset_eval, shuffle=False, drop_last=False)
    dataloader_eval = DataLoader(dataset_eval, batch_size=local_batch_size, num_workers=32, pin_memory=True, drop_last=True, sampler=val_sampler)

    # Get text label embeddings of all ImageNet classes
    if args.template == 'std':
        template = 'This is a photo of a {}'
    elif args.template == 'blurry':
        template = 'This is a blurry photo of a {}'
    else:
        raise ValueError(f'Unknown template: {args.template}')
    if rank == 0:
        print(f'template: {template}')
    texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
    text_tokens = open_clip.tokenize(texts)
    ##################################### Teacher embedding text Generation #####################################
    model_teacher.cuda(local_rank)
    model_teacher_clean.cuda(local_rank)
    model_teacher_origin.cuda(local_rank)

    with torch.no_grad():
        embedding_text_labels_norm_teacher = []
        for el in (text_tokens[:500], text_tokens[500:]):
            # we need to split the text tokens into two batches because otherwise we run out of memory
            # note that we are accessing the model directly here, not the CustomModel wrapper
            # thus its always normalizing the text embeddings
            embedding_text_labels_norm_teacher.append(
                model_teacher.encode_text(el.cuda(local_rank), normalize=True).detach().cpu()
            )
        embedding_text_labels_norm_teacher = torch.cat(embedding_text_labels_norm_teacher).T.cuda(local_rank)
        assert torch.allclose(
            F.normalize(embedding_text_labels_norm_teacher, dim=0),
            embedding_text_labels_norm_teacher
        )
        if args.teacher_arch == 'ViT-B-32':
            assert embedding_text_labels_norm_teacher.shape == (512, 1000), embedding_text_labels_norm_teacher.shape
        elif args.teacher_arch in ('ViT-L-14', 'ViT-L-14-336'):
            assert embedding_text_labels_norm_teacher.shape == (768, 1000), embedding_text_labels_norm_teacher.shape
        else:
            raise ValueError(f'Unknown model: {args.teacher_arch}')
        
        embedding_text_labels_norm_teacher_clean = []
        embedding_text_labels_norm_teacher_origin = []
        for el in (text_tokens[:500], text_tokens[500:]):
            # we need to split the text tokens into two batches because otherwise we run out of memory
            # note that we are accessing the model directly here, not the CustomModel wrapper
            # thus its always normalizing the text embeddings
            embedding_text_labels_norm_teacher_clean.append(
                model_teacher_clean.encode_text(el.cuda(local_rank), normalize=True).detach().cpu()
            )
            embedding_text_labels_norm_teacher_origin.append(
                model_teacher_origin.encode_text(el.cuda(local_rank), normalize=True).detach().cpu()
            )
        embedding_text_labels_norm_teacher_clean = torch.cat(embedding_text_labels_norm_teacher_clean).T.cuda(local_rank)
        assert torch.allclose(
            F.normalize(embedding_text_labels_norm_teacher_clean, dim=0),
            embedding_text_labels_norm_teacher_clean
        )
        embedding_text_labels_norm_teacher_origin = torch.cat(embedding_text_labels_norm_teacher_origin).T.cuda(local_rank)
        assert torch.allclose(
            F.normalize(embedding_text_labels_norm_teacher_origin, dim=0),
            embedding_text_labels_norm_teacher_origin
        )
        if args.teacher_arch == 'ViT-B-32':
            assert embedding_text_labels_norm_teacher_clean.shape == (512, 1000), embedding_text_labels_norm_teacher_clean.shape
        elif args.teacher_arch in ('ViT-L-14', 'ViT-L-14-336'):
            assert embedding_text_labels_norm_teacher_clean.shape == (768, 1000), embedding_text_labels_norm_teacher_clean.shape
        else:
            raise ValueError(f'Unknown model: {args.teacher_arch}')
    ##################################### Teacher embedding text Generation #####################################

    ##################################### Student embedding text Generation #####################################
    model.cuda(local_rank)
    with torch.no_grad():
        embedding_text_labels_norm_student = []
        for el in (text_tokens[:500], text_tokens[500:]):
            # we need to split the text tokens into two batches because otherwise we run out of memory
            # note that we are accessing the model directly here, not the CustomModel wrapper
            # thus its always normalizing the text embeddings
            embedding_text_labels_norm_student.append(
                model.encode_text(el.cuda(local_rank), normalize=True).detach().cpu()
            )
        embedding_text_labels_norm_student = torch.cat(embedding_text_labels_norm_student).T.cuda(local_rank)
        assert torch.allclose(
            F.normalize(embedding_text_labels_norm_student, dim=0),
            embedding_text_labels_norm_student
        )
        if 'ViT-B-32' in args.student_arch:
            assert embedding_text_labels_norm_student.shape == (512, 1000), embedding_text_labels_norm_student.shape
        elif args.student_arch in ('ViT-L-14', 'ViT-L-14-336'):
            assert embedding_text_labels_norm_student.shape == (768, 1000), embedding_text_labels_norm_student.shape
        elif args.student_arch == 'RN50':
            assert embedding_text_labels_norm_student.shape == (1024, 1000), embedding_text_labels_norm_student.shape
        elif args.student_arch == 'RN101':
            assert embedding_text_labels_norm_student.shape == (512, 1000), embedding_text_labels_norm_student.shape
        else:
            print(embedding_text_labels_norm_student.shape)
            raise ValueError(f'Unknown model: {args.student_arch}')
    ##################################### Student embedding text Generation #####################################
    # print(len(embedding_text_labels_norm_teacher_clean),embedding_text_labels_norm_teacher_clean[0].shape) 768 1000
    
    model_teacher.cpu()
    model_teacher = ClipVisionModel(model=model_teacher.visual, args=args, normalize=normalize)
    #device_id = rank%torch.cuda.device_count()
    model_teacher = model_teacher.cuda(local_rank)
    print(f"Rank {local_rank} model_teacher Params: {len(list(model_teacher.parameters()))}")
    model_teacher = DDP(model_teacher, device_ids=[local_rank], output_device=local_rank)

    model_teacher_clean.cpu()
    model_teacher_clean = ClipVisionModel(model=model_teacher_clean.visual, args=args, normalize=normalize)
    model_teacher_clean = model_teacher_clean.cuda(local_rank)
    print(f"Rank {local_rank} model_teacher_clean Params: {len(list(model_teacher_clean.parameters()))}")
    model_teacher_clean = DDP(model_teacher_clean, device_ids=[local_rank], output_device=local_rank)

    model_teacher_origin.cpu()
    model_teacher_origin = ClipVisionModel(model=model_teacher_origin.visual, args=args, normalize=normalize)
    model_teacher_origin = model_teacher_origin.cuda(local_rank)
    print(f"Rank {local_rank} model_teacher_origin Params: {len(list(model_teacher_origin.parameters()))}")
    model_teacher_origin = DDP(model_teacher_origin, device_ids=[local_rank], output_device=local_rank)
    print("Load teacher model done")
    model.cpu()
    model = ClipVisionModel(model=model.visual, args=args, normalize=normalize)
    model = model.cuda(local_rank)
    print(f"Rank {local_rank} model Params: {len(list(model.parameters()))}")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False,)

    # model_orig.cpu()
    # model_orig = ClipVisionModel(model=model_orig.visual, args=args, normalize=normalize)
    # model_orig.cuda(local_rank)
    # model_orig = DDP(model_orig, device_ids=[local_rank], output_device=local_rank)
    model_orig = None

    # set optimizer (all params have requires_grad=True)
    params = unwrap_model(model).model.parameters()

    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum_sgd,
            weight_decay=args.wd
        )
    else:
        raise ValueError(f'Optimizer {args.optimizer} not supported.')
    if args.optimizer_state != '':
        optimizer.load_state_dict(torch.load(args.optimizer_state))

    # set scheduler
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.steps)

    # compute amount of epochs
    total_epochs = args.steps / len(dataloader)
    if rank == 0:
        print(f'train for {total_epochs} epochs')
    args.total_epochs = total_epochs


    # finetune
    step_total = args.start_step
    epoch = 0
    while step_total < args.steps:
        step_total = train_one_epoch(
            step_total,
            model=model,
            model_teacher=model_teacher,
            model_teacher_clean=model_teacher_clean,
            model_teacher_origin=model_teacher_origin,
            model_orig=model_orig,
            dataloader=dataloader,
            dataloader_eval=dataloader_eval,
            optimizer=optimizer,
            scheduler=scheduler,
            embedding_text_labels_norm_student=embedding_text_labels_norm_student,
            embedding_text_labels_norm_teacher=embedding_text_labels_norm_teacher,
            embedding_text_labels_norm_teacher_clean=embedding_text_labels_norm_teacher_clean,
            embedding_text_labels_norm_teacher_origin=embedding_text_labels_norm_teacher_origin,
            normalize=normalize,
            args=args,
            epoch=epoch
        )
        if rank == 0:
            print(f'Epoch {epoch} done.')
        epoch += 1

    if rank == 0:
        # save final model
        torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/final.pt')
        torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/ckpts/final.pt')
        torch.save(optimizer.state_dict(), f'{args.output_dir}/ckpts/final_opt.pt')

        # if args.output_dir.endswith('_temp'):
        #     # rename temp dir to final dir
        #     os.rename(args.output_dir, args.output_dir[:-5])

class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, args, normalize):
        super().__init__()
        self.model = model
        self.args = args
        self.normalize = normalize

    def forward(self, vision, output_normalize):
        embedding = self.model(self.normalize(vision))
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding


class ComputeLossWrapper:
    def __init__(self, embedding_teacher, embedding_teacher_clean,embedding_teacher_origin,embedding_text_labels_norm_student, embedding_text_labels_norm_teacher, embedding_text_labels_norm_teacher_clean, embedding_text_labels_norm_teacher_origin,reduction='mean', loss=None,
                 logit_scale=100., embedding_orig_clean_ref=None):
        self.embedding_teacher = embedding_teacher  
        self.embedding_teacher_clean = embedding_teacher_clean
        self.embedding_teacher_origin = embedding_teacher_origin
        self.embedding_text_labels_norm_student = embedding_text_labels_norm_student
        self.embedding_text_labels_norm_teacher = embedding_text_labels_norm_teacher
        self.embedding_text_labels_norm_teacher_clean = embedding_text_labels_norm_teacher_clean
        self.embedding_text_labels_norm_teacher_origin = embedding_text_labels_norm_teacher_origin
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale
        self.embedding_orig_clean_ref = embedding_orig_clean_ref

    def __call__(self, embedding, targets):
        return compute_adv_loss(
            loss_type=self.loss_str, embedding_student=embedding, embedding_student_clean=None,targets=targets,
            embedding_teacher=self.embedding_teacher, embedding_teacher_clean=self.embedding_teacher_clean, embedding_teacher_origin=self.embedding_teacher_origin,
            logit_scale=self.logit_scale,
            embedding_text_labels_norm_student=self.embedding_text_labels_norm_student,
            embedding_text_labels_norm_teacher=self.embedding_text_labels_norm_teacher, 
            embedding_text_labels_norm_teacher_clean=self.embedding_text_labels_norm_teacher_clean,
            embedding_text_labels_norm_teacher_origin=self.embedding_text_labels_norm_teacher_origin,
            embedding_orig_clean_ref=self.embedding_orig_clean_ref, reduction=self.reduction)

def train_one_epoch(
        step_total, model, model_teacher, model_teacher_clean, model_teacher_origin, model_orig, dataloader, optimizer, scheduler, normalize,
        embedding_text_labels_norm_student, embedding_text_labels_norm_teacher, embedding_text_labels_norm_teacher_clean, embedding_text_labels_norm_teacher_origin,args, epoch, dataloader_eval=None
):
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    if dataloader_eval and isinstance(dataloader_eval.sampler, DistributedSampler):
        dataloader_eval.sampler.set_epoch(epoch)
    rank = dist.get_rank()

    model_teacher.eval()
    model_teacher_clean.eval()
    model_teacher_origin.eval()
    # model_orig.eval()
    model.train()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    racc_meter = AverageMeter('racc')

    epoch_start_time = time.time()


    

    for i, (data, targets) in enumerate(dataloader):
        print(f'Epoch {epoch} step {i}')
        is_classification = isinstance(targets, torch.Tensor)
        data = data.cuda()
        n_samples = data.shape[0]
        if is_classification:
            targets = targets.cuda()

        with torch.no_grad():
            # embedding_teacher = model_teacher(vision=data, output_normalize=args.output_normalize) 
            # #这里是用于生成对抗样本的，
            #目标：使 model 对 data_adv 的输出接近 model_teacher 对 data 的输出（通过 loss_inner_wrapper 实现）。
            embedding_teacher_clean = model_teacher_clean(vision=data, output_normalize=args.output_normalize)
            # embedding_orig_clean_ref = model_orig(data, output_normalize=args.output_normalize)
            embedding_orig_clean_ref = None

        # loss for the attack (adversary generation) during training
        loss_inner_wrapper = ComputeLossWrapper(
            embedding_teacher_clean,  None, None, embedding_text_labels_norm_student, embedding_text_labels_norm_teacher_clean, None, None,
            reduction='none' if args.attack == 'apgd' else 'mean', loss=args.adv_gen_loss,
            logit_scale=100., embedding_orig_clean_ref=embedding_orig_clean_ref)
        model.eval()
        
        if args.attack == 'pgd':
            data_adv = pgd(
                forward=model,
                loss_fn=loss_inner_wrapper,
                data_clean=data,
                targets=targets,
                norm=args.norm,
                eps=args.eps,
                iterations=args.iterations_adv,
                stepsize=args.stepsize_adv,
                output_normalize=args.output_normalize,
                perturbation=torch.zeros_like(data).uniform_(-args.eps, args.eps).requires_grad_(True),
                mode='max',
                verbose=False
            )
        elif args.attack == 'apgd':
            # apgd currently always applies output normalization
            data_adv = apgd(
                model=model,
                loss_fn=loss_inner_wrapper,
                x=data,
                y=targets,
                norm=args.norm,
                eps=args.eps,
                n_iter=args.iterations_adv,
                verbose=True
            )
        elif args.attack == 'none':
            data_adv = data

        del loss_inner_wrapper
        model.train()

        # Student --- clean and adv embeddings
        embedding_clean = model(data, output_normalize=args.output_normalize)
        embedding_adv = model(data_adv, output_normalize=args.output_normalize)

        with torch.no_grad():
            embedding_teacher = model_teacher(vision=data_adv, output_normalize=args.output_normalize) 
            embedding_teacher_origin = model_teacher_origin(vision=data, output_normalize=args.output_normalize)
        del data, data_adv


        # if args.trades:
        #     embedding_clean_no_grad = embedding_clean.detach().clone()
        #     embedding_orig.cpu()

        ############################### Distillation Loss Computation ###############################
        
        loss_align_TS = compute_adv_loss(
            loss_type="KL", embedding_student=embedding_adv, embedding_student_clean=embedding_clean, targets=targets,
            embedding_teacher=embedding_teacher, embedding_teacher_clean=embedding_teacher_clean, embedding_teacher_origin=embedding_teacher_origin,
            logit_scale=100., embedding_text_labels_norm_student=embedding_text_labels_norm_student,
            embedding_text_labels_norm_teacher=embedding_text_labels_norm_teacher,
            embedding_text_labels_norm_teacher_clean=embedding_text_labels_norm_teacher_clean,
            embedding_text_labels_norm_teacher_origin=embedding_text_labels_norm_teacher_origin)

        loss_total = loss_align_TS

        ############################### Distillation Loss Computation ###############################

        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_total += 1
        scheduler(step_total)

        with torch.no_grad():
            # only for logging
            embedding_teacher.cuda()
            if is_classification:
                logits_adv = embedding_adv @ embedding_text_labels_norm_student
                racc = compute_acc(logits_adv, targets)
                embedding_clean_norm = F.normalize(embedding_clean, dim=1)
                logits_clean = embedding_clean_norm @ embedding_text_labels_norm_student
                acc = compute_acc(logits_clean, targets)
                acc_meter.update(acc, n_samples)
                racc_meter.update(racc, n_samples)
                del embedding_clean_norm, embedding_clean
            else:
                acc = None
                racc = None

        loss_meter.update(loss_total.item(), n_samples)

        eval_logs = dict()
        if (step_total-1) % args.eval_freq == 0:
            # we compute acc and racc (against supervised apgd) on validation data
            model.eval()
            data_eval, targets_eval = next(iter(dataloader_eval))
            data_eval, targets_eval = data_eval.cuda(), targets_eval.cuda()
            loss_eval_wrapper = ComputeLossWrapper(
                embedding_teacher=None, embedding_teacher_clean=None, embedding_teacher_origin=None, embedding_text_labels_norm_student=embedding_text_labels_norm_student,
                embedding_text_labels_norm_teacher=embedding_text_labels_norm_teacher,
                embedding_text_labels_norm_teacher_clean=embedding_text_labels_norm_teacher_clean,
                embedding_text_labels_norm_teacher_origin = embedding_text_labels_norm_teacher_origin,
                reduction='none', loss='CE_label', logit_scale=100.
                )
            data_eval_adv = apgd(
                model=model,
                loss_fn=loss_eval_wrapper,
                x=data_eval,
                y=targets_eval,
                norm=args.norm,
                eps=args.eps,
                n_iter=50,
                initial_stepsize=0.05 * args.eps if args.clean_weight > 0 else None,
                verbose=False
            )
            with torch.no_grad():
                embedding_adv_eval_norm = model(data_eval_adv, output_normalize=True)  # we set output_normalize to True
                logits_eval_adv = embedding_adv_eval_norm @ embedding_text_labels_norm_student
                racc_eval = compute_acc(logits_eval_adv, targets_eval)
                embedding_eval_norm = model(data_eval, output_normalize=True)
                logits_eval = embedding_eval_norm @ embedding_text_labels_norm_student
                acc_eval = compute_acc(logits_eval, targets_eval)
                # note we compute the cosine sim between clean and adv embedding,
                # not between orig and adv embedding as for training
            eval_logs['eval/racc'] = racc_eval
            eval_logs['eval/acc'] = acc_eval
            if rank == 0:
                print(f'[eval-acc] {acc_eval:.2f} [eval-racc] {racc_eval:.2f}')
            model.train()
            del data_eval_adv, data_eval, targets_eval, embedding_adv_eval_norm, logits_eval_adv, embedding_eval_norm, logits_eval

        lr_ = optimizer.param_groups[0].get('lr')
        if (step_total-1) % args.log_freq == 0:
            log_str = f'[step] {step_total} [lr] {lr_:.6f} [loss] {loss_total.item():.6f}'
            if is_classification:
                log_str += f' [acc] {acc:.2f} [racc] {racc:.2f}'
            if rank == 0:
                print(log_str)

        if rank == 0:
            # save 10 models over the course of training
            if args.save_checkpoints and (step_total % (args.steps // 10) == 0):
                # save model and optimizer state_dict
                torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/ckpts/step_{step_total}.pt')
                torch.save(optimizer.state_dict(), f'{args.output_dir}/ckpts/step_{step_total}_opt.pt')
            # every 200 steps, save a fallback model, which gets overwritten
            if step_total % 200 == 0:
                torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/ckpts/fallback_{step_total}.pt')
                torch.save(optimizer.state_dict(), f'{args.output_dir}/ckpts/fallback_{step_total}_opt.pt')
                # remove old fallback models
                for file in os.listdir(f'{args.output_dir}/ckpts'):
                    if file.startswith('fallback') and not str(step_total) in file:
                        os.remove(f'{args.output_dir}/ckpts/{file}')

        if step_total >= args.steps:
            break

        torch.cuda.empty_cache()
    return step_total


@torch.no_grad()
def compute_acc(logits, targets):
    preds_clean = logits.max(dim=1)[1].detach()
    acc = (preds_clean.eq(targets).sum() / targets.shape[0]).item() * 100
    return acc

########################################################## Training Loss ##########################################################

# def compute_loss(loss_str, embedding, targets, embedding_teacher, logit_scale,
#                  embedding_text_labels_norm=None, reduction='mean'):
#     if loss_str == ''
#     if loss_str == 'l2':
#         loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
#     elif loss_str == 'ce':
#         loss = ce(
#             out=embedding @ (logit_scale * embedding_text_labels_norm),
#             targets=targets,
#             reduction=reduction
#         )
#     else:
#         raise ValueError(f'loss {loss_str} not supported')
#     return loss

# weight = {
#     "adv_loss": 1/2.0,
#     "nat_loss": 1/2.0,
# }
# init_loss_nat = None
# init_loss_adv = None
# weight_learn_rate = 0.025
# bert = 1
def compute_adv_loss(loss_type, embedding_student, embedding_student_clean,targets, embedding_teacher,  embedding_teacher_clean, embedding_teacher_origin, logit_scale,
                 embedding_text_labels_norm_student=None, embedding_text_labels_norm_teacher=None,embedding_text_labels_norm_teacher_clean=None,embedding_text_labels_norm_teacher_origin=None,
                 embedding_orig_clean_ref=None, reduction='mean'):
    """
    Mostly for adv generation
    embedding --- Student embeddings
    embedding_teacher --- Teacher embeddings
    embedding_orig_clean_ref --- Original Model Embedding (Clean)
    adv_gen_loss: l2_orig|KL_orig
    """
    # global weight
    # global init_loss_nat
    # global init_loss_adv
    # global weight_learn_rate
    # global bert
    
    if loss_type == 'KL':
        if embedding_teacher_clean is not None:
            # loss = kl(out=embedding_student @ (logit_scale * embedding_text_labels_norm_student),
            #         targets=embedding_teacher @ (logit_scale * embedding_text_labels_norm_teacher),
            #         reduction='batchmean')
            # print(loss.shape)   torch.Size([])

            loss = kl(out=embedding_student @ (logit_scale * embedding_text_labels_norm_student),
                    targets=embedding_teacher @ (logit_scale * embedding_text_labels_norm_teacher),
                    reduction='none')
            loss_clean = kl(out=embedding_student_clean @ (logit_scale * embedding_text_labels_norm_student),
                    targets=embedding_teacher_clean @ (logit_scale * embedding_text_labels_norm_teacher_clean),
                    reduction='none')
            loss_origin = kl(out=embedding_student_clean @ (logit_scale * embedding_text_labels_norm_student),
                    targets=embedding_teacher_origin @ (logit_scale * embedding_text_labels_norm_teacher_origin),
                    reduction='none')


            logits_adv = embedding_teacher @ embedding_text_labels_norm_teacher
            logits_clean = embedding_teacher_clean @ embedding_text_labels_norm_teacher_clean

            batch_ids = torch.arange(logits_adv.size(0))  # 生成样本序号 [0, 1, 2, ..., 127]
            values_adv = logits_adv[batch_ids, targets] #对应target的置信度
            values_clean = logits_clean[batch_ids, targets]


            loss = loss*values_adv.unsqueeze(1)*50 + loss_clean*values_clean.unsqueeze(1)*5
  
            loss = torch.mean(loss)

        else:
            loss = kl(out=embedding_student @ (logit_scale * embedding_text_labels_norm_student),
                    targets=embedding_teacher @ (logit_scale * embedding_text_labels_norm_teacher),
                    reduction='batchmean')

            
    elif loss_type == 'CE_label':
        loss = ce(out=embedding_student @ (logit_scale * embedding_text_labels_norm_student),
                targets=targets,reduction=reduction)
    else:
        raise ValueError(f'loss {loss_type} not supported')
    return loss

def l2(out, targets, reduction='none'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    assert out.shape[0] > 1
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
    return squared_error_batch

def ce(out, targets, reduction='mean'):
    # out = logits
    assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
    assert out.shape[0] > 1

    return F.cross_entropy(out, targets, reduction=reduction)


def kl(out, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(out, dim=1), F.softmax(targets, dim=1), reduction=reduction)

########################################################## Training Loss ##########################################################

if __name__ == '__main__':
    # DDP init
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Parse command-line arguments
    args = parser.parse_args()
    args.eps /= 255
    args.stepsize_adv /= 255
    args.warmup = int(0.07*args.steps)
    # make sure there is no string in args that should be a bool
    assert not any([isinstance(x, str) and x in ['True', 'False'] for x in args.__dict__.values()]), f'args contains a string that should be a bool: {args}'
    assert args.eval_freq % args.log_freq == 0, 'eval_freq must be a multiple of log_freq'

    if rank == 0:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f'Number of GPUs available: {num_gpus}')
        else:
            print('No multiple GPUs available.')

    # set model name and output dir
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    args.finetuned_model_name = f'{args.student_arch}_{args.experiment_name}'
    args.finetuned_model_name = args.finetuned_model_name.replace('/', '_')
    args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)
    # run
    main(args)