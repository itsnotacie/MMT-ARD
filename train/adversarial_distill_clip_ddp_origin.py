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
from train.utils import init_wandb, AverageMeter, LabelSmoothing
from train.sam_data import SamData
from open_flamingo.eval.models.utils import unwrap_model
from train.utils import str2bool

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import argparse

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
    model_teacher, _, image_processor = open_clip.create_model_and_transforms(
        args.teacher_arch, pretrained='openai'
    )
    if args.teacher_type == 'robust_PMG':
        if args.teacher_arch == 'ViT-L-14':
            robust_teacher_PMG_pretrained = "../save_ckpts/ViT-L-14_PMG_Fast2/final.pt"
            model_teacher, _, _ = load_clip_model(args.teacher_arch, robust_teacher_PMG_pretrained)
    ################### get teacher model ###################
    if args.optimizer_state != '':
        assert args.start_step > 0
        assert str(args.start_step) in args.optimizer_state
        assert args.pretrained in ['', 'none']
        args.pretrained = args.optimizer_state.replace('_opt', '')
    
    ################### student model loading ###################
    if args.student_type == 'vanilla':
        model, _, _ = load_clip_model(args.student_arch, "openai")

    # Get Original Model:
    model_orig, _, image_processor = open_clip.create_model_and_transforms(
        args.student_arch, pretrained='openai'
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
        if args.student_arch == 'ViT-B-32':
            assert embedding_text_labels_norm_student.shape == (512, 1000), embedding_text_labels_norm_student.shape
        elif args.student_arch in ('ViT-L-14', 'ViT-L-14-336'):
            assert embedding_text_labels_norm_student.shape == (768, 1000), embedding_text_labels_norm_student.shape
        else:
            raise ValueError(f'Unknown model: {args.student_arch}')
    ##################################### Student embedding text Generation #####################################

    model_teacher.cpu()
    model_teacher = ClipVisionModel(model=model_teacher.visual, args=args, normalize=normalize)
    model_teacher.cuda(local_rank)
    model_teacher = DDP(model_teacher, device_ids=[local_rank], output_device=local_rank)

    model.cpu()
    model = ClipVisionModel(model=model.visual, args=args, normalize=normalize)
    model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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
            model_orig=model_orig,
            dataloader=dataloader,
            dataloader_eval=dataloader_eval,
            optimizer=optimizer,
            scheduler=scheduler,
            embedding_text_labels_norm_student=embedding_text_labels_norm_student,
            embedding_text_labels_norm_teacher=embedding_text_labels_norm_teacher,
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
    def __init__(self, embedding_teacher, embedding_text_labels_norm_student, embedding_text_labels_norm_teacher, reduction='mean', loss=None,
                 logit_scale=100., embedding_orig_clean_ref=None):
        self.embedding_teacher = embedding_teacher
        self.embedding_text_labels_norm_student = embedding_text_labels_norm_student
        self.embedding_text_labels_norm_teacher = embedding_text_labels_norm_teacher
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale
        self.embedding_orig_clean_ref = embedding_orig_clean_ref

    def __call__(self, embedding, targets):
        return compute_adv_loss(
            loss_type=self.loss_str, embedding_student=embedding, targets=targets,
            embedding_teacher=self.embedding_teacher, logit_scale=self.logit_scale,
            embedding_text_labels_norm_student=self.embedding_text_labels_norm_student,
            embedding_text_labels_norm_teacher=self.embedding_text_labels_norm_teacher, 
            embedding_orig_clean_ref=self.embedding_orig_clean_ref, reduction=self.reduction)

def train_one_epoch(
        step_total, model, model_teacher, model_orig, dataloader, optimizer, scheduler, normalize,
        embedding_text_labels_norm_student, embedding_text_labels_norm_teacher, args, epoch, dataloader_eval=None
):
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    if dataloader_eval and isinstance(dataloader_eval.sampler, DistributedSampler):
        dataloader_eval.sampler.set_epoch(epoch)
    rank = dist.get_rank()

    model_teacher.eval()
    # model_orig.eval()
    model.train()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    racc_meter = AverageMeter('racc')

    epoch_start_time = time.time()
    for i, (data, targets) in enumerate(dataloader):
        is_classification = isinstance(targets, torch.Tensor)
        data = data.cuda()
        n_samples = data.shape[0]
        if is_classification:
            targets = targets.cuda()

        with torch.no_grad():
            embedding_teacher = model_teacher(vision=data, output_normalize=args.output_normalize)
            # embedding_orig_clean_ref = model_orig(data, output_normalize=args.output_normalize)
            embedding_orig_clean_ref = None

        # loss for the attack (adversary generation) during training
        loss_inner_wrapper = ComputeLossWrapper(
            embedding_teacher, embedding_text_labels_norm_student, embedding_text_labels_norm_teacher,
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

        del data, data_adv

        # if args.trades:
        #     embedding_clean_no_grad = embedding_clean.detach().clone()
        #     embedding_orig.cpu()

        ############################### Distillation Loss Computation ###############################
        
        loss_align_TS = compute_adv_loss(
            loss_type="KL", embedding_student=embedding_adv, targets=targets,
            embedding_teacher=embedding_teacher,
            logit_scale=100., embedding_text_labels_norm_student=embedding_text_labels_norm_student,
            embedding_text_labels_norm_teacher=embedding_text_labels_norm_teacher)

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
                embedding_teacher=None, embedding_text_labels_norm_student=embedding_text_labels_norm_student,
                embedding_text_labels_norm_teacher=embedding_text_labels_norm_teacher,
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

def compute_adv_loss(loss_type, embedding_student, targets, embedding_teacher, logit_scale,
                 embedding_text_labels_norm_student=None, embedding_text_labels_norm_teacher=None, 
                 embedding_orig_clean_ref=None, reduction='mean'):
    """
    Mostly for adv generation
    embedding --- Student embeddings
    embedding_teacher --- Teacher embeddings
    embedding_orig_clean_ref --- Original Model Embedding (Clean)
    adv_gen_loss: l2_orig|KL_orig
    """

    if loss_type == 'KL':
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