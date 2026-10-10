import os
import sys
import shutil
import argparse
import logging
import random
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloaders.dataset_bhsd import BHSDDataset, RandomRotFlip, RandomCrop, ToTensor
from dataloaders.dataset import TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.BCP_utils import context_mask

parser = argparse.ArgumentParser(description="PICK Training for BHSD Multi-class Hemorrhage Segmentation (6 classes)")
parser.add_argument('--root_path', type=str, default='/home/u001015/dataset/BHSD_h5/', help='Root directory of BHSD H5 dataset')
parser.add_argument('--exp', type=str, default='BHSD_multiclass_PICK', help='Experiment name')
parser.add_argument('--model', type=str, default='VNet', help='Model architecture')
parser.add_argument('--pre_max_iteration', type=int, default=4000, help='Maximum pre-train iterations')
parser.add_argument('--self_max_iteration', type=int, default=12000, help='Maximum self-train iterations')
parser.add_argument('--max_samples', type=int, default=150, help='Total training samples pool')
parser.add_argument('--labeled_bs', type=int, default=2, help='Batch size for labeled samples')
parser.add_argument('--batch_size', type=int, default=4, help='Total batch size (labeled + unlabeled)')
parser.add_argument('--base_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--deterministic', type=int, default=1, help='Whether to use deterministic training')
parser.add_argument('--labelnum', type=int, default=15, help='Number of labeled samples (~10% or ~20%)')
parser.add_argument('--gpu', type=str, default='auto', help='GPU ID to use (e.g. 0, 0,1, auto, or all)')
parser.add_argument('--seed', type=int, default=1337, help='Random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='Consistency loss weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='Consistency ramp-up epochs')
parser.add_argument('--lambda_', type=float, default=0.2, help='MIM loss balance weight')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='Mask ratio for CutMix and context masking')
parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4], help='Fold index for 5-fold cross-validation')
parser.add_argument('--exp_dir', type=str, default='../experiments', help='Base directory to save experiments')
parser.add_argument('--resume', action='store_true', default=False, help='Resume training from latest checkpoint if available')
parser.add_argument('--phase', type=str, default='all', choices=['all', 'pre_train', 'self_train'], help='Phase to execute: all, pre_train, or self_train')
parser.add_argument('--pretrain_checkpoint', type=str, default='', help='Path to pre-trained checkpoint for self-training')
parser.add_argument('--val_interval', type=int, default=500, help='Iterations interval between validations (e.g. 500)')
parser.add_argument('--val_stride_xy', type=int, default=32, help='Sliding-window XY stride during validation (32 or 48)')
parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable Weights & Biases logging')
parser.add_argument('--save_wandb_model', action='store_true', default=False, help='Save best model checkpoints to W&B Cloud Artifacts')
parser.add_argument('--wandb_project', type=str, default='PICK_BHSD', help='W&B project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/username/team')
parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run display name')
args = parser.parse_args()

if args.gpu not in ['auto', 'all']:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (32, 160, 160)
num_classes = 6  # 0: Background, 1: EDH, 2: IPH, 3: IVH, 4: SAH, 5: SDH
CLASS_NAMES = ["Background", "Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"]


def save_checkpoint(net, optimizer, iter_num, best_dice, phase, path):
    """Save full training state including weights, optimizer, iter_num, and best_dice."""
    raw_net = net.module if isinstance(net, nn.DataParallel) else net
    state = {
        'net': raw_net.state_dict(),
        'opt': optimizer.state_dict() if optimizer is not None else None,
        'iter_num': iter_num,
        'best_dice': best_dice,
        'phase': phase
    }
    torch.save(state, str(path))


def load_checkpoint(net, optimizer, path):
    """Load training state for resuming, handling DataParallel prefix seamlessly."""
    state = torch.load(str(path), map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    net_dict = state.get('net', state)
    raw_net = net.module if isinstance(net, nn.DataParallel) else net
    clean_dict = {k[7:] if k.startswith('module.') else k: v for k, v in net_dict.items()}
    raw_net.load_state_dict(clean_dict)

    if optimizer is not None and 'opt' in state and state['opt'] is not None:
        optimizer.load_state_dict(state['opt'])

    iter_num = state.get('iter_num', 0)
    best_dice = state.get('best_dice', 0.0)
    return iter_num, best_dice


def save_net_opt(net, optimizer, path, iter_num=0, best_dice=0.0, phase='train'):
    save_checkpoint(net, optimizer, iter_num, best_dice, phase, path)


def load_net(net, path):
    """Load model weights, handling DataParallel prefix seamlessly."""
    state = torch.load(str(path), map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    net_dict = state.get('net', state)
    raw_net = net.module if isinstance(net, nn.DataParallel) else net
    clean_dict = {k[7:] if k.startswith('module.') else k: v for k, v in net_dict.items()}
    raw_net.load_state_dict(clean_dict)


def restore_from_wandb(dest_dir, artifact_name, project, entity=None):
    """Attempt to restore a checkpoint artifact from W&B Cloud."""
    try:
        import wandb
        api = wandb.Api()
        if not entity:
            try:
                entity = api.default_entity
            except Exception:
                entity = None

        candidates = []
        if entity:
            candidates.append(f"{entity}/{project}/{artifact_name}:latest")
        candidates.append(f"{project}/{artifact_name}:latest")
        candidates.append(f"{artifact_name}:latest")

        for cand in candidates:
            try:
                artifact = api.artifact(cand)
                os.makedirs(dest_dir, exist_ok=True)
                artifact.download(root=dest_dir)
                logging.info(f"[W&B CLOUD] Successfully restored artifact '{cand}' to {dest_dir}!")
                return True
            except Exception:
                continue
    except Exception as e:
        logging.warning(f"[W&B CLOUD] Failed to restore from W&B API: {e}")
    return False


def cal_multiclass_dice(pred, gt, num_classes=6):
    """Calculate per-class Dice scores for classes 1..num_classes-1."""
    class_dices = []
    for c in range(1, num_classes):
        p_c = (pred == c).astype(np.float32)
        g_c = (gt == c).astype(np.float32)
        intersection = np.sum(p_c * g_c)
        total = np.sum(p_c) + np.sum(g_c)
        if total == 0:
            dice = 1.0  # True negative agreement
        else:
            dice = (2.0 * intersection) / total
        class_dices.append(dice)
    return class_dices


def validate_multiclass(model, val_dataset, patch_size=(32, 160, 160), stride_z=8, stride_xy=32):
    """Sliding-window 3D inference for multi-class (6 classes) evaluation."""
    model.eval()
    all_case_dices = []  # List of [dice_c1, dice_c2, ..., dice_c5]

    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample, _ = val_dataset[i]
            image, label = sample['image'], sample['label']
            z, y, x = image.shape

            pz = max(patch_size[0] - z, 0)
            py = max(patch_size[1] - y, 0)
            px = max(patch_size[2] - x, 0)
            if pz > 0 or py > 0 or px > 0:
                image = np.pad(image, [(0, pz), (0, py), (0, px)], mode='constant', constant_values=0)

            w_z, w_y, w_x = image.shape
            sz = max(int(np.ceil((w_z - patch_size[0]) / stride_z)) + 1, 1)
            sy = max(int(np.ceil((w_y - patch_size[1]) / stride_xy)) + 1, 1)
            sx = max(int(np.ceil((w_x - patch_size[2]) / stride_xy)) + 1, 1)

            score_map = np.zeros((num_classes, w_z, w_y, w_x), dtype=np.float32)
            cnt = np.zeros((w_z, w_y, w_x), dtype=np.float32)

            for iz in range(sz):
                zs = min(iz * stride_z, w_z - patch_size[0])
                for iy in range(sy):
                    ys = min(iy * stride_xy, w_y - patch_size[1])
                    for ix in range(sx):
                        xs = min(ix * stride_xy, w_x - patch_size[2])

                        patch = image[zs:zs + patch_size[0], ys:ys + patch_size[1], xs:xs + patch_size[2]]
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).cuda().float()

                        _, out, _ = model(patch_tensor)
                        prob = F.softmax(out, dim=1).squeeze(0).cpu().numpy()

                        score_map[:, zs:zs + patch_size[0], ys:ys + patch_size[1], xs:xs + patch_size[2]] += prob
                        cnt[zs:zs + patch_size[0], ys:ys + patch_size[1], xs:xs + patch_size[2]] += 1.0

            score_map = score_map / np.expand_dims(cnt, axis=0)
            pred = np.argmax(score_map, axis=0)[:z, :y, :x]

            case_dices = cal_multiclass_dice(pred, label, num_classes=num_classes)
            all_case_dices.append(case_dices)

    if len(all_case_dices) == 0:
        return 0.0, [0.0] * 5

    mean_per_class = np.mean(np.array(all_case_dices), axis=0)
    mDice = float(np.mean(mean_per_class))
    return mDice, list(mean_per_class)


def pre_train(snapshot_path, val_dataset):
    print("=== [BHSD Multi-class] STARTING PRE-TRAINING PHASE (6 Classes) ===")
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
        logging.info(f"[Model] Pre-train Multi-GPU enabled with {torch.cuda.device_count()} GPUs (DataParallel)")
    else:
        model = model.cuda()
        logging.info(f"[Model] Pre-train Single-GPU enabled on {torch.cuda.get_device_name(0)}")

    db_train = BHSDDataset(
        base_dir=args.root_path,
        split='train',
        binary=False,  # Retain multi-class labels (0..5)
        patch_size=patch_size,
        fold=args.fold,
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor()
        ])
    )

    labelnum = min(args.labelnum, len(db_train))
    max_samples = len(db_train) if args.max_samples <= 0 else min(args.max_samples, len(db_train))
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, max_samples))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=False)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    model.train()
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    iter_num = 0
    best_dice = 0.0

    latest_pth = os.path.join(snapshot_path, "checkpoint_latest.pth")
    if args.resume and not os.path.exists(latest_pth) and args.use_wandb:
        logging.info("[RESUME] Local pre-train checkpoint not found. Attempting to restore from W&B Cloud...")
        restore_from_wandb(snapshot_path, f"{args.exp}_pretrain_latest", args.wandb_project, args.wandb_entity)

    if args.resume and os.path.exists(latest_pth):
        iter_num, best_dice = load_checkpoint(model, optimizer, latest_pth)
        logging.info(f"[RESUME] Resumed pre-training from iter {iter_num} | best_mDice: {best_dice:.4f}")
        if iter_num >= args.pre_max_iteration:
            logging.info(f"[RESUME] Pre-training already completed ({iter_num}/{args.pre_max_iteration}). Skipping pre-train phase.")
            writer.close()
            return

    remaining_iters = max(args.pre_max_iteration - iter_num, 0)
    max_epoch = remaining_iters // max(len(trainloader), 1) + 2

    for _ in range(max_epoch):
        for _, (sampled_batch, mim_mask) in enumerate(trainloader):
            volume_batch = sampled_batch['image'][:args.labeled_bs].cuda()
            label_batch = sampled_batch['label'][:args.labeled_bs].cuda()
            mim_mask = mim_mask[:args.labeled_bs].unsqueeze(1).cuda().float()

            img_a, img_b = volume_batch, torch.flip(volume_batch, dims=[0])
            lab_a, lab_b = label_batch, torch.flip(label_batch, dims=[0])

            with torch.no_grad():
                img_mask, _ = context_mask(img_a, args.mask_ratio)

            cutmix_batch = img_a * img_mask + img_b * (1 - img_mask)
            cutmix_label = lab_a * img_mask + lab_b * (1 - img_mask)
            num_cutmix = cutmix_batch.shape[0]

            _, main_outputs, _ = model(torch.cat((cutmix_batch, volume_batch), dim=0))

            main_prob = F.softmax(main_outputs.detach(), dim=1)
            threshold = 0.5 if iter_num > 2000 else 0.2
            # Sum of probabilities across all 5 foreground lesion classes
            fg_prob = main_prob[num_cutmix:, 1:, :, :, :].sum(dim=1, keepdim=True)
            main_ps_lab = (fg_prob > threshold).float()

            mask_region = main_ps_lab if iter_num > 2000 else mim_mask
            mim_batch = volume_batch * (1 - mask_region)
            mim_outputs = model(mim_batch, mode='mim')

            re_batch = volume_batch * (1 - mask_region) + mim_outputs.detach() * mask_region
            aux_outputs = model(re_batch, mode='aux')

            combined_label = torch.cat((cutmix_label, label_batch), dim=0)
            loss_main_ce = F.cross_entropy(main_outputs, combined_label)
            loss_main_dice = DICE(main_outputs, combined_label)
            loss_main = (loss_main_ce + loss_main_dice) / 2

            loss_mim = F.l1_loss(volume_batch, mim_outputs, reduction='none')
            loss_mim = args.lambda_ * (loss_mim * mask_region).sum() / (mask_region.sum() + 1e-5)

            loss_aux_ce = F.cross_entropy(aux_outputs, label_batch)
            loss_aux_dice = DICE(aux_outputs, label_batch)
            loss_aux = (loss_aux_ce + loss_aux_dice) / 2

            loss = loss_main + loss_mim + loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('loss/total', loss.item(), iter_num)
            writer.add_scalar('loss/main', loss_main.item(), iter_num)
            writer.add_scalar('loss/mim', loss_mim.item(), iter_num)
            writer.add_scalar('loss/aux', loss_aux.item(), iter_num)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_num)

            if args.use_wandb:
                import wandb
                wandb.log({
                    'train/loss_total': loss.item(),
                    'train/loss_main': loss_main.item(),
                    'train/loss_mim': loss_mim.item(),
                    'train/loss_aux': loss_aux.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'phase': 'pre_train',
                    'pre_train/iter': iter_num,
                }, step=iter_num)

            if iter_num % 50 == 0:
                logging.info(f"Pre-train Iter {iter_num}: Loss: {loss.item():.4f}, Main: {loss_main.item():.4f}, MIM: {loss_mim.item():.4f}")

            if iter_num % args.val_interval == 0:
                val_mdice, per_class = validate_multiclass(model, val_dataset, patch_size=patch_size, stride_xy=args.val_stride_xy)
                class_str = ", ".join([f"{CLASS_NAMES[c+1]}: {d:.3f}" for c, d in enumerate(per_class)])
                logging.info(f"[Pre-train Validation] Iter {iter_num} | mDice: {val_mdice:.4f} ({class_str})")
                writer.add_scalar('val/mDice', val_mdice, iter_num)
                for c, d in enumerate(per_class):
                    writer.add_scalar(f'val/dice_{CLASS_NAMES[c+1]}', d, iter_num)
                if val_mdice > best_dice:
                    best_dice = val_mdice
                    best_model_path = os.path.join(snapshot_path, f"{args.model}_best_model.pth")
                    save_checkpoint(model, optimizer, iter_num, best_dice, 'pre_train', best_model_path)
                    if args.use_wandb and args.save_wandb_model:
                        try:
                            import wandb
                            artifact = wandb.Artifact(f"{args.exp}_pretrain_best", type="model", description=f"Pre-train best model mDice: {best_dice:.4f}")
                            artifact.add_file(best_model_path)
                            wandb.log_artifact(artifact)
                        except Exception as e:
                            logging.warning(f"[W&B] Failed to log artifact: {e}")
                writer.add_scalar('val/best_mDice', best_dice, iter_num)

                if args.use_wandb:
                    import wandb
                    val_metrics = {
                        'val/mDice': val_mdice,
                        'val/best_mDice': best_dice,
                        'pre_train/val_mDice': val_mdice,
                    }
                    for c, d in enumerate(per_class):
                        val_metrics[f'val/dice_{CLASS_NAMES[c+1]}'] = d
                    wandb.log(val_metrics, step=iter_num)

                # Regularly save latest checkpoint for resuming
                save_checkpoint(model, optimizer, iter_num, best_dice, 'pre_train', latest_pth)
                if args.use_wandb and args.save_wandb_model:
                    try:
                        import wandb
                        art = wandb.Artifact(f"{args.exp}_pretrain_latest", type="checkpoint", description=f"Pre-train latest checkpoint at iter {iter_num}")
                        art.add_file(latest_pth)
                        wandb.log_artifact(art)
                    except Exception as e:
                        logging.warning(f"[W&B] Failed to log latest checkpoint artifact: {e}")
                model.train()

            if iter_num >= args.pre_max_iteration:
                break
        if iter_num >= args.pre_max_iteration:
            break

    # Save final checkpoints
    save_checkpoint(model, optimizer, iter_num, best_dice, 'pre_train', latest_pth)
    best_pth = os.path.join(snapshot_path, f"{args.model}_best_model.pth")
    if not os.path.exists(best_pth):
        save_checkpoint(model, optimizer, iter_num, best_dice, 'pre_train', best_pth)

    writer.close()
    print(f"=== [BHSD Multi-class] PRE-TRAINING FINISHED | Best mDice: {best_dice:.4f} ===")


def self_train(pre_snapshot_path, self_snapshot_path, val_dataset):
    print("=== [BHSD Multi-class] STARTING SELF-TRAINING PHASE (6 Classes) ===")
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
        logging.info(f"[Model] Self-train Multi-GPU enabled with {torch.cuda.device_count()} GPUs (DataParallel)")
    else:
        model = model.cuda()
        logging.info(f"[Model] Self-train Single-GPU enabled on {torch.cuda.get_device_name(0)}")

    db_train = BHSDDataset(
        base_dir=args.root_path,
        split='train',
        binary=False,
        patch_size=patch_size,
        fold=args.fold,
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor()
        ])
    )

    labelnum = min(args.labelnum, len(db_train))
    max_samples = len(db_train) if args.max_samples <= 0 else min(args.max_samples, len(db_train))
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, max_samples))
    sub_bs = int(args.labeled_bs / 2)

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=False)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    model.train()
    writer = SummaryWriter(os.path.join(self_snapshot_path, 'log'))
    iter_num = 0
    best_dice = 0.0

    latest_pth = os.path.join(self_snapshot_path, "checkpoint_latest.pth")
    if args.resume and not os.path.exists(latest_pth) and args.use_wandb:
        logging.info("[RESUME] Local self-train checkpoint not found. Attempting to restore from W&B Cloud...")
        restore_from_wandb(self_snapshot_path, f"{args.exp}_self_latest", args.wandb_project, args.wandb_entity)

    if args.resume and os.path.exists(latest_pth):
        iter_num, best_dice = load_checkpoint(model, optimizer, latest_pth)
        logging.info(f"[RESUME] Resumed self-training from iter {iter_num} | best_mDice: {best_dice:.4f}")
        if iter_num >= args.self_max_iteration:
            logging.info(f"[RESUME] Self-training already completed ({iter_num}/{args.self_max_iteration}). Skipping self-train phase.")
            writer.close()
            return
    else:
        pretrain_candidate = None
        if args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
            pretrain_candidate = args.pretrain_checkpoint
        else:
            best_pth = os.path.join(pre_snapshot_path, f"{args.model}_best_model.pth")
            if not os.path.exists(best_pth) and args.use_wandb:
                logging.info("[RESUME] Attempting to restore pre-train weights from W&B Cloud...")
                restore_from_wandb(pre_snapshot_path, f"{args.exp}_pretrain_best", args.wandb_project, args.wandb_entity)

            if os.path.exists(best_pth):
                pretrain_candidate = best_pth
            elif os.path.exists(os.path.join(pre_snapshot_path, "checkpoint_latest.pth")):
                pretrain_candidate = os.path.join(pre_snapshot_path, "checkpoint_latest.pth")

        if pretrain_candidate and os.path.exists(pretrain_candidate):
            load_net(model, pretrain_candidate)
            logging.info(f"[INFO] Loaded pre-trained weights from {pretrain_candidate}")
        else:
            logging.warning("[WARN] No pre-trained checkpoint found! Starting self-training from scratch.")

    remaining_iters = max(args.self_max_iteration - iter_num, 0)
    max_epoch = remaining_iters // max(len(trainloader), 1) + 2

    for _ in range(max_epoch):
        for _, (sampled_batch, mim_mask) in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()

            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg = volume_batch[args.labeled_bs:]
            laimg = volume_batch[:args.labeled_bs]

            unimg_a, unimg_b = unimg[:sub_bs], unimg[sub_bs:]

            with torch.no_grad():
                _, un_main_outputs, _ = model(unimg)
                un_prob = F.softmax(un_main_outputs, dim=1)
                # Foreground lesion mask for MIM
                fg_un_prob = un_prob[:, 1:, :, :, :].sum(dim=1, keepdim=True)
                un_ps_lab = (fg_un_prob > 0.5).float()

            with torch.no_grad():
                img_mask, _ = context_mask(img_a, args.mask_ratio)

            cutmix_batch_f = img_a * img_mask + unimg_a * (1 - img_mask)
            cutmix_label_f = lab_a * img_mask
            cutmix_batch_b = unimg_b * img_mask + img_b * (1 - img_mask)
            cutmix_label_b = lab_b * (1 - img_mask)
            cutmix_label = torch.cat((cutmix_label_f, cutmix_label_b), dim=0)

            _, main_outputs, _ = model(torch.cat((cutmix_batch_f, cutmix_batch_b), dim=0))
            main_outputs[:sub_bs] = main_outputs[:sub_bs] * img_mask
            main_outputs[sub_bs:] = main_outputs[sub_bs:] * (1 - img_mask)

            mask_region = un_ps_lab
            mim_batch = unimg * (1 - mask_region)
            mim_outputs = model(mim_batch, mode='mim')

            re_batch = unimg * (1 - mask_region) + mim_outputs.detach() * mask_region
            re_batch = torch.flip(re_batch, dims=[2])
            aux_outputs = model(torch.cat((re_batch, laimg), dim=0), mode='aux')

            loss_main_ce = F.cross_entropy(main_outputs, cutmix_label)
            loss_main_dice = DICE(main_outputs, cutmix_label)
            loss_main = (loss_main_ce + loss_main_dice) / 2

            loss_mim = F.l1_loss(unimg, mim_outputs, reduction='none')
            loss_mim = args.lambda_ * (loss_mim * mask_region).sum() / (mask_region.sum() + 1e-5)

            loss_aux_ce = F.cross_entropy(aux_outputs[unimg.shape[0]:], label_batch[:args.labeled_bs])
            loss_aux_dice = DICE(aux_outputs[unimg.shape[0]:], label_batch[:args.labeled_bs])
            loss_aux = (loss_aux_ce + loss_aux_dice) / 2

            loss = loss_main + loss_mim + loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('loss/total', loss.item(), iter_num)
            writer.add_scalar('loss/main', loss_main.item(), iter_num)
            writer.add_scalar('loss/mim', loss_mim.item(), iter_num)
            writer.add_scalar('loss/aux', loss_aux.item(), iter_num)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_num)

            if args.use_wandb:
                import wandb
                global_step = args.pre_max_iteration + iter_num
                wandb.log({
                    'train/loss_total': loss.item(),
                    'train/loss_main': loss_main.item(),
                    'train/loss_mim': loss_mim.item(),
                    'train/loss_aux': loss_aux.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'phase': 'self_train',
                    'self_train/iter': iter_num,
                }, step=global_step)

            if iter_num % 50 == 0:
                logging.info(f"Self-train Iter {iter_num}: Loss: {loss.item():.4f}, Main: {loss_main.item():.4f}, MIM: {loss_mim.item():.4f}")

            if iter_num % args.val_interval == 0:
                val_mdice, per_class = validate_multiclass(model, val_dataset, patch_size=patch_size, stride_xy=args.val_stride_xy)
                class_str = ", ".join([f"{CLASS_NAMES[c+1]}: {d:.3f}" for c, d in enumerate(per_class)])
                logging.info(f"[Self-train Validation] Iter {iter_num} | mDice: {val_mdice:.4f} ({class_str})")
                writer.add_scalar('val/mDice', val_mdice, iter_num)
                for c, d in enumerate(per_class):
                    writer.add_scalar(f'val/dice_{CLASS_NAMES[c+1]}', d, iter_num)
                if val_mdice > best_dice:
                    best_dice = val_mdice
                    best_model_path = os.path.join(self_snapshot_path, f"{args.model}_best_model.pth")
                    save_checkpoint(model, optimizer, iter_num, best_dice, 'self_train', best_model_path)
                    if args.use_wandb and args.save_wandb_model:
                        try:
                            import wandb
                            artifact = wandb.Artifact(f"{args.exp}_best_model", type="model", description=f"Self-train best model mDice: {best_dice:.4f}")
                            artifact.add_file(best_model_path)
                            wandb.log_artifact(artifact)
                        except Exception as e:
                            logging.warning(f"[W&B] Failed to log artifact: {e}")
                writer.add_scalar('val/best_mDice', best_dice, iter_num)

                if args.use_wandb:
                    import wandb
                    global_step = args.pre_max_iteration + iter_num
                    val_metrics = {
                        'val/mDice': val_mdice,
                        'val/best_mDice': best_dice,
                        'self_train/val_mDice': val_mdice,
                    }
                    for c, d in enumerate(per_class):
                        val_metrics[f'val/dice_{CLASS_NAMES[c+1]}'] = d
                    wandb.log(val_metrics, step=global_step)

                # Regularly save latest checkpoint for resuming
                save_checkpoint(model, optimizer, iter_num, best_dice, 'self_train', latest_pth)
                if args.use_wandb and args.save_wandb_model:
                    try:
                        import wandb
                        art = wandb.Artifact(f"{args.exp}_self_latest", type="checkpoint", description=f"Self-train latest checkpoint at iter {iter_num}")
                        art.add_file(latest_pth)
                        wandb.log_artifact(art)
                    except Exception as e:
                        logging.warning(f"[W&B] Failed to log latest checkpoint artifact: {e}")
                model.train()

            if iter_num >= args.self_max_iteration:
                break
        if iter_num >= args.self_max_iteration:
            break

    # Save final checkpoints
    save_checkpoint(model, optimizer, iter_num, best_dice, 'self_train', latest_pth)
    best_pth = os.path.join(self_snapshot_path, f"{args.model}_best_model.pth")
    if not os.path.exists(best_pth):
        save_checkpoint(model, optimizer, iter_num, best_dice, 'self_train', best_pth)

    if args.use_wandb and args.save_wandb_model and os.path.exists(latest_pth):
        try:
            import wandb
            artifact = wandb.Artifact(f"{args.exp}_latest_checkpoint", type="model", description="Latest self-train checkpoint")
            artifact.add_file(latest_pth)
            wandb.log_artifact(artifact)
        except Exception:
            pass

    writer.close()
    print(f"=== [BHSD Multi-class] SELF-TRAINING FINISHED | Best mDice: {best_dice:.4f} ===")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    fold_dir = os.path.join(args.exp_dir, args.exp, f"fold_{args.fold}")
    pre_snapshot = os.path.join(fold_dir, "pre_train")
    self_snapshot = os.path.join(fold_dir, "self_train")
    os.makedirs(pre_snapshot, exist_ok=True)
    os.makedirs(self_snapshot, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(fold_dir, "train.log"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    val_dataset = BHSDDataset(base_dir=args.root_path, split='val', binary=False, patch_size=patch_size, fold=args.fold)

    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name if args.wandb_run_name else f"{args.exp}_fold{args.fold}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                resume="allow",
                id=run_name
            )
            logging.info(f"[W&B] Initialized run '{run_name}' in project '{args.wandb_project}'")
        except Exception as e:
            logging.warning(f"[W&B] Failed to initialize W&B ({e}). Continuing training without W&B.")
            args.use_wandb = False

    if args.phase in ['all', 'pre_train']:
        pre_train(pre_snapshot, val_dataset)
    else:
        logging.info(f"[PHASE] Skipping pre-training phase (--phase={args.phase})")

    if args.phase in ['all', 'self_train']:
        self_train(pre_snapshot, self_snapshot, val_dataset)
    else:
        logging.info(f"[PHASE] Skipping self-training phase (--phase={args.phase})")

    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
