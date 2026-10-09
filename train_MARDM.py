import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models.AE import AE_models
from models.MARDM import MARDM, MARDM_models
from models.PhysMARD import physmard_ddpm_xl
from utils.evaluators import Evaluators
from utils.datasets import Text2MotionDataset, collate_fn
import time
import copy
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss, update_ema
from utils.eval_utils import evaluation_mardm
from utils.physics_calibration import calibrate_foot_height_error_seeded
import argparse

def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    if args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        dim_pose = 67
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        dim_pose = 64
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    train_split_file = pjoin(data_root, 'train.txt')
    val_split_file = pjoin(data_root, 'val.txt')

    train_dataset = Text2MotionDataset(mean, std, train_split_file, args.dataset_name, motion_dir, text_dir,
                                          args.unit_length, args.max_motion_length, 20, evaluation=False)
    val_dataset = Text2MotionDataset(mean, std, val_split_file, args.dataset_name, motion_dir, text_dir,
                                          args.unit_length, args.max_motion_length, 20, evaluation=False)

    calibration_loader = None
    if args.physics:
        if args.dataset_name != "t2m":
            raise ValueError("Foot calibration currently supports HumanML3D only.")

        generator = torch.Generator().manual_seed(args.seed)
        count = min(512, len(train_dataset))
        indices = torch.randperm(
            len(train_dataset), generator=generator
        )[:count].tolist()

        calibration_loader = DataLoader(
            Subset(train_dataset, indices),
            batch_size=16,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            generator=generator,
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            shuffle=True)

    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    if args.need_evaluation:
        eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
        eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
        split_file = pjoin(data_root, 'val.txt')
        eval_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
                                          4, 196, 20, evaluation=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                                 collate_fn=collate_fn, shuffle=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)

    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model',
                            'latest.tar' if args.dataset_name == 't2m' else 'net_best_fid.tar'), map_location='cpu')
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])
    # Added by JA: keep the pretrained autoencoder fixed.
    ae.requires_grad_(False)
    ae.eval()

    # Select physics-aware training only when requested.
    model = physmard_ddpm_xl if args.physics else MARDM_models[args.model]
    mardm = model(
        ae_dim=ae.output_emb_width, cond_mode='text', use_kl=args.use_kl,
    )
    ema_mardm = copy.deepcopy(mardm)
    ema_mardm.eval()
    for param in ema_mardm.parameters():
        param.requires_grad_(False)

    all_params = 0
    pc_transformer = sum(param.numel() for param in
                         [p for name, p in mardm.named_parameters() if not name.startswith('clip_model.')])
    all_params += pc_transformer
    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.need_evaluation:
        eval_wrapper = Evaluators(args.dataset_name, device=device)
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    logger = SummaryWriter(model_dir)
    ae.eval()
    ae.to(device)
    mardm.to(device)
    ema_mardm.to(device)

    motion_mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    motion_std = torch.as_tensor(std, dtype=torch.float32, device=device)

    foot_height_bias = None
    foot_height_sigma = None

    if args.physics:
        foot_height_bias, foot_height_sigma = calibrate_foot_height_error_seeded(
            ae, calibration_loader, motion_mean, motion_std, seed=args.seed,
        )

    optimizer = optim.AdamW(mardm.parameters(), betas=(0.9, 0.99), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    epoch = 0
    it = 0
    if args.is_continue:
        # Modified by JA: preserve the directory used for subsequent checkpoint saves.
        checkpoint_path = pjoin(model_dir, 'latest.tar')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        missing_keys, unexpected_keys = mardm.load_state_dict(checkpoint['mardm'], strict=False)
        missing_keys2, unexpected_keys2 = ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
        assert len(unexpected_keys) == 0
        assert len(unexpected_keys2) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        assert all([k.startswith('clip_model.') for k in missing_keys2])
        optimizer.load_state_dict(checkpoint['opt_mardm'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # Modified by JA: saved ep is the completed epoch's zero-based index.
        epoch, it = checkpoint['ep'] + 1, checkpoint['total_it']
        del checkpoint  # Added by JA: release loaded checkpoint tensors after restoring state.
        print("Load model epoch:%d iterations:%d" % (epoch, it))

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

    logs = defaultdict(def_value, OrderedDict())

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, clip_score = 1000, 0, 0, 0, 0, 100, -1
    worst_loss = 100

    while epoch < args.epoch:
        ae.eval()
        mardm.train()

        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            m_lens = m_lens.detach().long().to(device)

            # Modified by JA: fixed latent targets need no encoder gradient graph.
            with torch.no_grad():
                latent = ae.encode(motion)
            m_lens = m_lens // 4

            conds = conds.to(device).float() if torch.is_tensor(conds) else conds
            
            if args.physics:
                loss = mardm.forward_loss(
                    latent, conds, m_lens,
                    decoder=ae.decode,
                    reference_motion=motion,
                    mean=motion_mean,
                    std=motion_std,
                    foot_height_bias=foot_height_bias,
                    foot_height_sigma=foot_height_sigma,
                )
            else:
                loss = mardm.forward_loss(latent, conds, m_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logs['loss'] += loss.item()
            logs['lr'] += optimizer.param_groups[0]['lr']
            update_ema(mardm, ema_mardm, 0.9999)

            if it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar('Train/%s' % tag, value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

        save(pjoin(model_dir, 'latest.tar'), epoch, mardm, optimizer, scheduler,
             it, 'mardm', ema_mardm=ema_mardm)
        epoch += 1
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        print('Validation time:')
        ae.eval()
        mardm.eval()
        val_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                conds, motion, m_lens = batch_data
                motion = motion.detach().float().to(device)
                m_lens = m_lens.detach().long().to(device)

                latent = ae.encode(motion)
                m_lens = m_lens // 4

                conds = conds.to(device).float() if torch.is_tensor(conds) else conds

                # Validation reports the data objective without physical observations.
                loss = MARDM.forward_loss(mardm, latent, conds, m_lens)
                val_loss.append(loss.item())

        print(f"Validation loss:{np.mean(val_loss):.3f}")
        logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
        if np.mean(val_loss) < worst_loss:
            print(f"Improved loss from {worst_loss:.02f} to {np.mean(val_loss)}!!!")
            worst_loss = np.mean(val_loss)
        # Modified by JA: full generation metrics only at the requested epoch interval.
        if args.need_evaluation and epoch % args.eval_every == 0:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _, clip_score, writer, save_now= evaluation_mardm(
                model_dir, eval_loader, ema_mardm, ae, logger, epoch-1, best_fid=best_fid, clip_score_old=clip_score,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper, device=device, train_mean=mean, train_std=std)
            if save_now:
                save(pjoin(model_dir, 'net_best_fid.tar'), epoch-1, mardm, optimizer, scheduler,
                     it, 'mardm', ema_mardm=ema_mardm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MARDM')
    parser.add_argument('--ae_name', type=str, default="AE")
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-DDPM-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument("--max_motion_length", type=int, default=196)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[50_000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)

    parser.add_argument('--diffmlps_batch_mul', type=int, default=4)
    parser.add_argument('--need_evaluation', action="store_true" )
    # Added by JA: keep per-epoch validation loss, reduce full generation evaluation.
    parser.add_argument('--eval_every', type=int, default=20, help='Run full motion evaluation every N epochs.')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--log_every', default=50, type=int)

    # Added by JA: omit this flag for MSE; include it for continuous-latent ELBO.
    parser.add_argument('--use_kl', action='store_true', help='Use the DDPM continuous-latent RESCALED_KL objective.')
    parser.add_argument("--physics", action="store_true", help="Enable the predicted-clean physics likelihood.")

    arg = parser.parse_args()
    if arg.eval_every < 1:
        parser.error('--eval_every must be positive')
    main(arg)
