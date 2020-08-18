import torch
import random
import numpy as np
from torch import optim


def get_opt(params, cfgopt):
    if cfgopt.type == 'adam':
        optimizer = optim.Adam(params, lr=float(cfgopt.lr),
                               betas=(cfgopt.beta1, cfgopt.beta2),
                               weight_decay=cfgopt.weight_decay)
    elif cfgopt.type == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=float(cfgopt.lr), momentum=cfgopt.momentum)
    else:
        assert 0, "Optimizer type should be either 'adam' or 'sgd'"

    scheduler = None
    scheduler_type = getattr(cfgopt, "scheduler", None)
    if scheduler_type is not None:
        if scheduler_type == 'exponential':
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
        elif scheduler_type == 'step':
            step_size = int(getattr(cfgopt, "step_epoch", 500))
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay)
        elif scheduler_type == 'linear':
            step_size = int(getattr(cfgopt, "step_epoch", 2000))
            final_ratio = float(getattr(cfgopt, "final_ratio", 0.01))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.5))
            duration_ratio = float(getattr(cfgopt, "duration_ratio", 0.45))

            def lambda_rule(ep):
                lr_l = 1.0 - min(1, max(0, ep - start_ratio * step_size) / float(duration_ratio * step_size)) * (1 - final_ratio)
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif scheduler_type == 'cosine_anneal_nocycle':
            final_lr_ratio = float(getattr(cfgopt, "final_lr_ratio", 0.01))
            eta_min = float(cfgopt.lr) * final_lr_ratio
            eta_max = float(cfgopt.lr)

            total_epoch = int(getattr(cfgopt, "step_epoch", 2000))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.2))
            T_max = total_epoch * (1 - start_ratio)

            def lambda_rule(ep):
                curr_ep = max(0., ep - start_ratio * total_epoch)
                lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * curr_ep / T_max))
                lr_l = lr / eta_max
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear' or 'step'"
    return optimizer, scheduler


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ground_truth_field(prior_points, tr_pts, sigma):
    bs, num_pts = tr_pts.size(0), tr_pts.size(1)
    smp_pts = prior_points.size(1)
    prior_points = prior_points.view(bs, smp_pts, 1, -1)
    tr_pts = tr_pts.view(bs, 1, num_pts, -1)
    dist = (prior_points - tr_pts).norm(dim=3, keepdim=True) ** 2.
    a = - dist / sigma ** 2.
    max_a, _ = torch.max(a, dim=2, keepdim=True)
    diff = torch.exp(a - max_a)
    w_i = diff / diff.sum(dim=2, keepdim=True)

    # (bs, #pts-prior, 1, dim)
    trg_pts = (w_i * tr_pts).sum(dim=2, keepdim=True)
    y = - ((prior_points - trg_pts) / sigma ** 2.).view(bs, smp_pts, -1)
    return y


def ground_truth_reconstruct(inp, sigma, step_size, num_points=2048,
                             num_steps=100, decay=1, interval=10, weight=1):
    with torch.no_grad():
        x = get_prior(inp.size(0), inp.size(1), inp.size(-1)).cuda()
        x_list = []
        x_list.append(x.clone())

        for t in range(num_steps):
            z_t = torch.randn_like(x) * weight
            x += np.sqrt(step_size) * z_t
            grad = ground_truth_field(x, inp, sigma)
            x += 0.5 * step_size * grad
            if t % (num_steps // interval) == 0:
                step_size *= decay
                x_list.append(x.clone())
    return x, x_list


def ground_truth_reconstruct_multi(inp, cfg):
    with torch.no_grad():
        assert hasattr(cfg, "inference")
        step_size_ratio = float(getattr(cfg.inference, "step_size_ratio", 1))
        num_steps = int(getattr(cfg.inference, "num_steps", 5))
        num_points = int(getattr(cfg.inference, "num_points", inp.size(1)))
        weight = float(getattr(cfg.inference, "weight", 1))

        x = get_prior(
            inp.size(0), num_points, cfg.models.scorenet.dim).cuda()
        if hasattr(cfg.trainer, "sigmas"):
            sigmas = cfg.trainer.sigmas
        else:
            sigma_begin = float(cfg.trainer.sigma_begin)
            sigma_end = float(cfg.trainer.sigma_end)
            num_classes = int(cfg.trainer.sigma_num)
            sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                                        num_classes))
        x_list = []
        x_list.append(x.clone())
        bs, num_pts = x.size(0), x.size(1)

        for sigma in sigmas:
            sigma = torch.ones((1,)) * sigma
            sigma = sigma.cuda()
            step_size = 2 * sigma ** 2 * step_size_ratio
            for t in range(num_steps):
                z_t = torch.randn_like(x) * weight
                x += torch.sqrt(step_size) * z_t
                grad = ground_truth_field(x, inp, sigma)
                x += 0.5 * step_size * grad
            x_list.append(x.clone())
    return x, x_list


def get_prior(batch_size, num_points, inp_dim):
    # -1 to 1, uniform
    return (torch.rand(batch_size, num_points, inp_dim) * 2 - 1.) * 1.5


