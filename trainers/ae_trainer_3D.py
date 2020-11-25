import os
import tqdm
import torch
import importlib
import numpy as np
from trainers.base_trainer import BaseTrainer
from trainers.utils.vis_utils import visualize_point_clouds_3d, visualize_procedure
from trainers.utils.utils import (
    get_opt,
    get_prior,
    ground_truth_reconstruct_multi,
    set_random_seed,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from evaluation.evaluation_metrics import EMD_CD

    eval_reconstruciton = True
except:  # noqa
    # Skip evaluation
    eval_reconstruciton = False


def score_matching_loss(score_net, shape_latent, tr_pts, sigma):
    bs, num_pts = tr_pts.size(0), tr_pts.size(1)
    sigma = sigma.view(bs, 1, 1)
    perturbed_points = tr_pts + torch.randn_like(tr_pts) * sigma

    # For numerical stability, the network predicts the field in a normalized
    # scale (i.e. the norm of the gradient is not scaled by `sigma`)
    # As a result, when computing the ground truth for supervision, we are using
    # its original scale without scaling by `sigma`
    y_pred = score_net(perturbed_points, shape_latent)  # field (B, #points, 3)
    y_gtr = -(perturbed_points - tr_pts).view(bs, num_pts, -1)

    # The loss for each sigma is weighted
    lambda_sigma = 1.0 / sigma
    loss = 0.5 * ((y_gtr - y_pred) ** 2.0 * lambda_sigma).sum(dim=2).mean()
    return {"loss": loss, "x": perturbed_points}


class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        sn_lib = importlib.import_module(cfg.models.scorenet.type)
        self.score_net = sn_lib.Decoder(cfg, cfg.models.scorenet)
        self.score_net.to(DEVICE)
        print("ScoreNet:")
        print(self.score_net)

        encoder_lib = importlib.import_module(cfg.models.encoder.type)
        self.encoder = encoder_lib.Encoder(cfg.models.encoder)
        self.encoder.to(DEVICE)
        print("Encoder:")
        print(self.encoder)

        # The optimizer
        if not (
            hasattr(self.cfg.trainer, "opt_enc")
            and hasattr(self.cfg.trainer, "opt_dec")
        ):
            self.cfg.trainer.opt_enc = self.cfg.trainer.opt
            self.cfg.trainer.opt_dec = self.cfg.trainer.opt

        self.opt_enc, self.scheduler_enc = get_opt(
            self.encoder.parameters(), self.cfg.trainer.opt_enc
        )
        self.opt_dec, self.scheduler_dec = get_opt(
            self.score_net.parameters(), self.cfg.trainer.opt_dec
        )

        # Sigmas
        if hasattr(cfg.trainer, "sigmas"):
            self.sigmas = cfg.trainer.sigmas
        else:
            self.sigma_begin = float(cfg.trainer.sigma_begin)
            self.sigma_end = float(cfg.trainer.sigma_end)
            self.num_classes = int(cfg.trainer.sigma_num)
            self.sigmas = np.exp(
                np.linspace(
                    np.log(self.sigma_begin), np.log(self.sigma_end), self.num_classes
                )
            )
        print("Sigma:, ", self.sigmas)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)

        # Prepare variable for summy
        self.oracle_res = None

    def multi_gpu_wrapper(self, wrapper):
        self.encoder = wrapper(self.encoder)
        self.score_net = wrapper(self.score_net)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.scheduler_dec is not None:
            self.scheduler_dec.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    "train/opt_dec_lr", self.scheduler_dec.get_lr()[0], epoch
                )
        if self.scheduler_enc is not None:
            self.scheduler_enc.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    "train/opt_enc_lr", self.scheduler_enc.get_lr()[0], epoch
                )

    def update(self, data, *args, **kwargs):
        if "no_update" in kwargs:
            no_update = kwargs["no_update"]
        else:
            no_update = False
        if not no_update:
            self.encoder.train()
            self.score_net.train()
            self.opt_enc.zero_grad()
            self.opt_dec.zero_grad()

        tr_pts = data["tr_points"].to(DEVICE)  # (B, #points, 3)smn_ae_trainer.py
        batch_size = tr_pts.size(0)
        z_mu, z_sigma = self.encoder(tr_pts)
        z = z_mu + 0 * z_sigma

        # Randomly sample sigma
        labels = torch.randint(0, len(self.sigmas), (batch_size,), device=tr_pts.device)
        used_sigmas = (
            torch.tensor(np.array(self.sigmas))[labels]
            .float()
            .view(batch_size, 1)
            .to(DEVICE)
        )
        z = torch.cat((z, used_sigmas), dim=1)

        res = score_matching_loss(self.score_net, z, tr_pts, used_sigmas)
        loss = res["loss"]
        if not no_update:
            loss.backward()
            self.opt_enc.step()
            self.opt_dec.step()

        return {
            "loss": loss.detach().cpu().item(),
            "x": res["x"].detach().cpu(),  # perturbed data
        }

    def log_train(
        self,
        train_info,
        train_data,
        writer=None,
        step=None,
        epoch=None,
        visualize=False,
        **kwargs
    ):
        if writer is None:
            return

        # Log training information to tensorboard
        train_info = {
            k: (v.cpu() if not isinstance(v, float) else v)
            for k, v in train_info.items()
        }
        for k, v in train_info.items():
            if not ("loss" in k):
                continue
            if step is not None:
                writer.add_scalar("train/" + k, v, step)
            else:
                assert epoch is not None
                writer.add_scalar("train/" + k, v, epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize: %s" % step)
                gtr = train_data["te_points"]  # ground truth point cloud
                inp = train_data["tr_points"]  # input for encoder
                ptb = train_info["x"]  # perturbed data
                num_vis = min(getattr(self.cfg.viz, "num_vis_samples", 5), gtr.size(0))

                print("Recon:")
                rec, rec_list = self.reconstruct(
                    inp[:num_vis].to(DEVICE), num_points=inp.size(1)
                )
                print("Ground truth recon:")
                rec_gt, rec_gt_list = ground_truth_reconstruct_multi(
                    inp[:num_vis].to(DEVICE), self.cfg
                )
                # Overview
                all_imgs = []
                for idx in range(num_vis):
                    img = visualize_point_clouds_3d(
                        [rec_gt[idx], rec[idx], gtr[idx], ptb[idx]],
                        ["rec_gt", "recon", "shape", "perturbed"],
                    )
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image("tr_vis/overview", torch.as_tensor(img), step)

                # Reconstruction gt procedure
                img = visualize_procedure(
                    self.sigmas, rec_gt_list, gtr, num_vis, self.cfg, "Rec_gt"
                )
                writer.add_image("tr_vis/rec_gt_process", torch.as_tensor(img), step)

                # Reconstruction procedure
                img = visualize_procedure(
                    self.sigmas, rec_list, gtr, num_vis, self.cfg, "Rec"
                )
                writer.add_image("tr_vis/rec_process", torch.as_tensor(img), step)

    def validate(self, test_loader, epoch, *args, **kwargs):
        if not eval_reconstruciton:
            return {}

        print("Validation (reconstruction):")
        all_ref, all_rec, all_smp, all_ref_denorm = [], [], [], []
        all_rec_gt, all_inp_denorm, all_inp = [], [], []
        for data in tqdm.tqdm(test_loader):
            ref_pts = data["te_points"].to(DEVICE)
            inp_pts = data["tr_points"].to(DEVICE)
            m = data["mean"].to(DEVICE)
            std = data["std"].to(DEVICE)
            rec_pts, _ = self.reconstruct(inp_pts, num_points=inp_pts.size(1))

            # denormalize
            inp_pts_denorm = inp_pts.clone() * std + m
            ref_pts_denorm = ref_pts.clone() * std + m
            rec_pts = rec_pts * std + m

            all_inp.append(inp_pts)
            all_inp_denorm.append(inp_pts_denorm.view(*inp_pts.size()))
            all_ref_denorm.append(ref_pts_denorm.view(*ref_pts.size()))
            all_rec.append(rec_pts.view(*ref_pts.size()))
            all_ref.append(ref_pts)

        inp = torch.cat(all_inp, dim=0)
        rec = torch.cat(all_rec, dim=0)
        ref = torch.cat(all_ref, dim=0)
        ref_denorm = torch.cat(all_ref_denorm, dim=0)
        inp_denorm = torch.cat(all_inp_denorm, dim=0)
        for name, arr in [
            ("inp", inp),
            ("rec", rec),
            ("ref", ref),
            ("ref_denorm", ref_denorm),
            ("inp_denorm", inp_denorm),
        ]:
            np.save(
                os.path.join(self.cfg.save_dir, "val", "%s_ep%d.npy" % (name, epoch)),
                arr.detach().cpu().numpy(),
            )
        all_res = {}

        # Oracle CD/EMD, will compute only once
        if self.oracle_res is None:
            rec_res = EMD_CD(inp_denorm, ref_denorm, 1)
            rec_res = {
                ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
                for k, v in rec_res.items()
            }
            all_res.update(rec_res)
            print("Validation oracle (denormalize) Epoch:%d " % epoch, rec_res)
            self.oracle_res = rec_res
        else:
            all_res.update(self.oracle_res)

        # Reconstruction CD/EMD
        all_res = {}
        rec_res = EMD_CD(rec, ref_denorm, 1)
        rec_res = {
            ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
            for k, v in rec_res.items()
        }
        all_res.update(rec_res)
        print("Validation Recon (denormalize) Epoch:%d " % epoch, rec_res)

        return all_res

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            "opt_enc": self.opt_enc.state_dict(),
            "opt_dec": self.opt_dec.state_dict(),
            "sn": self.score_net.state_dict(),
            "enc": self.encoder.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path, map_location=torch.device(DEVICE))
        self.encoder.load_state_dict(ckpt["enc"], strict=strict)
        self.score_net.load_state_dict(ckpt["sn"], strict=strict)
        self.opt_enc.load_state_dict(ckpt["opt_enc"])
        self.opt_dec.load_state_dict(ckpt["opt_dec"])
        start_epoch = ckpt["epoch"]
        return start_epoch

    def langevin_dynamics(self, z, num_points=2048):
        with torch.no_grad():
            assert hasattr(self.cfg, "inference")
            step_size_ratio = float(getattr(self.cfg.inference, "step_size_ratio", 1))
            num_steps = int(getattr(self.cfg.inference, "num_steps", 5))
            num_points = int(getattr(self.cfg.inference, "num_points", num_points))
            weight = float(getattr(self.cfg.inference, "weight", 1))
            sigmas = self.sigmas

            x_list = []
            self.score_net.eval()
            x = get_prior(z.size(0), num_points, self.cfg.models.scorenet.dim)
            x = x.to(z)
            x_list.append(x.clone())
            for sigma in sigmas:
                sigma = torch.ones((1,)).to(DEVICE) * sigma
                z_sigma = torch.cat((z, sigma.expand(z.size(0), 1)), dim=1)
                step_size = 2 * sigma ** 2 * step_size_ratio
                for t in range(num_steps):
                    z_t = torch.randn_like(x) * weight
                    x += torch.sqrt(step_size) * z_t
                    grad = self.score_net(x, z_sigma)
                    grad = grad / sigma ** 2
                    x += 0.5 * step_size * grad
                x_list.append(x.clone())
        return x, x_list

    def sample(self, num_shapes=1, num_points=2048):
        with torch.no_grad():
            z = torch.randn(num_shapes, self.cfg.models.encoder.zdim).to(DEVICE)
            return self.langevin_dynamics(z, num_points=num_points)

    def reconstruct(self, inp, num_points=2048):
        with torch.no_grad():
            self.encoder.eval()
            z, _ = self.encoder(inp)
            return self.langevin_dynamics(z, num_points=num_points)

