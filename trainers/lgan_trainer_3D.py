import os
import tqdm
import torch
import random
import importlib
import numpy as np
from trainers.utils.utils import get_opt
from trainers.ae_trainer_3D import Trainer as BaseTrainer
from trainers.utils.gan_losses import gen_loss, dis_loss, gradient_penalty
from trainers.utils.vis_utils import visualize_procedure, \
    visualize_point_clouds_3d


try:
    from evaluation.evaluation_metrics import compute_all_metrics

    eval_generation = True
except:  # noqa
    eval_generation = False


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        # Now initialize the GAN part
        gen_lib = importlib.import_module(cfg.models.gen.type)
        self.gen = gen_lib.Generator(cfg, cfg.models.gen)
        self.gen.cuda()
        print("Generator:")
        print(self.gen)

        dis_lib = importlib.import_module(cfg.models.dis.type)
        self.dis = dis_lib.Discriminator(cfg, cfg.models.dis)
        self.dis.cuda()
        print("Discriminator:")
        print(self.dis)

        # Optimizers
        if not (hasattr(self.cfg.trainer, "opt_gen") and
                hasattr(self.cfg.trainer, "opt_dis")):
            self.cfg.trainer.opt_gen = self.cfg.trainer.opt
            self.cfg.trainer.opt_dis = self.cfg.trainer.opt
        self.opt_gen, self.scheduler_gen = get_opt(
            self.gen.parameters(), self.cfg.trainer.opt_gen)
        self.opt_dis, self.scheduler_dis = get_opt(
            self.dis.parameters(), self.cfg.trainer.opt_dis)

        # book keeping
        self.total_iters = 0
        self.total_gan_iters = 0
        self.n_critics = getattr(self.cfg.trainer, "n_critics", 1)
        self.gan_only = getattr(self.cfg.trainer, "gan_only", True)

        # If pretrained AE, then load it up
        if hasattr(self.cfg.trainer, "ae_pretrained"):
            ckpt = torch.load(self.cfg.trainer.ae_pretrained)
            print(self.cfg.trainer.ae_pretrained)
            strict = getattr(self.cfg.trainer, "resume_strict", True)
            self.encoder.load_state_dict(ckpt['enc'], strict=strict)
            self.score_net.load_state_dict(ckpt['sn'], strict=strict)
            if getattr(self.cfg.trainer, "resume_opt", False):
                self.opt_enc.load_state_dict(ckpt['opt_enc'])
                self.opt_dec.load_state_dict(ckpt['opt_dec'])
        self.gan_pass_update_enc = getattr(
            self.cfg.trainer, "gan_pass_update_enc", False)

    def epoch_end(self, epoch, writer=None, **kwargs):
        super().epoch_end(epoch, writer=writer)

        if self.scheduler_dis is not None:
            self.scheduler_dis.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dis_lr', self.scheduler_dis.get_lr()[0], epoch)

        if self.scheduler_gen is not None:
            self.scheduler_gen.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_gen_lr', self.scheduler_gen.get_lr()[0], epoch)

    def _update_gan_(self, data, gen=False):
        self.gen.train()
        self.dis.train()
        self.opt_gen.zero_grad()
        self.opt_dis.zero_grad()
        if self.gan_pass_update_enc:
            self.encoder.train()
            self.opt_enc.zero_grad()
        else:
            self.encoder.eval()

        tr_pts = data['tr_points'].cuda()  # (B, #points, 3)smn_ae_trainer.py
        x_real, _ = self.encoder(tr_pts)
        batch_size = x_real.size(0)
        x_fake = self.gen(bs=batch_size)

        x_inp = torch.cat([x_real, x_fake], dim=0)
        d_res = self.dis(x_inp, return_all=True)
        d_out = d_res['x']
        d_real = d_out[:batch_size, ...]
        d_fake = d_out[batch_size:, ...]

        loss, loss_res = None, {}
        loss_type = getattr(self.cfg.trainer, "gan_loss_type", "wgan")
        if gen:
            gen_loss_weight = getattr(self.cfg.trainer, "gen_loss_weight", 1.)
            loss_gen, gen_loss_res = gen_loss(
                d_real, d_fake, weight=gen_loss_weight, loss_type=loss_type)
            loss = loss_gen + (0. if loss is None else loss)
            loss.backward()
            self.opt_gen.step()
            if self.gan_pass_update_enc:
                assert self.opt_enc is not None
                self.opt_enc.step()
            loss_res.update({
                ("train/gan_pass/gen/%s" % k): v
                for k, v in gen_loss_res.items()
            })
            loss_res['loss'] = loss.detach().cpu().item()
        else:
            # Get gradient penalty
            gp_weight = getattr(self.cfg.trainer, 'gp_weight', 0.)
            if gp_weight > 0:
                gp_type = getattr(self.cfg.trainer, 'gp_type', "zero_center")
                gp, gp_res = gradient_penalty(
                    x_real, x_fake, d_real, d_fake,
                    weight=gp_weight, gp_type=gp_type)
                loss = gp + (0. if loss is None else loss)
                loss_res.update({
                    ("train/gan_pass/gp_loss/%s" % k): v
                    for k, v in gp_res.items()
                })

            dis_loss_weight = getattr(self.cfg.trainer, "dis_loss_weight", 1.)
            loss_dis, dis_loss_res = dis_loss(
                d_real, d_fake, weight=dis_loss_weight, loss_type=loss_type)
            loss = loss_dis + (0. if loss is None else loss)
            loss.backward()
            self.opt_dis.step()
            loss_res.update({
                ("train/gan_pass/dis/%s" % k): v for k, v in dis_loss_res.items()
            })
            loss_res['loss'] = loss.detach().cpu().item()

        loss_res['x_real'] = x_real.clone().detach().cpu()
        loss_res['x_fake'] = x_fake.clone().detach().cpu()
        return loss_res

    def update_lgan(self, data):
        self.total_gan_iters += 1
        res = {}
        if self.total_gan_iters % self.n_critics == 0:
            gen_res = self._update_gan_(data, gen=True)
            res.update(gen_res)
        dis_res = self._update_gan_(data, gen=False)
        res.update(dis_res)
        return res

    def update(self, data, *args, **kwargs):
        res = {}
        if not self.gan_only:
            ae_res = super().update(data, *args, **kwargs)
            res.update(ae_res)
        gan_res = self.update_lgan(data)
        res.update(gan_res)
        return res

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        with torch.no_grad():
            train_info.update(super().update(train_data, no_update=True))
        super().log_train(train_info, train_data, writer=writer, step=step,
                          epoch=epoch, visualize=visualize)
        if step is not None:
            writer.add_histogram('tr/latent_real', train_info['x_real'], step)
            writer.add_histogram('tr/latent_fake', train_info['x_fake'], step)
        else:
            assert epoch is not None
            writer.add_histogram('tr/latent_real', train_info['x_real'], epoch)
            writer.add_histogram('tr/latent_fake', train_info['x_fake'], epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize generation results: %s" % step)
                gtr = train_data['te_points']  # ground truth point cloud
                inp = train_data['tr_points']  # input for encoder
                num_vis = min(
                    getattr(self.cfg.viz, "num_vis_samples", 5),
                    gtr.size(0)
                )
                smp, smp_list = self.sample(num_shapes=num_vis,
                                            num_points=inp.size(1))

                all_imgs = []
                for idx in range(num_vis):
                    img = visualize_point_clouds_3d(
                        [smp[idx], gtr[idx]], ["gen", "ref"])
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image(
                    'tr_vis/gen', torch.as_tensor(img), step)

                img = visualize_procedure(
                    self.sigmas, smp_list, gtr, num_vis, self.cfg, "gen")
                writer.add_image(
                    'tr_vis/gen_process', torch.as_tensor(img), step)

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_enc': self.opt_enc.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            'opt_dis': self.opt_dis.state_dict(),
            'opt_gen': self.opt_gen.state_dict(),
            'sn': self.score_net.state_dict(),
            'enc': self.encoder.state_dict(),
            'dis': self.dis.state_dict(),
            'gen': self.gen.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, **args):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.score_net.load_state_dict(ckpt['sn'], strict=strict)
        self.opt_enc.load_state_dict(ckpt['opt_enc'])
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']

        if 'gen' in ckpt:
            self.gen.load_state_dict(ckpt['gen'], strict=strict)
        if 'dis' in ckpt:
            self.dis.load_state_dict(ckpt['dis'], strict=strict)
        if 'opt_gen' in ckpt:
            self.opt_gen.load_state_dict(ckpt['opt_gen'])
        if 'opt_dis' in ckpt:
            self.opt_dis.load_state_dict(ckpt['opt_dis'])
        return start_epoch

    def sample(self, num_shapes=1, num_points=2048):
        with torch.no_grad():
            self.gen.eval()
            z = self.gen(bs=num_shapes)
            return self.langevin_dynamics(z, num_points=num_points)

    def validate(self, test_loader, epoch, *args, **kwargs):
        all_res = {}

        if eval_generation:
            with torch.no_grad():
                print("l-GAN validation:")
                all_ref, all_smp = [], []
                for data in tqdm.tqdm(test_loader):
                    ref_pts = data['te_points'].cuda()
                    inp_pts = data['tr_points'].cuda()
                    smp_pts, _ = self.sample(
                        num_shapes=inp_pts.size(0),
                        num_points=inp_pts.size(1),
                    )
                    all_smp.append(smp_pts.view(
                        ref_pts.size(0), ref_pts.size(1), ref_pts.size(2)))
                    all_ref.append(
                        ref_pts.view(ref_pts.size(0), ref_pts.size(1),
                                     ref_pts.size(2)))

                smp = torch.cat(all_smp, dim=0)
                np.save(
                    os.path.join(self.cfg.save_dir, 'val',
                                 'smp_ep%d.npy' % epoch),
                    smp.detach().cpu().numpy()
                )
                ref = torch.cat(all_ref, dim=0)

                # Sample CD/EMD
                # step 1: subsample shapes
                max_gen_vali_shape = int(getattr(
                    self.cfg.trainer, "max_gen_validate_shapes",
                    int(smp.size(0))))
                sub_sampled = random.sample(
                    range(smp.size(0)), min(smp.size(0), max_gen_vali_shape))
                smp_sub = smp[sub_sampled, ...].contiguous()
                ref_sub = ref[sub_sampled, ...].contiguous()

                gen_res = compute_all_metrics(
                    smp_sub, ref_sub,
                    batch_size=int(getattr(
                        self.cfg.trainer, "val_metrics_batch_size", 100)),
                    accelerated_cd=True
                )
                all_res = {
                    ("val/gen/%s" % k):
                        (v if isinstance(v, float) else v.item())
                    for k, v in gen_res.items()}
                print("Validation Sample (unit) Epoch:%d " % epoch, gen_res)


        # Call super class validation
        if getattr(self.cfg.trainer, "validate_recon", False):
            all_res.update(super().validate(
                test_loader, epoch, *args, **kwargs))

        return all_res