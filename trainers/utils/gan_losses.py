import torch


def gen_loss(d_real, d_fake, loss_type="wgan", weight=1., **kwargs):
    if loss_type.lower() == "wgan":
        wg_loss_orig = - d_fake.mean()
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_gen_loss": wg_loss.clone().detach().item(),
            "wgan_gen_loss_orig": wg_loss_orig.clone().detach().item(),
        }
    elif loss_type.lower() == "hinge":
        g_loss = -d_fake.mean()
        d_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        d_acc = d_correct / float(d_real.size(0) + d_fake.size(0))

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            "dis_acc": d_acc.clone().detach(),
            "dis_correct": d_correct.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }

    else:
        raise NotImplementedError("Not implement: %s" % loss_type)


def dis_loss(d_real, d_fake, loss_type="wgan", weight=1., **kwargs):
    if loss_type.lower() == "wgan":
        loss_fake = d_fake.mean()
        loss_real = d_real.mean()
        wg_loss_orig = loss_fake - loss_real
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_dis_loss": wg_loss.clone().detach().item(),
            "wgan_dis_loss_orig": wg_loss_orig.clone().detach().item(),
            "wgan_dis_loss_real": loss_real.clone().detach().item(),
            "wgan_dis_loss_fake": loss_fake.clone().detach().item()
        }
    elif loss_type.lower() == "hinge":
        d_loss_real = -torch.min(d_real - 1, d_real * 0).mean()
        d_loss_fake = -torch.min(-d_fake - 1, d_fake * 0).mean()
        d_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        d_acc = d_correct / float(d_real.size(0) + d_fake.size(0))

        d_loss = d_loss_real + d_loss_fake
        loss = d_loss * weight
        return loss, {
            "loss": loss.clone().detach(),
            "d_loss": d_loss.clone().detach(),
            "dis_acc": d_acc.clone().detach(),
            "dis_correct": d_correct.clone().detach(),
            "loss_real": d_loss_real.clone().detach(),
            "loss_fake": d_loss_fake.clone().detach(),
        }
    else:
        raise NotImplementedError("Not implement: %s" % loss_type)


def dis_acc(d_real, d_fake, loss_type="wgan", **kwargs):
    if loss_type.lower() == "wgan":
        # No threshold, don't know which one is correct which is not
        return {}
    elif loss_type.lower() == "hinge":
        return {}
    else:
        raise NotImplementedError("Not implement: %s" % loss_type)


def gradient_penalty(x_real, x_fake, d_real, d_fake,
                     weight=1., gp_type='zero_center', eps=1e-8):
    if gp_type == "zero_center":
        bs = d_real.size(0)
        grad = torch.autograd.grad(
            outputs=d_real, inputs=x_real,
            grad_outputs=torch.ones_like(d_real).to(d_real),
            create_graph=True, retain_graph=True)[0]
        # [grad] should be either (B, D) or (B, #points, D)
        grad = grad.reshape(bs, -1)
        grad_norm = gp_orig = torch.sqrt(torch.sum(grad ** 2, dim=1)).mean()
        gp = gp_orig ** 2. * weight
        return gp, {
            'gp': gp.clone().detach().cpu(),
            'gp_orig': gp_orig.clone().detach().cpu(),
            'grad_norm': grad_norm.clone().detach().cpu()
        }
    else:
        raise NotImplemented("Invalid gp type:%s" % gp_type)
