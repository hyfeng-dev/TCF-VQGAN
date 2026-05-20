import torch
import torch.nn as nn
import torch.nn.functional as F

from models.discriminator import NLayerDiscriminator, weights_init
from pytorch_msssim import ssim, ms_ssim
from pytorch_wavelets import DWTForward

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

class DynamicallyWeightedMultiScaleWaveletLoss(nn.Module):
    def __init__(self, J, mode, wave):
        super().__init__()
        self.xfm = DWTForward(J=J, mode=mode, wave=wave)
        self.register_buffer('init_weights', torch.ones(J))
        self.weights = nn.Parameter(self.init_weights.clone(), requires_grad=True)
        
    def forward(self, x, y):
        weights = F.softmax(self.weights, dim=0)
        
        _, x_Yhs = self.xfm(x)
        _, y_Yhs = self.xfm(y)
        
        loss = torch.tensor(0.)
        for i in range(len(x_Yhs)):
            loss = loss + weights[i] * F.l1_loss(x_Yhs[i], y_Yhs[i], reduction='mean')
        
        return loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    # return 0.
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class L1_SSIM_DW2A_Discriminator(nn.Module):
    def __init__(
        self, 
        disc_start, 
        codebook_weight=1.0, 
        pixelloss_weight=1.0,
        disc_num_layers=3, 
        disc_in_channels=3, 
        disc_factor=1.0, 
        disc_weight=1.0, 
        use_actnorm=False, 
        disc_conditional=False,
        disc_ndf=64, 
        addition_loss=["ssim"], 
        addition_loss_weight=0.1, 
        disc_loss="hinge"
        ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.addition_loss = addition_loss
        self.ssim_loss = None
        self.dwdwtloss = None
        
        if "ssim" in addition_loss:
            self.ssim_loss = ms_ssim
        
        if "2dwt" in addition_loss:
            self.dwdwtloss = DynamicallyWeightedMultiScaleWaveletLoss(J=3, mode="symmetric", wave="db4")
        self.addition_loss_weight = addition_loss_weight
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f" * TCF-VQ GAN running with {disc_loss} loss.")

        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional


    def forward(
        self, 
        codebook_loss, 
        inputs, 
        reconstructions, 
        optimizer_idx,
        global_step, 
        cond=None, 
        split="train"):
        # reconstruction loss
        rec_loss = self.pixel_weight * torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = torch.mean(rec_loss)

        ssim_loss = torch.tensor(0.0)
        dwdwt_loss = torch.tensor(0.0)
        
        if self.ssim_loss:
            # ms ssim loss
            ssim_loss = self.addition_loss_weight[self.addition_loss.index("ssim")] * \
                (1 - self.ssim_loss(inputs.contiguous(), reconstructions.contiguous(), data_range=1.0, size_average=True))

        if self.dwdwtloss:
            # 2dwt loss
            dwdwt_loss = self.addition_loss_weight[self.addition_loss.index("2dwt")] * self.dwdwtloss(inputs.contiguous(), reconstructions.contiguous())

        nll_loss = rec_loss + ssim_loss + dwdwt_loss

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
            
            d_weight = 0.0001

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/ssim_loss".format(split): ssim_loss.detach().mean(),
                   "{}/2dwt_loss".format(split): dwdwt_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight,
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }

            if self.dwdwtloss:
                for i, w in enumerate(self.dwdwtloss.weights.detach().cpu().numpy()):
                    log["{}/2dwt_weight_level_{}".format(split, i)] = torch.tensor(w)
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
        
