import torch
import pytorch_lightning as pl

from main import instantiate_from_config


class TCFVQGAN(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        base_learning_rate,
        ckpt_path=None,
        ignore_keys=[],
        image_key="lq",
        special_params_lr_scale=1.0,
    ):
        super().__init__()
        self.image_key = image_key
        
        self.vqvae = instantiate_from_config(ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.fix_decoder = ddconfig["params"]["fix_decoder"]
        self.disc_start = lossconfig["params"]["disc_start"]
        self.learning_rate = base_learning_rate
        self.special_params_lr_scale = special_params_lr_scale
        self.automatic_optimization = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        state_dict = self.state_dict()
        require_keys = state_dict.keys()
        keys = sd.keys()
        un_pretrained_keys = []
        for k in require_keys:
            if k not in keys:
                # miss 'vqvae.'
                if k[6:] in keys:
                    state_dict[k] = sd[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                state_dict[k] = sd[k]

        print(f'*************************************************')
        print(f"Layers without pretraining: {un_pretrained_keys}")
        print(f'*************************************************')

        self.load_state_dict(state_dict, strict=True)
        print(f" * Restored from {path}")

    def forward(self, input):
        dec, diff, info, hs = self.vqvae(input)
        return dec, diff, info, hs

    def training_step(self, batch, batch_idx):
        x = batch["lq"]
        y = batch["gt"]
            
        xrec, qloss, info, hs = self(x)
        opt_ae, opt_disc = self.optimizers()
        scheduler_ae, scheduler_disc = self.lr_schedulers()

        aeloss, log_dict_ae = self.loss(qloss, y, xrec, optimizer_idx=0, global_step=self.global_step, split="train")
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        # torch.nn.utils.clip_grad_norm_(self.vqvae.parameters(), 0.1)
        opt_ae.step()
        scheduler_ae.step()

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # discriminator
        discloss, log_dict_disc = self.loss(qloss, y, xrec, optimizer_idx=1, global_step=self.global_step, split="train")
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        # torch.nn.utils.clip_grad_norm_(self.loss.discriminator.parameters(), 1.0)
        opt_disc.step()
        scheduler_disc.step()
        
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return 

    def validation_step(self, batch, batch_idx):
        x = batch["lq"]
        y = batch["gt"]
            
        xrec, qloss, info, hs = self(x)

        aeloss, log_dict_ae = self.loss(qloss, y, xrec, optimizer_idx=0, global_step=self.global_step, split="val")
        discloss, log_dict_disc = self.loss(qloss, y, xrec, optimizer_idx=1, global_step=self.global_step, split="val")
        
        rec_loss = log_dict_ae.pop("val/rec_loss")
        
        self.log(
            "val/rec_loss",
            rec_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return 
    
    def configure_optimizers(self):
        lr = self.learning_rate

        normal_params = []
        special_params = []
        for name, param in self.vqvae.named_parameters():
            if not param.requires_grad:
                continue
            if "decoder" in name and "attn" in name:
                special_params.append(param)
            else:
                normal_params.append(param)
        # print('special_params', special_params)
        if self.loss.dwdwtloss:
            normal_params.append(self.loss.dwdwtloss.weights)
        opt_ae_params = [
            {"params": normal_params, "lr": lr},
            {"params": special_params, "lr": lr * self.special_params_lr_scale},
        ]
        
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_ae = torch.optim.Adam(opt_ae_params, betas=(0.5, 0.9))
        
        scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ae, T_max=self.trainer.max_steps // 2, eta_min=0)
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=self.trainer.max_steps // 2, eta_min=0)

        # scheduler_ae = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_ae, gamma=0.98)
        # scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_disc, gamma=0.98)

        return (
            [opt_ae, opt_disc],
            [
                {"scheduler": scheduler_ae},
                {"scheduler": scheduler_disc},
            ]
        )
