import torch
import random

from tqdm import tqdm

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import SpectralAngleMapper
from matplotlib import pyplot as plt

def sam(x, y, eps=1e-8):
    """
    Spectral Angle Mapper (SAM) between two tensors.
    Args:
        x (torch.Tensor): shape (..., C) where C is the spectral dimension.
        y (torch.Tensor): same shape as x.
        eps (float): small value to avoid division by zero.
    Returns:
        torch.Tensor: SAM in radians.
    """
    # Compute dot product and norms
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    
    # Compute cosine similarity and clip to avoid numerical issues
    cosine_sim = dot_product / (norm_x * norm_y + eps)
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)  # Ensure valid arccos input
    
    # Compute SAM (in radians)
    sam_rad = torch.acos(cosine_sim)
    return sam_rad


def calculate_metrics(model, data_iter, device, is_train=False):
    """
    Calculate both SSIM and PSNR for a given model and dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_iter (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the computation on.
        is_train (bool): Whether the dataset is training or validation.

    Returns:
        tuple: Average SSIM and PSNR values.
    """
    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    mae_metric = MeanAbsoluteError().to(device)
    sam_metric = 0

    print(f" * Calculating SSIM and PSNR for {'train' if is_train else 'val'} ...")
    
    total = len(data_iter)
    if is_train:
        total = total * 0.1
    for i, data in enumerate(tqdm(data_iter, desc="Metrics Progress", unit="batch")):
        # x, y = data["lq"], data["gt"]
        x = data["lq"]
        y = data["gt"]
        
        x, y = x.to(device), y.to(device)
        y_hat, diff, info, hs = model(x)

        y_hat = y_hat * 0.5 + 0.5
        y = y * 0.5 + 0.5
        
        y_hat = torch.clamp(y_hat, 0, 1)
        y = torch.clamp(y, 0, 1)

        ssim_metric.update(y_hat, y)
        psnr_metric.update(y_hat, y)
        mae_metric.update(y_hat, y)   
        sam_metric += sam(y_hat.permute(0, 2, 3, 1), y.permute(0, 2, 3, 1)).mean()

        if is_train and i > total:
            break

    avg_ssim = ssim_metric.compute().item()
    avg_psnr = psnr_metric.compute().item()
    avg_mae = mae_metric.compute().item()
    avg_sam = ((sam_metric / total) * (180.0 / torch.pi)).item()
    
    print(f" * Average SSIM: {avg_ssim:.4f}")
    print(f" * Average PSNR: {avg_psnr:.4f} dB")
    print(f" * Average MAE: {avg_mae:.6f}")
    print(f" * Average SAM: {avg_sam:.6f}")
    
    model.train()
    return avg_ssim, avg_psnr, avg_mae, avg_sam

def save_result_sen12(model, data, dir, global_step):
    model.eval()
    
    sample = random.choice(data)
    x = sample["lq"].to("cuda").unsqueeze(0)
    y = sample["gt"].to("cuda").unsqueeze(0)
    y_hat = model.forward(x)[0]

    x = x * 0.5 + 0.5
    y_hat = y_hat * 0.5 + 0.5
    y = y * 0.5 + 0.5
    
    x = x.clamp(0, 1)
    y_hat = y_hat.clamp(0, 1)
    y = y.clamp(0, 1)
    figure, axes = plt.subplots(3, 14, figsize=(20, 5), gridspec_kw={'wspace':0.1, 'hspace':0.1})

    for i, (xi, yi, y_hati, ax) in enumerate(zip(x[0][2:], y[0], y_hat[0], axes.transpose())):
        ax[0].imshow(xi.detach().cpu(), cmap="gray")
        ax[0].axis("off")
        ax[1].imshow(yi.detach().cpu(), cmap="gray")
        ax[1].axis("off")
        ax[2].imshow(y_hati.detach().cpu(), cmap="gray")
        ax[2].axis("off")

    axes[0][-1].imshow(x[0][3:6, :, :].detach().cpu().permute(1, 2, 0))
    axes[0][-1].axis("off")
    axes[1][-1].imshow(y[0][1:4, :, :].detach().cpu().permute(1, 2, 0))
    axes[1][-1].axis("off")
    axes[2][-1].imshow(y_hat[0][1:4, :, :].detach().cpu().permute(1, 2, 0))
    axes[2][-1].axis("off")

    figure.savefig(f"{dir}/test_{global_step}.png", dpi=300, bbox_inches="tight")
    model.train()


def save_result_smile(model, data, dir, global_step):
    sample = random.choice(data)
    x = sample["lq"].to("cuda").unsqueeze(0)
    y = sample["gt"].to("cuda").unsqueeze(0)
    y_hat = model.forward(x)[0]
    y_hat = y_hat.clamp(0, 1)
    figure, axes = plt.subplots(3, 7, figsize=(20, 5), gridspec_kw={'wspace':0.1, 'hspace':0.1})

    for i, (xi, yi, y_hati, ax) in enumerate(zip(x[0], y[0], y_hat[0], axes.transpose())):
        ax[0].imshow(xi.detach().cpu(), cmap="gray")
        ax[0].axis("off")
        ax[1].imshow(yi.detach().cpu(), cmap="gray")
        ax[1].axis("off")
        ax[2].imshow(y_hati.detach().cpu(), cmap="gray")
        ax[2].axis("off")

    axes[0][-1].imshow(x[0][[3, 2, 1], :, :].detach().cpu().permute(1, 2, 0))
    axes[0][-1].axis("off")
    axes[1][-1].imshow(y[0][[3, 2, 1], :, :].detach().cpu().permute(1, 2, 0))
    axes[1][-1].axis("off")
    axes[2][-1].imshow(y_hat[0][[3, 2, 1], :, :].detach().cpu().permute(1, 2, 0))
    axes[2][-1].axis("off")

    figure.savefig(f"{dir}/test_{global_step}.png", dpi=300, bbox_inches="tight")
