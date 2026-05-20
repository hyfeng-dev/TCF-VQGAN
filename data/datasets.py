import glob
import numpy as np

import torch
import torchvision
import torchvision.transforms.v2 as transforms
import rasterio
import tifffile as tiff

trans_s1 = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    transforms.Lambda(lambda x: torch.cat([
        x[0, :, :].clamp_(-25.0, 0).unsqueeze(0),
        x[1, :, :].clamp_(-32.5, 0).unsqueeze(0)
    ], dim=0)),
    transforms.Lambda(lambda x: torch.stack([
        (x[0] - (-25.0)) / (0.0 - (-25.0)),
        (x[1] - (-32.5)) / (0.0 - (-32.5))
    ], dim=0)),
    transforms.Lambda(lambda x: x * 2. - 1.),
    transforms.Lambda(lambda x: torch.nan_to_num(x, nan=0.0))
])

trans_s2 = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    transforms.Lambda(lambda x: x.clamp_(0, 10000) / 10000),
    transforms.Lambda(lambda x: x * 2. - 1.),
    transforms.Lambda(lambda x: torch.nan_to_num(x, nan=0.0))
])

def random_band_drop(x, drop_prob=0.1):
    if torch.rand(1) < drop_prob:
        band = torch.randint(0, x.shape[0], (1,))
        x[band] = 0
    return x

def random_rot90(x):
    k = torch.randint(0, 4, (1,)).item()
    return torch.rot90(x, k, [1, 2])

general_augmentation = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    # torchvision.transforms.RandomApply([
    #     torchvision.transforms.Lambda(random_rot90)
    # ], p=0.5)
])


def read_tif(path):
    tif = rasterio.open(path)
    return tif.read().astype(np.float32)


class SEN12MS_CR(torch.utils.data.Dataset):
    def __init__(self, path, aug=False, for_codebook=False, for_paper=False, **kwargs):
        super().__init__(**kwargs)
        self.input_s1 = sorted(glob.glob(path + "/input_s1/*.*"))
        self.input_s2 = sorted(glob.glob(path + "/input_s2_cloudy/*.*"))
        self.label = sorted(glob.glob(path + "/labels/*.*"))
        self.lens = len(self.input_s1)
        self.aug = aug
        self.for_codebook = for_codebook
        self.for_paper = for_paper
        if self.for_paper:
            print(" * For paper demo")
        elif self.aug and not self.for_codebook:
            print(" * Augmentation Sets for training")
        elif self.aug and self.for_codebook:
            print(" * Augmentation Sets for codebook")
        elif self.for_codebook:
            print(" * For codebook")
        else:
            print(" * No augmentation")
        
    def __getitem__(self, index):
        input_s1 = trans_s1(read_tif(self.input_s1[index % self.lens]))
        input_s2 = trans_s2(read_tif(self.input_s2[index % self.lens]))
        label = trans_s2(read_tif(self.label[index % self.lens]))
        # cloud_mask = get_cloud_mask(read_tif(self.input_s2[index % self.lens]), cloud_threshold=0.2, binarize=True)
        # cloud_mask = torch.tensor(cloud_mask, dtype=torch.float32).unsqueeze_(0)
        if self.for_paper:
            return {"s1": input_s1, "s2": input_s2, "gt": label}
        
        if self.aug:
            combined = torch.cat([input_s1, input_s2, label], dim=0)
            combined = general_augmentation(combined)
            input_s1, input_s2, label = torch.split(combined, [2, 13, 13], dim=0)

            # combined = torch.cat([input_s1, input_s2], dim=0)
            # combined = random_band_drop(combined, drop_prob=0.5)
            # input_s1, input_s2 = torch.split(combined, [2, 13], dim=0)

        if self.for_codebook:
            return {"lq": torch.cat([input_s1, label], dim=0), "gt": label}
        else:
            return {"lq": torch.cat([input_s1, input_s2], dim=0), "gt": label}
        
    def __len__(self):
        return self.lens


class SMILE_CR(torch.utils.data.Dataset):
    def __init__(self, path, aug=False, for_codebook=False, **kwargs):
        super().__init__(**kwargs)
        self.input_landset = sorted(glob.glob(path + "/CloudLandsat_2020/*.*"))
        self.input_modis = sorted(glob.glob(path + "/MODIS_2020/*.*"))
        self.input_s1 = sorted(glob.glob(path + "/Sentinel-1_2020-De/*.*"))
        self.label = sorted(glob.glob(path + "/Landsat-8_2020/*.*"))
        self.lens = len(self.input_landset)
        self.aug = aug
        self.for_codebook = for_codebook
        
        if self.aug and not self.for_codebook:
            print(" * Augmentation Sets for training")
        elif self.aug and self.for_codebook:
            print(" * Augmentation Sets for codebook")
        elif self.for_codebook:
            print(" * For codebook")
        else:
            print(" * No augmentation")
            
    def __getitem__(self, index):
        input_landset = self._preprocess(tiff.imread(self.input_landset[index]).transpose(1, 2, 0))
        input_modis = self._preprocess(tiff.imread(self.input_modis[index]).transpose(1, 2, 0))
        input_s1 = self._preprocess(tiff.imread(self.input_s1[index]).transpose(1, 2, 0))
        label = self._preprocess(tiff.imread(self.label[index]).transpose(1, 2, 0))
        
        if self.aug:
            combeined = torch.cat([input_landset, input_modis, input_s1, label], dim=0)
            combeined = general_augmentation(combeined)
            input_landset, input_modis, input_s1, label = torch.split(combeined, [6, 6, 2, 6], dim=0)
        
        if self.for_codebook:
            # single modality for codebook
            # return {"lq": torch.cat([label], dim=0), "gt": label}

            # two modalities for codebook
            # return {"lq": torch.cat([input_s1, label], dim=0), "gt": label}
            # return {"lq": torch.cat([input_modis, label], dim=0), "gt": label}

            # trio modalities for codebook
            return {"lq": torch.cat([input_s1, input_modis, label], dim=0), "gt": label}
        else:
            # single modality for codebook
            # return {"lq": torch.cat([input_landset], dim=0), "gt": label}

            # two modalities for codebook
            # return {"lq": torch.cat([input_s1, input_landset], dim=0), "gt": label}
            # return {"lq": torch.cat([input_modis, input_landset], dim=0), "gt": label}

            # trio modalities for codebook
            return {"lq": torch.cat([input_s1, input_modis, input_landset], dim=0), "gt": label}
            
    def __len__(self):
        return self.lens
    
    def _preprocess(self, img):
        return transforms.Compose([
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=False),
            transforms.Lambda(lambda x: torch.nan_to_num(x, nan=0.0)),
            transforms.Lambda(lambda x: x.permute(2, 0, 1)),
            transforms.Lambda(lambda x: x * 2. - 1.),
            transforms.Resize((256, 256))
        ])(img)
    