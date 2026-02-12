import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from pathlib import Path
import torch.distributed as dist
from datasets.utils import augment_img


class RealMixedScaleSRDataset(Dataset):
    def __init__(self, 
                 dataroot: str, 
                 scale: list, 
                 augmentation: bool, 
                 test: bool, 
                 camera_types: list[str] = None, 
                 patch_size: int = None, 
                 preload: bool = False,
                 **kwargs):
        super().__init__()
        self.test = test
        self.scale = scale  # Scales to choose LR from, e.g., [2, 3, 4]
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.preload = preload

        dataroot_path = Path(dataroot)
        self.image_list = []  # List to store (camera_type, image_prefix, hr_path)
        split = 'Test' if self.test else 'Train'
        
        first = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

        if camera_types is None or not camera_types:
            # If no camera_types specified, find all subdirectories in dataroot
            camera_types = [d.name for d in dataroot_path.iterdir() if d.is_dir()]
            if not camera_types:
                raise FileNotFoundError(f"No camera type subdirectories found in {dataroot}")

        if first:
            print(f"Loading {split} data from camera types: {camera_types}")

        for camera_type in camera_types:
            camera_path = dataroot_path / camera_type
            split_path = camera_path / split  # Add split path
            if not split_path.is_dir():
                continue

            # Use scale 2 path within the camera/split directory for finding HR images
            hr_base_path = split_path / '2'
            if not hr_base_path.is_dir():
                continue

            # Find HR images using string manipulation
            for hr_file in hr_base_path.glob(f'{camera_type}_*_HR.png'):
                filename_parts = hr_file.stem.split('_')  # Split filename (without extension)
                # Expected format: [CameraType, Index, HR]
                if len(filename_parts) == 3 and filename_parts[0] == camera_type and filename_parts[2] == 'HR' and filename_parts[1].isdigit():
                    image_index = filename_parts[1]
                    image_prefix = f"{camera_type}_{image_index}"  # e.g., Canon_000
                    # Check if corresponding LR images exist for all required scales within the split path
                    lr_paths_exist = True
                    for s in [2, 3, 4]:  # Check standard scales
                        lr_path = split_path / str(s) / f"{image_prefix}_LR{s}.png"
                        if not lr_path.exists():
                            lr_paths_exist = False
                            break
                    if lr_paths_exist:
                        self.image_list.append((camera_type, image_prefix, hr_file))

        self.real_size = len(self.image_list)
        if self.real_size == 0:
            raise FileNotFoundError(f"No valid {split} image sets found for camera types {camera_types} in {dataroot}")
        if first:
            print(f"Found {self.real_size} {split} images.")

        # Preload images to memory if requested
        if self.preload:
            # only print on rank 0
            first = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
            if first:
                print(f"Preloading {split} dataset into memory...")

            self.imgs: dict[str, np.ndarray] = {}
            for camera_type, prefix, hr_file in self.image_list:
                hr_path = str(hr_file)
                self.imgs[hr_path] = np.array(Image.open(hr_path).convert('RGB'))
                split_path = hr_file.parent.parent
                for s in self.scale:
                    if s == 1:
                        # use GT for scale 1
                        continue
                    lr_path = split_path / str(s) / f"{prefix}_LR{s}.png"
                    p = str(lr_path)
                    self.imgs[p] = np.array(Image.open(p).convert('RGB'))

            if first:
                print(f"Preloading {split} dataset done")

    def __len__(self):
        return self.real_size

    def __getitem__(self, index):
        camera_type, image_prefix, gt_img_path = self.image_list[index]
        # gt_img_path already contains the full path including split and scale=2
        split_path = gt_img_path.parent.parent  # Get the split path (e.g., .../Canon/train)
                                    
        random_scale = random.choice(self.scale)

        # load GT
        if self.preload:
            gt = self.imgs[str(gt_img_path)]
        else:
            gt = np.array(Image.open(gt_img_path).convert('RGB'))

        # Load the appropriate image for LQ based on the selected scale
        if random_scale == 1:
            lr = gt.copy() # Use the GT image itself for scale 1
        else:
            lr_path = split_path / str(random_scale) / f'{image_prefix}_LR{random_scale}.png'
            lr = np.array(Image.open(lr_path).convert('RGB')) # Load the selected LR image

        # Apply cropping if patch_size is specified (assuming GT and LR have same dimensions)
        if self.patch_size is not None:
            if not self.test:
                # Random crop for training
                h, w, _ = gt.shape
                # Ensure patch size is not larger than image dimensions
                patch_h = min(self.patch_size, h)
                patch_w = min(self.patch_size, w)
                rnd_h = random.randint(0, max(0, h - patch_h))
                rnd_w = random.randint(0, max(0, w - patch_w))
                gt = gt[rnd_h:rnd_h + patch_h, rnd_w:rnd_w + patch_w, :]
                lr = lr[rnd_h:rnd_h + patch_h, rnd_w:rnd_w + patch_w, :] # Apply same crop to LR
            else:
                # Center crop for testing
                h, w, _ = gt.shape
                # Ensure patch size is not larger than image dimensions
                patch_h = min(self.patch_size, h)
                patch_w = min(self.patch_size, w)
                start_h = max(0, (h - patch_h) // 2)
                start_w = max(0, (w - patch_w) // 2)
                gt = gt[start_h:start_h + patch_h, start_w:start_w + patch_w, :]
                lr = lr[start_h:start_h + patch_h, start_w:start_w + patch_w, :] # Apply same crop to LR

        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            gt = augment_img(gt, mode)
            lr = augment_img(lr, mode)

        img_item = {}
        img_item['GT'] = gt.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
        img_item['LQ'] = lr.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
        img_item['scale'] = random_scale
        img_item['camera'] = camera_type  
        img_item['prefix'] = image_prefix 

        if self.test:
            img_item['img_idx'] = index 
        return img_item


class RealAllScalesSRDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        augmentation: bool,
        test: bool,
        camera_types: list[str] = None,
        patch_size: int = None,
        preload: bool = False,
    ):
        super().__init__()
        self.test = test
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.preload = preload
        self.scales = [2, 3, 4]

        dataroot_path = Path(dataroot)
        self.image_list: list[tuple[str, str, Path]] = []
        split = 'Test' if self.test else 'Train'
        first = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

        if not camera_types:
            camera_types = [d.name for d in dataroot_path.iterdir() if d.is_dir()]
            if not camera_types:
                raise FileNotFoundError(f"No camera type subdirectories found in {dataroot}")
        if first:
            print(f"Loading {split} data from camera types: {camera_types}")

        for camera_type in camera_types:
            split_path = dataroot_path / camera_type / split
            hr_base = split_path / '2'
            if not hr_base.is_dir():
                continue

            for hr_file in hr_base.glob(f'{camera_type}_*_HR.png'):
                parts = hr_file.stem.split('_')
                if len(parts) == 3 and parts[0] == camera_type and parts[2] == 'HR' and parts[1].isdigit():
                    prefix = f"{camera_type}_{parts[1]}"
                    if all((split_path / str(s) / f"{prefix}_LR{s}.png").exists() for s in self.scales):
                        self.image_list.append((camera_type, prefix, hr_file))

        self.real_size = len(self.image_list)
        if self.real_size == 0:
            raise FileNotFoundError(
                f"No complete {split} image sets found for camera types {camera_types} in {dataroot}"
            )
        
        if first:
            print(f"Found {self.real_size} {split} images.")

        if self.preload:
            if first:
                print(f"Preloading {split} dataset into memory...")

            self.imgs: dict[str, np.ndarray] = {}
            for _, prefix, hr_file in self.image_list:
                hr_path = str(hr_file)
                self.imgs[hr_path] = np.array(Image.open(hr_path).convert('RGB'))

                split_path = hr_file.parent.parent
                for s in self.scales:
                    lr_path = split_path / str(s) / f"{prefix}_LR{s}.png"
                    self.imgs[str(lr_path)] = np.array(Image.open(lr_path).convert('RGB'))

            if first:
                print(f"Preloading {split} dataset done")

    def __len__(self):
        return self.real_size

    def __getitem__(self, index):
        camera_type, prefix, hr_file = self.image_list[index]
        split_path = hr_file.parent.parent

        paths = {
            'GT': hr_file,
            2: split_path / '2' / f"{prefix}_LR2.png",
            3: split_path / '3' / f"{prefix}_LR3.png",
            4: split_path / '4' / f"{prefix}_LR4.png",
        }

        if self.preload:
            imgs = {k: self.imgs[str(p)] for k, p in paths.items()}
        else:
            imgs = {k: np.array(Image.open(p).convert('RGB')) for k, p in paths.items()}

        gt = imgs['GT']
        lr2 = imgs[2]
        lr3 = imgs[3]
        lr4 = imgs[4]

        if self.patch_size is not None:
            h, w, _ = gt.shape
            ph = min(self.patch_size, h)
            pw = min(self.patch_size, w)
            if not self.test:
                rh = random.randint(0, h - ph)
                rw = random.randint(0, w - pw)
            else:
                rh = (h - ph) // 2
                rw = (w - pw) // 2

            gt = gt[rh:rh+ph, rw:rw+pw]
            lr2 = lr2[rh:rh+ph, rw:rw+pw]
            lr3 = lr3[rh:rh+ph, rw:rw+pw]
            lr4 = lr4[rh:rh+ph, rw:rw+pw]
        else:
            gt = shave_on_four(gt)
            lr2 = shave_on_four(lr2)
            lr3 = shave_on_four(lr3)
            lr4 = shave_on_four(lr4)

        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            gt = augment_img(gt, mode)
            lr2 = augment_img(lr2, mode)
            lr3 = augment_img(lr3, mode)
            lr4 = augment_img(lr4, mode)

        item = {
            'GT':  gt.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.,
            'LR2': lr2.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.,
            'LR3': lr3.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.,
            'LR4': lr4.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.,
            'camera': camera_type,
            'prefix': prefix,
        }
        if self.test:
            item['img_idx'] = index

        return item



class DIV2KDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        augmentation: bool,
        test: bool,
        patch_size: int = None,
        preload: bool = False,
    ):
        super().__init__()
        self.test = test
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.preload = preload

        split = 'Test' if self.test else 'Train'
        dataroot_path = Path(dataroot)
        self.split_path = dataroot_path / split

        # Only rank 0 prints
        first = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
        if first:
            print(f"Loading {split} data from {self.split_path}")

        # Gather all .png files like 0001.png ... 0800.png
        self.image_list = sorted(self.split_path.glob('*.png'))
        self.real_size = len(self.image_list)
        if self.real_size == 0:
            raise FileNotFoundError(f"No images found in {self.split_path}")
        if first:
            print(f"Found {self.real_size} images.")

        # Preload into memory if requested
        if self.preload:
            if first:
                print(f"Preloading {split} dataset into memory...")
            self.imgs: dict[str, np.ndarray] = {}
            for hr_file in self.image_list:
                self.imgs[str(hr_file)] = np.array(Image.open(hr_file).convert('RGB'))
            if first:
                print("Preloading done.")

    def __len__(self):
        return self.real_size

    def __getitem__(self, index):
        hr_file = self.image_list[index]
        # Load image (from memory if preloaded)
        if self.preload:
            hr = self.imgs[str(hr_file)]
        else:
            hr = np.array(Image.open(hr_file).convert('RGB'))

        # Random or center crop to patch_size
        if self.patch_size is not None:
            h, w, _ = hr.shape
            ph = min(self.patch_size, h)
            pw = min(self.patch_size, w)
            if not self.test:
                rh = random.randint(0, h - ph)
                rw = random.randint(0, w - pw)
            else:
                rh = (h - ph) // 2
                rw = (w - pw) // 2
            hr = hr[rh:rh+ph, rw:rw+pw]

        # Data augmentation
        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            hr = augment_img(hr, mode)

        # Make dimensions multiples of 4
        hr = shave_on_four(hr)

        item = {
            'GT': hr.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0,
            'prefix': hr_file.stem,
        }
        if self.test:
            item['img_idx'] = index

        return item


class DRealSRDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        augmentation: bool,
        test: bool,
        patch_size: int = None,
        preload: bool = False,
    ):
        super().__init__()
        self.test = test
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.preload = preload

        split = 'test' if self.test else 'train_HR'
        dataroot_path = Path(dataroot)
        self.split_path = dataroot_path / split

        # Only rank 0 prints
        first = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
        if first:
            print(f"Loading {split} data from {self.split_path}")

        # Gather all .png files like 0001.png ... 0800.png
        self.image_list = sorted(self.split_path.glob('*.png'))
        self.real_size = len(self.image_list)
        if self.real_size == 0:
            raise FileNotFoundError(f"No images found in {self.split_path}")
        if first:
            print(f"Found {self.real_size} images.")

        # Preload into memory if requested
        if self.preload:
            if first:
                print(f"Preloading {split} dataset into memory...")
            self.imgs: dict[str, np.ndarray] = {}
            for hr_file in self.image_list:
                self.imgs[str(hr_file)] = np.array(Image.open(hr_file).convert('RGB'))
            if first:
                print("Preloading done.")

    def __len__(self):
        return self.real_size

    def __getitem__(self, index):
        hr_file = self.image_list[index]
        # Load image (from memory if preloaded)
        if self.preload:
            hr = self.imgs[str(hr_file)]
        else:
            hr = np.array(Image.open(hr_file).convert('RGB'))

        # Random or center crop to patch_size
        if self.patch_size is not None:
            h, w, _ = hr.shape
            ph = min(self.patch_size, h)
            pw = min(self.patch_size, w)
            if not self.test:
                rh = random.randint(0, h - ph)
                rw = random.randint(0, w - pw)
            else:
                rh = (h - ph) // 2
                rw = (w - pw) // 2
            hr = hr[rh:rh+ph, rw:rw+pw]

        # Data augmentation
        if not self.test and self.augmentation:
            mode = random.randint(0, 7)
            hr = augment_img(hr, mode)

        # Make dimensions multiples of 4
        hr = shave_on_four(hr)

        item = {
            'GT': hr.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0,
            'prefix': hr_file.stem,
        }
        if self.test:
            item['img_idx'] = index

        return item

class augmentation(object):
    def __call__(self, *inputs):

        hor_flip = random.randrange(0,2)
        ver_flip = random.randrange(0,2)
        rot = random.randrange(0,2)

        output_list = []
        for inp in inputs:
            if hor_flip:
                tmp_inp = np.fliplr(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if ver_flip:
                tmp_inp = np.flipud(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if rot:
                inp = inp.transpose(1, 0, 2)
            output_list.append(inp)

        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]

class crop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, *inputs):
        ih, iw = inputs[0].shape[:2]
        try:
            ix = random.randrange(0, iw - self.patch_size +1)
            iy = random.randrange(0, ih - self.patch_size +1)
        except(ValueError):
            print('>> patch size: {}'.format(self.patch_size))
            print('>> ih, iw: {}, {}'.format(ih, iw))
            exit()

        output_list = [] 
        for inp in inputs:
            output_list.append(inp[iy : iy + self.patch_size, ix : ix + self.patch_size])
        
        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]

def shave_on_four(img):
    shave = 64

    h, w, _ = img.shape
    if(h % shave != 0):
        img = img[:-(h%shave), :, :]
    if(w % shave != 0):
        img = img[:, :-(w%shave), :]
    return img


def limit_size(img, size_limit):
    h, w, _ = img.shape
    if(h > size_limit):
        img = img[:size_limit, :, :]
    if(w > size_limit):
        img = img[:, :size_limit, :]
    return img