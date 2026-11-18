"""Stanford Cars held entirely on the GPU, with GPU-side augmentation.

The data order and the augmentation use their own seeded generators, so they are identical
between a baseline run and a reactive run with the same seed, whatever the optimizer does.
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CROP = 224


class CarsData:
    def __init__(self, root: str, device: torch.device, seed: int, batch_size: int):
        split = np.load(os.path.join(root, "cars_split.npz"))
        x = np.load(os.path.join(root, "cars_train_x.npy"), mmap_mode="r")
        y = np.load(os.path.join(root, "cars_train_y.npy"))

        def put(a):
            return torch.from_numpy(np.ascontiguousarray(a)).to(device)

        # uint8 NHWC on the GPU: about 0.2 MB per image
        self.train_x, self.train_y = put(x[split["train_idx"]]), put(y[split["train_idx"]])
        self.val_x, self.val_y = put(x[split["val_idx"]]), put(y[split["val_idx"]])
        self.test_x = put(np.load(os.path.join(root, "cars_test_x.npy")))
        self.test_y = put(np.load(os.path.join(root, "cars_test_y.npy")))

        self.device = device
        self.batch_size = batch_size
        self.updates_per_epoch = len(self.train_x) // batch_size  # last partial batch dropped
        self.num_classes = int(self.train_y.max().item()) + 1
        self._order_gen = torch.Generator(device="cpu").manual_seed(seed)
        self._aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
        self._mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
        self._std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255

    def _normalize(self, x_uint8_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_uint8_nhwc.permute(0, 3, 1, 2).float()
        return ((x - self._mean) / self._std).contiguous(memory_format=torch.channels_last)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random resized crop (scale 0.25-1, ratio 3/4-4/3) and horizontal flip, batched."""
        n, g = x.shape[0], self._aug_gen
        scale = torch.empty(n, device=x.device).uniform_(0.25, 1.0, generator=g)
        log_ratio = torch.empty(n, device=x.device).uniform_(math.log(3 / 4), math.log(4 / 3), generator=g)
        ratio = torch.exp(log_ratio)
        w = torch.sqrt(scale * ratio).clamp(max=1.0)  # crop size as a fraction of the image
        h = torch.sqrt(scale / ratio).clamp(max=1.0)
        cx = (torch.rand(n, device=x.device, generator=g) * 2 - 1) * (1 - w)
        cy = (torch.rand(n, device=x.device, generator=g) * 2 - 1) * (1 - h)
        flip = torch.where(torch.rand(n, device=x.device, generator=g) < 0.5, -1.0, 1.0)
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 0, 2] = w * flip, cx
        theta[:, 1, 1], theta[:, 1, 2] = h, cy
        grid = F.affine_grid(theta, (n, 3, CROP, CROP), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def train_epoch(self):
        """Yield (images, labels) for one epoch; call once per epoch, in order."""
        perm = torch.randperm(len(self.train_x), generator=self._order_gen).to(self.device)
        for i in range(self.updates_per_epoch):
            idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
            yield self._augment(self._normalize(self.train_x[idx])), self.train_y[idx]

    def eval_batches(self, split: str, batch_size: int = 250):
        x, y = (self.val_x, self.val_y) if split == "val" else (self.test_x, self.test_y)
        off = (x.shape[1] - CROP) // 2
        for i in range(0, len(x), batch_size):
            xb = x[i : i + batch_size, off : off + CROP, off : off + CROP]
            yield self._normalize(xb), y[i : i + batch_size]
