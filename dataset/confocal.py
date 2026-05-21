from dataset.base_function import dataset_path, crop_3
import glob
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

confocal_path = os.path.join(dataset_path, 'all_filtered')

class ConfocalTrainDataset(Dataset):
    def __init__(self, patch_size, pin_memory=True):
        super().__init__()
        self.patch_size = patch_size
        self.pin_memory = pin_memory

        self._img_paths = self._get_img_paths()
        if self.pin_memory:
            self._imgs = self._open_images()

    def __getitem__(self, index):
        index = index % len(self._img_paths)

        if self.pin_memory:
            img_L = self.imgs[index]['L']
        else:
            img_path = self._img_paths[index]
            img_L = self._open_image(img_path['L'])

        patch_L = crop_3(self.patch_size, img_L)

        return {'L': patch_L.copy()}

    def __len__(self):
        return len(self._img_paths) * 100

    def _get_img_paths(self):
        # Confocal data now uses standard grayscale image files; .mat patterns are removed.
        patterns = [
            os.path.join(confocal_path, '**', f'*.{ext}')
            for ext in ('png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp')
        ]
        L_paths = []
        for pattern in patterns:
            L_paths.extend(glob.glob(pattern, recursive=True))
        L_paths = sorted(L_paths)

        img_paths = []
        for L_path in L_paths:
            img_paths.append({'L':L_path})
        return img_paths

    def _open_images(self):
        self.imgs = []
        for img_path in self._img_paths:
            img_L = self._open_image(img_path['L'])
            self.imgs.append({'L': img_L})

    def _open_image(self, path):
        with Image.open(path) as img:
            if img.size[0] != img.size[1] or img.size[0] not in (512, 1024):
                raise ValueError(
                    f'Unexpected confocal size {img.size} in {path}; '
                    'expected 512x512 or 1024x1024.'
                )
            img = np.asarray(img)
            if img.ndim == 3:
                img = img[:, :, 0]
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)