import os
import cv2
import glob
import torch
import random
import numpy as np
import torch.utils.data as data

from torchvision.transforms import ToTensor
from utils.data_utils import smart_random_crop


class HybridAberrationTrainingData(data.Dataset):
    """
    支持2种不同数据类型的训练数据集：
    1. 配对数据 (paired data) - 使用 prompt1
    2. 实拍数据 (real data) - 使用 prompt2
    
    支持通过meta info file指定要读取的文件
    """
    def __init__(self, lq_folder, gt_folder, real_folder=None, 
                 crop_size=512, p=0.3, 
                 paired_meta_file=None, real_meta_file=None,
                 filename_tmpl='{}', prompt=None):
        super().__init__()
        self.lq_folder = lq_folder
        self.gt_folder = gt_folder
        self.real_folder = real_folder
        self.p = p  # probability for real_folder
        self.crop_size = crop_size
        self.filename_tmpl = filename_tmpl
        
        # Load paths from meta files or folders
        self.paired_paths = []
        self.real_paths = []
        
        # Load paired data paths
        if paired_meta_file is not None:
            self.paired_paths = self._load_paths_from_meta_file(
                [lq_folder, gt_folder], ['lq', 'gt'], paired_meta_file, filename_tmpl
            )
        else:
            # Fallback to folder scanning
            lq_names = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                lq_names.extend([
                    os.path.basename(name)
                    for name in glob.glob(os.path.join(lq_folder, ext))
                ])
            seen = set()
            lq_names = [n for n in lq_names if not (n in seen or seen.add(n))]
            for lq_name in lq_names:
                gt_path = self._resolve_gt_path(lq_name)
                self.paired_paths.append({
                    'lq_path': os.path.join(lq_folder, lq_name),
                    'gt_path': gt_path
                })
        
        # Load real data paths
        if real_folder is not None:
            if real_meta_file is not None:
                self.real_paths = self._load_paths_from_meta_file(
                    [real_folder], ['real'], real_meta_file, filename_tmpl
                )
            else:
                # Fallback to folder scanning
                real_names = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                    real_names.extend([
                        os.path.basename(name)
                        for name in glob.glob(os.path.join(real_folder, ext))
                    ])
                seen_real = set()
                real_names = [n for n in real_names if not (n in seen_real or seen_real.add(n))]
                for real_name in real_names:
                    self.real_paths.append({
                        'real_path': os.path.join(real_folder, real_name)
                    })
        
        # Define prompts for different data types
        self.prompt1 = "a photograph with spatially varying PSF blur, optical aberrations, defocus, and chromatic fringing."  # src paired data
        # 
        self.prompt2 = "a photograph with spatially varying PSF blur, optical aberrations, defocus, chromatic fringing, and noticeable stray light with veiling glare." # tgt unpaired data
        
        print('########################################################')
        print('len paired_paths:', len(self.paired_paths))
        print('len real_paths:', len(self.real_paths))
        print('########################################################')
    
    def _load_paths_from_meta_file(self, folders, keys, meta_info_file, filename_tmpl):
        """Load paths from meta info file, similar to paired_paths_from_meta_info_file"""
        paths = []
        with open(meta_info_file, 'r') as fin:
            lines = [line.strip() for line in fin if line.strip()]
        
        for line in lines:
            # Parse line: could be "filename" or "filename (shape)" or "lq_filename gt_filename"
            parts = line.split()
            if len(parts) == 1:
                # Single filename
                filename = parts[0]
                if len(folders) == 2:  # Paired data
                    lq_path = os.path.join(folders[0], filename)
                    gt_path = os.path.join(folders[1], filename)
                    paths.append({
                        f'{keys[0]}_path': lq_path,
                        f'{keys[1]}_path': gt_path
                    })
                else:  # Single folder
                    file_path = os.path.join(folders[0], filename)
                    paths.append({
                        f'{keys[0]}_path': file_path
                    })
            elif len(parts) == 2 and '(' in parts[1]:
                # Filename with shape info: "filename (shape)"
                filename = parts[0]
                if len(folders) == 2:  # Paired data
                    lq_path = os.path.join(folders[0], filename)
                    gt_path = os.path.join(folders[1], filename)
                    paths.append({
                        f'{keys[0]}_path': lq_path,
                        f'{keys[1]}_path': gt_path
                    })
                else:  # Single folder
                    file_path = os.path.join(folders[0], filename)
                    paths.append({
                        f'{keys[0]}_path': file_path
                    })
            elif len(parts) == 2:
                # Two filenames: "lq_filename gt_filename"
                lq_filename, gt_filename = parts
                lq_path = os.path.join(folders[0], lq_filename)
                gt_path = os.path.join(folders[1], gt_filename)
                paths.append({
                    f'{keys[0]}_path': lq_path,
                    f'{keys[1]}_path': gt_path
                })
        
        return paths

    def __len__(self):
        return len(self.paired_paths)

    def _resolve_gt_path(self, basename):
        stem, ext = os.path.splitext(basename)
        # Prefer same extension first
        candidate_paths = []
        if ext:
            candidate_paths.append(os.path.join(self.gt_folder, stem + ext))
        # Try common alternatives
        for alt in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            candidate_paths.append(os.path.join(self.gt_folder, stem + alt))
        for path in candidate_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"GT image not found for {basename} in {self.gt_folder}")

    def __getitem__(self, index):
        # Determine which type of data to use based on probabilities
        rand_val = np.random.rand()
        
        if rand_val < self.p and len(self.real_paths) > 0:
            # Use real data 1
            real_path_info = random.choice(self.real_paths)
            real_path = real_path_info['real_path']
            
            real_img = cv2.imread(real_path)
            if real_img is None:
                raise FileNotFoundError(f"Failed to read real aberration image: {real_path}")
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            real_tensor = ToTensor()(smart_random_crop(real_img, self.crop_size)) * 2.0 - 1.0
            real_tensor = real_tensor.clip(-1.0, 1.0)
            return real_tensor, torch.zeros_like(real_tensor, dtype=real_tensor.dtype), self.prompt2, 'uncond'
            
        else:
            # Use paired data
            paired_path_info = self.paired_paths[index % len(self.paired_paths)]
            lq_path = paired_path_info['lq_path']
            gt_path = paired_path_info['gt_path']

            lq = cv2.imread(lq_path)
            gt = cv2.imread(gt_path)

            if lq is None:
                raise FileNotFoundError(f"Failed to read LQ image: {lq_path}")
            if gt is None:
                raise FileNotFoundError(f"Failed to read GT image: {gt_path}")

            # Convert to float [0,1] and RGB for ToTensor
            lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Joint random crop to keep pairing aligned
            stack = np.concatenate((gt, lq), axis=2)
            stack = smart_random_crop(stack, self.crop_size)
            gt_cropped, lq_cropped = stack[:, :, :3], stack[:, :, 3:]

            gt_tensor = ToTensor()(gt_cropped).clip(0.0, 1.0)
            lq_tensor = ToTensor()(lq_cropped).clip(0.0, 1.0)

            # Follow HybridTrainingData convention:
            # target in [-1,1] (LQ), condition in [0,1] (GT), with 'cond'
            return lq_tensor * 2.0 - 1.0, gt_tensor, self.prompt1, 'cond'
