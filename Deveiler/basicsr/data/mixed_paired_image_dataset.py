from torch.utils import data as data
from torchvision.transforms.functional import normalize
import os
from os import path as osp
import numpy as np
import torch
import warnings

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MixedPairedImageDataset(data.Dataset):
    """Mixed paired image dataset for image restoration.
    
    Support multiple GT and LQ folder pairs for domain adaptation.
    
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (list): List of data root paths for gt folders.
        dataroot_lq (list): List of data root paths for lq folders.
        folder_weights (list): Weights for each folder pair. If None, equal weights.
        meta_info_file (str): Path for a single meta information file applied to all folder pairs.
        meta_info_files (list[str|None]): Optional list of meta files aligned with each folder pair.
            If provided, it overrides `meta_info_file` on a per-folder basis. Use `null` to fall
            back to folder scanning for a specific pair.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(MixedPairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # Support both single folder and multiple folders
        if isinstance(opt['dataroot_gt'], str):
            self.gt_folders = [opt['dataroot_gt']]
            self.lq_folders = [opt['dataroot_lq']]
        else:
            self.gt_folders = opt['dataroot_gt']
            self.lq_folders = opt['dataroot_lq']
            
        assert len(self.gt_folders) == len(self.lq_folders), \
            f"Number of GT folders ({len(self.gt_folders)}) must match LQ folders ({len(self.lq_folders)})"
        
        # Folder probabilities for sampling
        self.folder_probs = opt.get('folder_probs', None)
        if self.folder_probs is None:
            self.folder_probs = [1.0 / len(self.gt_folders)] * len(self.gt_folders)  # Equal probability
        else:
            assert len(self.folder_probs) == len(self.gt_folders), \
                f"Number of folder probabilities ({len(self.folder_probs)}) must match number of folders ({len(self.gt_folders)})"
            # Normalize probabilities to sum to 1
            prob_sum = sum(self.folder_probs)
            self.folder_probs = [p / prob_sum for p in self.folder_probs]
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # Collect all paths from all folders
        self.folder_paths = []  # Store paths for each folder separately
        self.folder_sizes = []  # Store size of each folder
        # Backward compatible meta config: support either a single meta file or a list per folder
        self.global_meta_info_file = self.opt.get('meta_info_file', None)
        meta_files_cfg = self.opt.get('meta_info_files', None)
        if meta_files_cfg is not None:
            # Accept a single string but prefer a list; convert to list for uniform handling
            if isinstance(meta_files_cfg, str):
                self.meta_info_files = [meta_files_cfg]
            else:
                self.meta_info_files = list(meta_files_cfg)
            assert len(self.meta_info_files) == len(self.gt_folders), \
                f"Number of meta_info_files ({len(self.meta_info_files)}) must match number of folders ({len(self.gt_folders)})"
        else:
            self.meta_info_files = None
        
        for folder_idx, (gt_folder, lq_folder) in enumerate(zip(self.gt_folders, self.lq_folders)):
            if self.io_backend_opt['type'] == 'lmdb':
                # For lmdb, we need to handle each folder separately
                folder_paths = paired_paths_from_lmdb([lq_folder, gt_folder], ['lq', 'gt'])
            else:
                # Resolve per-folder meta file with backward compatibility
                meta_file_for_folder = None
                if self.meta_info_files is not None:
                    meta_file_for_folder = self.meta_info_files[folder_idx]
                if meta_file_for_folder is None and self.global_meta_info_file is not None:
                    meta_file_for_folder = self.global_meta_info_file
                if meta_file_for_folder is not None:
                    folder_paths = paired_paths_from_meta_info_file([lq_folder, gt_folder], ['lq', 'gt'],
                                                                   meta_file_for_folder, self.filename_tmpl)
                else:
                    folder_paths = paired_paths_from_folder([lq_folder, gt_folder], ['lq', 'gt'], self.filename_tmpl)
            
            self.folder_paths.append(folder_paths)
            self.folder_sizes.append(len(folder_paths))
        
        # Calculate total dataset size (sum of all folder sizes)
        self.total_size = sum(self.folder_sizes)
        
        print(f"MixedPairedImageDataset: Loaded {self.total_size} image pairs from {len(self.gt_folders)} folder pairs")
        for i, (gt_folder, lq_folder, prob) in enumerate(zip(self.gt_folders, self.lq_folders, self.folder_probs)):
            print(f"  Folder pair {i}: {self.folder_sizes[i]} pairs, probability: {prob:.3f}")
            print(f"    GT: {gt_folder}")
            print(f"    LQ: {lq_folder}")
            # Report which meta file was used for this folder, if any
            meta_used = None
            if self.meta_info_files is not None:
                meta_used = self.meta_info_files[i]
            if meta_used is None:
                meta_used = self.global_meta_info_file
            if meta_used is not None:
                print(f"    meta: {meta_used}")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Sample folder based on probabilities
        import random
        folder_idx = random.choices(range(len(self.folder_paths)), weights=self.folder_probs, k=1)[0]
        
        # Sample random index within the selected folder
        folder_size = self.folder_sizes[folder_idx]
        folder_index = random.randint(0, folder_size - 1)
        
        # Get the actual path
        path_info = self.folder_paths[folder_idx][folder_index]
        gt_path = path_info['gt_path']
        lq_path = path_info['lq_path']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        result = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        
        # Add folder index for domain adaptation
        result['folder_idx'] = folder_idx
        
        return result

    def __len__(self):
        return self.total_size




@DATASET_REGISTRY.register()
class MixedPairedImageDataset_with_LPE_Map(data.Dataset):
    """Mixed paired image dataset for image restoration with LPE map support.
    
    Support multiple GT and LQ folder pairs for domain adaptation.
    Support loading LPE (Local Parameter Enhancement) maps for each image.
    
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (list): List of data root paths for gt folders.
        dataroot_lq (list): List of data root paths for lq folders.
        LPE_map_folder (list): List of LPE map folder paths. Use null for folders without LPE maps.
        folder_weights (list): Weights for each folder pair. If None, equal weights.
        meta_info_file (str): Path for a single meta information file applied to all folder pairs.
        meta_info_files (list[str|None]): Optional list of meta files aligned with each folder pair.
            If provided, it overrides `meta_info_file` on a per-folder basis. Use `null` to fall
            back to folder scanning for a specific pair.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(MixedPairedImageDataset_with_LPE_Map, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # Support both single folder and multiple folders
        if isinstance(opt['dataroot_gt'], str):
            self.gt_folders = [opt['dataroot_gt']]
            self.lq_folders = [opt['dataroot_lq']]
        else:
            self.gt_folders = opt['dataroot_gt']
            self.lq_folders = opt['dataroot_lq']
            
        assert len(self.gt_folders) == len(self.lq_folders), \
            f"Number of GT folders ({len(self.gt_folders)}) must match LQ folders ({len(self.lq_folders)})"
        
        # LPE MAP folders configuration
        lpe_map_folders_cfg = opt.get('LPE_map_folder', None)
        if lpe_map_folders_cfg is not None:
            if isinstance(lpe_map_folders_cfg, str):
                self.lpe_map_folders = [lpe_map_folders_cfg]
            else:
                self.lpe_map_folders = list(lpe_map_folders_cfg)
            assert len(self.lpe_map_folders) == len(self.gt_folders), \
                f"Number of LPE_map_folders ({len(self.lpe_map_folders)}) must match number of folders ({len(self.gt_folders)})"
        else:
            self.lpe_map_folders = [None] * len(self.gt_folders)
        
        # Folder probabilities for sampling
        self.folder_probs = opt.get('folder_probs', None)
        if self.folder_probs is None:
            self.folder_probs = [1.0 / len(self.gt_folders)] * len(self.gt_folders)  # Equal probability
        else:
            assert len(self.folder_probs) == len(self.gt_folders), \
                f"Number of folder probabilities ({len(self.folder_probs)}) must match number of folders ({len(self.gt_folders)})"
            # Normalize probabilities to sum to 1
            prob_sum = sum(self.folder_probs)
            self.folder_probs = [p / prob_sum for p in self.folder_probs]
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # Collect all paths from all folders
        self.folder_paths = []  # Store paths for each folder separately
        self.folder_sizes = []  # Store size of each folder
        # Backward compatible meta config: support either a single meta file or a list per folder
        self.global_meta_info_file = self.opt.get('meta_info_file', None)
        meta_files_cfg = self.opt.get('meta_info_files', None)
        if meta_files_cfg is not None:
            # Accept a single string but prefer a list; convert to list for uniform handling
            if isinstance(meta_files_cfg, str):
                self.meta_info_files = [meta_files_cfg]
            else:
                self.meta_info_files = list(meta_files_cfg)
            assert len(self.meta_info_files) == len(self.gt_folders), \
                f"Number of meta_info_files ({len(self.meta_info_files)}) must match number of folders ({len(self.gt_folders)})"
        else:
            self.meta_info_files = None
        
        for folder_idx, (gt_folder, lq_folder) in enumerate(zip(self.gt_folders, self.lq_folders)):
            if self.io_backend_opt['type'] == 'lmdb':
                # For lmdb, we need to handle each folder separately
                folder_paths = paired_paths_from_lmdb([lq_folder, gt_folder], ['lq', 'gt'])
            else:
                # Resolve per-folder meta file with backward compatibility
                meta_file_for_folder = None
                if self.meta_info_files is not None:
                    meta_file_for_folder = self.meta_info_files[folder_idx]
                if meta_file_for_folder is None and self.global_meta_info_file is not None:
                    meta_file_for_folder = self.global_meta_info_file
                if meta_file_for_folder is not None:
                    folder_paths = paired_paths_from_meta_info_file([lq_folder, gt_folder], ['lq', 'gt'],
                                                                   meta_file_for_folder, self.filename_tmpl)
                else:
                    folder_paths = paired_paths_from_folder([lq_folder, gt_folder], ['lq', 'gt'], self.filename_tmpl)
            
            self.folder_paths.append(folder_paths)
            self.folder_sizes.append(len(folder_paths))
        
        # Calculate total dataset size (sum of all folder sizes)
        self.total_size = sum(self.folder_sizes)
        
        print(f"MixedPairedImageDataset_with_LPE_Map: Loaded {self.total_size} image pairs from {len(self.gt_folders)} folder pairs")
        for i, (gt_folder, lq_folder, prob) in enumerate(zip(self.gt_folders, self.lq_folders, self.folder_probs)):
            print(f"  Folder pair {i}: {self.folder_sizes[i]} pairs, probability: {prob:.3f}")
            print(f"    GT: {gt_folder}")
            print(f"    LQ: {lq_folder}")
            # Report LPE map folder
            lpe_folder = self.lpe_map_folders[i]
            if lpe_folder is not None:
                print(f"    LPE_map: {lpe_folder}")
            else:
                print(f"    LPE_map: None (will use default scale_map=0, shift_map=0)")
            # Report which meta file was used for this folder, if any
            meta_used = None
            if self.meta_info_files is not None:
                meta_used = self.meta_info_files[i]
            if meta_used is None:
                meta_used = self.global_meta_info_file
            if meta_used is not None:
                print(f"    meta: {meta_used}")
        
        # No iteration counter needed for original random sampling
        # Maximum data diversity is achieved through completely random sampling

    def _load_lpe_maps(self, img_name, folder_idx):
        """Load LPE scale and shift maps for the given image name and folder index.
        
        Args:
            img_name (str): Image name without extension
            folder_idx (int): Folder index
            
        Returns:
            tuple: (scale_map, shift_map) as torch tensors
        """
        lpe_folder = self.lpe_map_folders[folder_idx]
        
        if lpe_folder is None:
            if self.opt['phase'] == 'train':
                # print(f"No LPE maps available for {img_name} in folder {folder_idx} ")
                return None, None
            if self.opt['phase'] == 'val':
                error_msg = f"VALIDATION ERROR: No LPE map folder configured for {img_name} in folder {folder_idx}. Expected LPE maps but none available."
                print(error_msg)
                raise ValueError(error_msg)
            # Return None to indicate no LPE maps available
            return None, None
        
        try:
            # Construct file paths
            scale_path = osp.join(lpe_folder, f"{img_name}_scale.npy")
            shift_path = osp.join(lpe_folder, f"{img_name}_shift.npy")
            
            # Load numpy arrays
            scale_map = np.load(scale_path)
            shift_map = np.load(shift_path)
            # Keep as numpy arrays for now, will convert to tensor after cropping and augmentation
            return scale_map, shift_map
            
        except FileNotFoundError as e:
            error_msg = f"ERROR: LPE map files not found for {img_name} in folder {folder_idx}. Expected files: {scale_path}, {shift_path}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"ERROR: Failed to load LPE maps for {img_name} in folder {folder_idx}. Error: {e}"
            print(error_msg)
            raise Exception(error_msg)

    def _crop_paired_images_and_lpe_maps(self, img_gt, img_lq, scale_map, shift_map, gt_size, scale, gt_path):
        """Crop paired images and corresponding LPE maps with consistent coordinates.
        
        Args:
            img_gt (numpy.ndarray): GT image
            img_lq (numpy.ndarray): LQ image  
            scale_map (numpy.ndarray): LPE scale map (16x downsampled)
            shift_map (numpy.ndarray): LPE shift map (16x downsampled)
            gt_size (int): Target crop size
            scale (int): Scale factor
            gt_path (str): GT image path for seeding
            
        Returns:
            tuple: (cropped_gt, cropped_lq, cropped_scale_map, cropped_shift_map)
        """
        import random
        
        # Original random cropping - ensures maximum data diversity
        # Each time the same image is accessed, it will be cropped at different positions
        # This maximizes the utilization of all image regions
        # No fixed seed - completely random cropping for maximum diversity
        
        # Crop images using the same logic as paired_random_crop
        h, w = img_lq.shape[:2]
        lq_size = gt_size // scale
        
        # Ensure we have enough space to crop
        if h < lq_size or w < lq_size:
            raise ValueError(f'Cannot crop {lq_size}x{lq_size} from image of size {h}x{w}')
        
        # Random crop coordinates
        top = random.randint(0, h - lq_size)
        left = random.randint(0, w - lq_size)
        
        # Crop images
        img_gt = img_gt[top*scale:(top+lq_size)*scale, left*scale:(left+lq_size)*scale, :]
        img_lq = img_lq[top:top+lq_size, left:left+lq_size, :]
        
        # Crop LPE maps (16x downsampled coordinates)
        if scale_map is not None and shift_map is not None:
            # Since img and LPE maps are strictly 16x relationship, we can directly crop
            lpe_top = top // 16
            lpe_left = left // 16
            lpe_size = lq_size // 16
            
            # Direct cropping (much simpler!)
            scale_map = scale_map[..., lpe_top:lpe_top+lpe_size, lpe_left:lpe_left+lpe_size]
            shift_map = shift_map[..., lpe_top:lpe_top+lpe_size, lpe_left:lpe_left+lpe_size]
        
        return img_gt, img_lq, scale_map, shift_map

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        if self.opt['phase'] == 'train':
            # Training: use completely random sampling for maximum data diversity
            import random
            
            # Original random sampling - ensures maximum data diversity
            # Each call to __getitem__ will potentially return different data
            # This maximizes the utilization of all image regions and augmentations
            folder_idx = random.choices(range(len(self.folder_paths)), weights=self.folder_probs, k=1)[0]
            folder_size = self.folder_sizes[folder_idx]
            folder_index = random.randint(0, folder_size - 1)
        else:
            # Validation/Test: use deterministic indexing to avoid duplicates
            # Convert global index to folder_idx and folder_index
            folder_idx = 0
            remaining_index = index
            
            while folder_idx < len(self.folder_paths) and remaining_index >= self.folder_sizes[folder_idx]:
                remaining_index -= self.folder_sizes[folder_idx]
                folder_idx += 1
            
            if folder_idx >= len(self.folder_paths):
                # Fallback to last folder if index is out of bounds
                folder_idx = len(self.folder_paths) - 1
                folder_index = remaining_index % self.folder_sizes[folder_idx]
            else:
                folder_index = remaining_index
        
        # Get the actual path
        path_info = self.folder_paths[folder_idx][folder_index]
        gt_path = path_info['gt_path']
        lq_path = path_info['lq_path']
        
        # Extract image name for LPE map loading
        img_name = osp.splitext(osp.basename(lq_path))[0]
        
        # Load LPE maps if available
        scale_map, shift_map = self._load_lpe_maps(img_name, folder_idx)
        
        # Print loading information
        lq_name = osp.basename(lq_path)
        gt_name = osp.basename(gt_path)
        if scale_map is not None and shift_map is not None:
            lpe_status = f"LPE map: {img_name}_scale.npy, {img_name}_shift.npy"
        else:
            lpe_status = "No LPE map"
        
        # print(f"Loading - LQ: {lq_name}, GT: {gt_name}, {lpe_status}")
        
        # Process LPE maps if they exist
        if scale_map is not None and shift_map is not None:
            # (1, 4, 80, 120) -> (4, 80, 120) numpy
            scale_map = scale_map.squeeze(0)
            shift_map = shift_map.squeeze(0)
        else:
            # No LPE maps available for this image
            # Will create zero tensors later in the pipeline
            scale_map = np.zeros((4, 80, 120))
            shift_map = np.zeros((4, 80, 120))
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Store original dimensions for LPE map processing
        # Note: Original image dimensions are (3, 1280, 1920), LPE maps are (1, 4, 80, 120)
        # LPE maps are 16x downsampled from the original image (1280/80 = 16, 1920/120 = 16)
        orig_h, orig_w = img_lq.shape[:2]

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # print('img_gt',img_gt.shape,'img_lq',img_lq.shape,'scale_map',scale_map.shape,'shift_map',shift_map.shape)
            # Crop images and LPE maps with consistent coordinates
            img_gt, img_lq, scale_map, shift_map = self._crop_paired_images_and_lpe_maps(
                img_gt, img_lq, scale_map, shift_map, gt_size, scale, gt_path
            )
            
            # Apply augmentations to both images and LPE maps with random decisions
            import random
            # Original random augmentation - ensures maximum data diversity
            # Each time the same image is accessed, it will have different augmentations
            # This maximizes the variety of training data
            
            # Store augmentation flags
            do_hflip = self.opt['use_hflip'] and random.random() > 0.5
            do_rot = self.opt['use_rot'] and random.random() > 0.5
            
            # Apply to images
            if do_hflip:
                img_gt = np.fliplr(img_gt)
                img_lq = np.fliplr(img_lq)
            if do_rot:
                img_gt = np.rot90(img_gt, k=1)
                img_lq = np.rot90(img_lq, k=1)
            
            # Apply the same augmentations to LPE maps
            if scale_map is not None and shift_map is not None:
                # Convert to tensor if not already (for integer cropping case)
                if not isinstance(scale_map, torch.Tensor):
                    scale_map = torch.from_numpy(scale_map).float()
                    shift_map = torch.from_numpy(shift_map).float()
                
                if do_hflip:
                    scale_map = torch.flip(scale_map, dims=[-1])  # Horizontal flip
                    shift_map = torch.flip(shift_map, dims=[-1])
                
                if do_rot:
                    scale_map = torch.rot90(scale_map, k=1, dims=[-2, -1])  # 90 degree rotation
                    shift_map = torch.rot90(shift_map, k=1, dims=[-2, -1])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        # Process LPE maps to match image dimensions
        # Note: scale_map and shift_map are now guaranteed to be valid arrays (either real LPE maps or zeros)
        
        # Ensure LPE maps have the same spatial dimensions as the images
        # If LPE maps are 2D, add channel dimension
        # if scale_map.dim() == 2:
        #     scale_map = scale_map.unsqueeze(0)  # Add channel dimension
        #     shift_map = shift_map.unsqueeze(0)  # Add channel dimension
        
        # Convert LPE maps to tensor if they are still numpy arrays
        if not isinstance(scale_map, torch.Tensor):
            scale_map = torch.from_numpy(scale_map).float()
            shift_map = torch.from_numpy(shift_map).float()
        
        # For training phase, LPE maps have already been cropped and augmented
        # # For validation/test phase, we need to handle the case where images might be cropped differently
        # if self.opt['phase'] != 'train':
        #     # During validation, if images are cropped, we need to handle LPE maps accordingly
        #     # This is a simplified approach - you might need to adjust based on your validation setup
        #     if scale_map.shape[-2:] != img_lq.shape[-2:]:
        #         scale_map = torch.nn.functional.interpolate(
        #             scale_map.unsqueeze(0), size=img_lq.shape[-2:], mode='bilinear', align_corners=False
        #         ).squeeze(0)
        #         shift_map = torch.nn.functional.interpolate(
        #             shift_map.unsqueeze(0), size=img_lq.shape[-2:], mode='bilinear', align_corners=False
        #         ).squeeze(0)

        #判断scale_map和shift_map是否全为0，若全为零，则idf == cond
        if scale_map.all() == 0 and shift_map.all() == 0:
            idf = 'cond'
        else:
            idf = 'uncond'
        result = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        
        # Add LPE maps to result
        result['scale_map'] = scale_map
        result['shift_map'] = shift_map
        
        # Add folder index for domain adaptation
        result['folder_idx'] = folder_idx
        result['idf'] = idf
        # print('img_lq',img_lq.shape,'img_gt',img_gt.shape,'scale_map',scale_map.shape,'shift_map',shift_map.shape)
        # print('scale max',scale_map.max(),'scale min',scale_map.min(),'shift max',shift_map.max(),'shift min',shift_map.min())
        return result

    def __len__(self):
        return self.total_size