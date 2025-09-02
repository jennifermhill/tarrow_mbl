import logging
from pathlib import Path
import bisect
from pyparsing import annotations
from tqdm import tqdm
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from skimage.transform import downscale_local_mean
from ..utils import normalize as utils_normalize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TarrowDataset(Dataset):
    def __init__(
        self,
        imgs,
        split_start=0,
        split_end=1,
        n_images=None,
        n_frames=2,
        delta_frames=[1],
        subsample=1,
        mode="flip",
        permute=True,
        augmenter=None,
        normalize=None,
        channels=0,
        device="cpu",
        binarize=False,
        crop_size=[128, 128],
        random_crop=True,
        vis_crop=False,
        annotations=None,
        annotation_range=30,
    ):
        """Returns 2d+time crops. The image sequence is stored in-memory.

        Args:
            imgs:
                Path to a sequence of 2d images, a single 2d+time image, or a list of 2d np.ndarrays.
            split_start:
                Start point of relative split of image sequence to use.
            split_end:
                End point of relative split of image sequence to use.
            n_images:
                Limit the number of images to use. Useful for debugging.
            n_frames:
                Number of frames in each crop.
            delta_frames:
                Temporal delta(s) between input frames.
            subsample:
                Subsample the input images by this factor.
            size:
                Patch size. If None, use the full image size.
            mode:
                `flip` or `roll` the images along the time axis.
            permute:
                Whether to permute the axes of the images. Set to False for visualizations.
            augmenter:
                Torch transform to apply to the images.
            normalize:
                Image normalization function, applied before croppning. If None, use default percentile-based normalization.
            channels:
                Take the n leading channels from the ones stored in the raw images (leading dimension). 0 means there is no channel dimension in raw files.
            device:
                Where to store the precomputed crops.
            binarize:
                Binarize the input images. Should only be used for images stored in integer format.
            random_crop:
                If `True`, crop random patches in spatial dimensions. If `False`, center-crop the images (e.g. for visualization).
            reject_background:
                help="Set to `True` to heuristically reject background patches.
        """

        super().__init__()
        
        self._split_start = split_start
        self._split_end = split_end
        self._n_images = n_images
        self._n_frames = n_frames
        self._delta_frames = delta_frames
        self._subsample = subsample

        self.random_crop = random_crop

        assert mode in ["flip", "roll"]
        self._mode = mode

        self._permute = permute
        self._channels = channels
        self._device = device
        self._binarize = binarize
        self._augmenter = augmenter
        self._normalize = normalize
        self._random_crop = random_crop
        self._vis_crop = vis_crop
        self._annotations = annotations
        self._annotation_range = annotation_range
        if self._augmenter is not None:
            self._augmenter.to(device)
    
        if isinstance(imgs, (str, Path)):
            self._imgs = self._load_zarr(path=imgs)

        else:
            raise ValueError(
                f"Cannot form a dataset from {imgs}. "
                "Input should be a path to a zarr file."
            )
        if crop_size is not None:
            self._crop_size = crop_size
        else:
            imgs_shape = self._imgs.shape
            self._crop_size = [imgs_shape[3], imgs_shape[4]]

        if not isinstance(subsample, int) or subsample < 1:
            raise NotImplementedError(
                "Spatial subsampling only implemented for positive integer values."
            )

        if self._imgs.ndim != 5:  # T, C, Z, X, Y
            raise NotImplementedError(
                f"only ome-zarr format supported (total image shape: {imgs.shape})"
            )
        self._size = self._imgs.shape[3:]
        min_number = max(self._delta_frames) * (n_frames - 1) + 1
        if self._imgs.shape[0] < min_number:
            raise ValueError(f"imgs should contain at least {min_number} elements")
        if len(self._imgs.shape[3:]) != len(self._size):
            raise ValueError(
                f"incompatible shapes between images and size last {n_frames} elements"
            )
        
        for delta in self._delta_frames:
            n, k = self._n_frames, delta
            logger.debug(f"Creating delta {delta} crops")
            self._tslices = tuple(
                slice(i, i + k * (n - 1) + 1, k) for i in range(self._imgs.shape[0] - (n - 1) * k)
            )

        self._crops_per_image = max(
            1, int(np.prod(self._imgs.shape[1:3]) / np.prod(self._size))
        )

    # def _reject_background(self, threshold=0.02, max_iterations=10):
    #     rc = transforms.RandomCrop(
    #         self._size,
    #         padding_mode="reflect",
    #         pad_if_needed=True,
    #     )

    #     def smoother(img):
    #         img = skimage.util.img_as_ubyte(img.squeeze(1).numpy().clip(-1, 1))
    #         img = skimage.filters.rank.median(
    #             img, footprint=np.ones((self._n_frames, 3, 3))
    #         )
    #         return torch.as_tensor(skimage.util.img_as_float32(img)).unsqueeze(1)

    #     def crop(x):
    #         with torch.no_grad():
    #             for i in range(max_iterations):
    #                 out = rc(x)
    #                 mask = smoother(out)
    #                 if mask.std() > threshold:
    #                     return out
    #                 # logger.debug(f"Reject {i}")
    #             return out

    #     return crop

    def _default_normalize(self, imgs):
        """Default normalization.

        Normalizes each image separately. Can be overwritten in subclasses.

        Args:
            imgs: List of images or ndarray.

        Returns:
            ndarray

        """
        imgs_norm = []
        for img in imgs:
            imgs_norm.append(utils_normalize(img, subsample=8))
        return np.stack(imgs_norm)

    def _load_zarr(self, path):

        logger.info(f"Loading {path}")
        imgs = zarr.open(str(path), mode="r")
        return imgs
    
    def _crop_from_annotations(self, img_sequence, annot, crop_size):

        # check if the crop will be out of range in x or y and shift away from edge if true
        if annot[2] - crop_size[0]//2 < 0:
            annot[2] = crop_size[0]//2
        if annot[3] - crop_size[1]//2 < 0:
            annot[3] = crop_size[1]//2
        if annot[2] + crop_size[0]//2 > img_sequence.shape[3]:
            annot[2] = img_sequence.shape[3] - crop_size[0]//2
        if annot[3] + crop_size[1]//2 > img_sequence.shape[4]:
            annot[3] = img_sequence.shape[4] - crop_size[1]//2
        return img_sequence[:, :, 0,
               annot[2] - crop_size[0]//2:annot[2] + crop_size[0]//2,
               annot[3] - crop_size[1]//2:annot[3] + crop_size[1]//2]

    def __len__(self):
        return len(self._tslices)

    def __getitem__(self, idx):

        imgs_shape = self._imgs.shape
        assert (type(self._crop_size[0]) == int), (type(self._crop_size[0]), self._crop_size[0])
        assert (type(imgs_shape[4]) == type(self._crop_size[1])), (type(imgs_shape[4]), type(self._crop_size[1]))
        if imgs_shape[3] < self._crop_size[0] or imgs_shape[4] < self._crop_size[1]:
            raise ValueError("Crop size must be smaller than image size")
        
        # Get a random tslice
        if self._random_crop:
            t = np.random.randint(0, len(self._tslices))
        else:
            # randomly choose annotation location
            annot_idx = np.random.randint(0, len(self._annotations))
            annot = self._annotations[annot_idx]
            t_min = max(0, annot[0]-self._annotation_range)
            t_max = min(len(self._tslices),annot[0])
            t = np.random.randint(t_min, t_max)
        tslice = self._tslices[t]
        img_sequence = self._imgs[tslice]

        # Binarize and normalize
        if self._binarize:
            #logger.debug("Binarize images")
            img_sequence = (img_sequence > 0).astype(np.float32)
        else:
            #logger.debug("Normalize images")
            if self._normalize is None:
                img_sequence = self._default_normalize(img_sequence)
            else:
                img_sequence = self._normalize(img_sequence)

        if not self._vis_crop:
            # Get a (random) crop
            if self._random_crop:
                i = np.random.randint(self._crop_size[0]//2, imgs_shape[3] - self._crop_size[0]//2 + 1)
                j = np.random.randint(self._crop_size[1]//2, imgs_shape[4] - self._crop_size[1]//2 + 1)
                # Get the cropped image
                x = img_sequence[:, :, 0, i - self._crop_size[0]//2:i + self._crop_size[0]//2, j - self._crop_size[1]//2:j + self._crop_size[1]//2]
            else:
                x = self._crop_from_annotations(img_sequence, annot, self._crop_size)
        else:
            x = img_sequence[:, :, 0, :, :]

        x = torch.tensor(x, dtype=torch.float32)

        if self._subsample > 1:
            factors = (1,) + (self._subsample,) * (x.dim() - 2)
            full_size = x[0].shape
            x = downscale_local_mean(x, factors)
            logger.debug(f"Subsampled from {full_size} to {x[0].shape}")

        if self._permute:
            if self._mode == "flip":
                label = torch.randint(0, 2, (1,))[0]
                if label == 1:
                    x = torch.flip(x, dims=(0,))
            elif self._mode == "roll":
                label = torch.randint(0, self._n_frames, (1,))[0]
                x = torch.roll(x, label.item(), dims=(0,))
            else:
                raise ValueError()
        else:
            label = torch.tensor(0, dtype=torch.long)


        if self._augmenter is not None:
            x = self._augmenter(x)

        x, label = x.to(self._device), label.to(self._device)
        return x, label


class ConcatDatasetWithIndex(ConcatDataset):
    """Additionally returns index"""

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], idx
