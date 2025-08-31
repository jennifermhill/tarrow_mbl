# %%
from tarrow.data import TarrowDataset
from tarrow.data import get_augmenter
import matplotlib.pyplot as plt
import torch

augmenter = get_augmenter(5)
dataset = TarrowDataset(
    imgs="/mnt/efs/aimbl_2025/student_data/S-CV/pre/zarr/C02",
    delta_frames=[1],
    augmenter=augmenter
    )


img, label = dataset
print(img.shape)

plt.imshow(torch.squeeze(img)[0])

# %%
