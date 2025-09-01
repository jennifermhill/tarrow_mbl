# %%
from tarrow.data import TarrowDataset
from tarrow.data import get_augmenter
import matplotlib.pyplot as plt
import torch 

aug = get_augmenter(0)

dataset = TarrowDataset(
    imgs="/mnt/efs/aimbl_2025/student_data/S-CV/pre/zarr/C02",
    delta_frames=[1],
    mode="flip",
    augmenter=aug,
    random_crop=False,
    split_start = 100,
    split_end = 301
    )

img, label = dataset[0]
print(img.shape)

fig, axs = plt.subplots(1,2,figsize=(12,10))
axs[0].imshow(torch.squeeze(img)[0])
axs[1].imshow(torch.squeeze(img)[1])


# %%
import zarr
path = "/mnt/efs/aimbl_2025/student_data/S-CV/pre/zarr/C02"
imgs = zarr.open(str(path), mode="r")
print(type(imgs))
print(imgs.shape)
imgs_sliced = imgs[15:]
print(imgs_sliced.shape)
# %%
