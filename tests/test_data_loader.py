# %%
from tarrow.data import TarrowDataset
from tarrow.data import get_augmenter



# %%
aug = get_augmenter(5)

# %%

augmenter = get_augmenter(5)
dataset = TarrowDataset(
    imgs="/mnt/efs/aimbl_2025/student_data/S-CV/pre/zarr/C02",
    delta_frames=[1],
    mode="flip",
    augmenter=aug,
    )

img = dataset[0]
img[0].shape
# %%
