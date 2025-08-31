# %%
from tarrow.data import TarrowDataset
from tarrow.data import get_augmenter



# %%
aug = get_augmenter(5)

# %%

augmenter = get_augmenter(5)
dataset = TarrowDataset(
    imgs="/Volumes/sgrolab/jennifer/cryolite/cryolite_mixin_test66_2024-04-30/ERH_2024-04-30_mixin66_plate1_day1_ERH Red FarRed.zarr",
    delta_frames=[1],
    mode="flip",
    augmenter=aug,
    )

img = dataset[0]
img[0].shape
# %%
