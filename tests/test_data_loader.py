# %%
from tarrow.data import TarrowDataset
from tarrow.data import get_augmenter



# %%
aug = get_augmenter(5)

# %%

dataset = TarrowDataset(
    imgs="/Volumes/sgrolab/jennifer/cryolite/cryolite_mixin_test65_2024-04-16/WS205_overnight_day2/2024-04-17_ERH_mixin65_plate2_WS205_overnight_day2_ERH Red FarRed/analysis/max_projections/maxz",
    delta_frames=[1],
    mode="flip",
    augmenter=aug,
    )

img = dataset[0]
img[0].shape
# %%
