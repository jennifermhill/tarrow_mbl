# %%
from tarrow.data import TarrowDataset
from tarrow.data import get_augmenter

import matplotlib.pyplot as plt
import zarr


# %%

aug = get_augmenter(5)
dataset = TarrowDataset(
    imgs="/Volumes/sgrolab/jennifer/cryolite/cryolite_mixin_test66_2024-04-30/analysis/max_projections/maxz",
    delta_frames=[5],
    n_frames=5,
    mode="flip",
    augmenter=aug,
    random_crop=False,
    annotations=[[94, 0, 1129, 1143], [105, 6, 1211, 227], [55, 10, 850, 993], [129, 2, 1690, 240], [117, 4, 374, 1278], [57, 5, 1044, 828], [114, 6, 656, 411], [124, 10, 64, 128]],
    annotation_range=30,
    )

img = dataset[0]
img[0].shape
# %%

# Visualize both images
plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plt.imshow(img[0][0][0].numpy(), vmax=1, cmap="gray")
plt.axis("off")
plt.title("Image 1")

plt.subplot(1, 3, 2)
plt.imshow(img[0][1][0].numpy(), vmax=1, cmap="gray")
plt.axis("off")
plt.title("Image 2")

plt.subplot(1, 3, 3)
plt.imshow(img[0][0][0].numpy()-img[0][1][0].numpy(), vmin=0, vmax=0.3, cmap="gray")
plt.axis("off")
plt.title("Difference")

# %%

zarr_path = "/Volumes/sgrolab/jennifer/cryolite/cryolite_mixin_test66_2024-04-30/analysis/max_projections/maxz"
zarr = zarr.open(zarr_path, mode="r")
# %%
frame1 = zarr[75, 0, 0, 1065:1193, 1079:1207]
frame2 = zarr[80, 0, 0, 1065:1193, 1079:1207]

frame1.shape


# %%
def plot_two_frames(frame1, frame2):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(frame1, vmax=6000,cmap="gray")
    plt.axis("off")
    plt.title("Image 1")

    plt.subplot(1, 3, 2)
    plt.imshow(frame2, vmax=6000, cmap="gray")
    plt.axis("off")
    plt.title("Image 2")

    plt.subplot(1, 3, 3)
    plt.imshow(frame1-frame2, vmin=100, vmax=8000, cmap="gray")
    plt.axis("off")
    plt.title("Difference")
# %%
frame1 = zarr[45, 0, 0, 1147:1275, 163:291]
frame2 = zarr[50, 0, 0, 1147:1275, 163:291]

frame1.shape

# %%
plot_two_frames(frame1, frame2)

# %%
