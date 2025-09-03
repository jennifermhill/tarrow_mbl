#%%
import zarr
import matplotlib.pyplot as plt
import numpy as np
from image_modules import normalize

path = "/mnt/efs/aimbl_2025/student_data/S-CV/pre/zarr/B02"
imgs = normalize(np.squeeze(zarr.open(str(path), mode="r")))
print("Image loaded")

avg_stack = np.mean(imgs, axis=0)
max_stack = np.max(imgs, axis=0)
std_stack = np.std(imgs, axis=0)
range_stack = max_stack - np.min(imgs, axis=0)

median = np.median(imgs, axis=0)
ff = np.max(imgs-avg_stack, axis=0)

#%%

fig, axs = plt.subplots(7, 1, figsize=(10,40))
axs[0].imshow(imgs[150])
axs[0].set_title("Original")
axs[1].imshow(avg_stack)
axs[1].set_title("Average Projection")
axs[2].imshow(max_stack)
axs[2].set_title("Max Projection")
axs[3].imshow(std_stack)
axs[3].set_title("StDev Projection")
axs[4].imshow(range_stack)
axs[4].set_title("Range Projection")
axs[5].imshow(median)
axs[5].set_title("Median Projection")
axs[6].imshow(ff)
axs[6].set_title("Max Projection w/o bgd")



# %%
