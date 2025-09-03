#%%

import zarr
import matplotlib.pyplot as plt
import numpy as np
from image_modules import normalize, get_slices
from covariance_segment import get_covariance_segmentation
from affinity_segment import get_affinity_segmentation
from tqdm import tqdm
import torch
from torch.nn import MaxPool2d
import mwatershed
from scipy.ndimage import gaussian_filter
import mwatershed

def relu(x, b=0):
    if x >= b:
        return x
    else:
        return 0
relu_np = np.vectorize(relu)

def minmax(x):
    return (x - np.min(x))/(np.max(x) - np.min(x) + 1e-10)

def get_percentile(x, pmin, pmax):
    if x.ndim == 3:
        return np.array([get_percentile(elt, pmin, pmax) for elt in x])
    else:
        assert x.ndim == 2
    out = np.zeros_like(x)
    min_, max_ = np.percentile(x, (pmin, pmax))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] < min_:
                out[i,j] = 0
            elif x[i,j] > max_:
                out[i,j] = 1
            else:
                out[i,j] = (x[i,j] - min_)/(max_-min_+1e-10)
    
    return out

def sigmoid(x, T, b):
    return 1/(1+np.exp(-(x-b)/T))

def gaussian_smooth(x, sigma):
    if x.ndim == 3:
        return np.array([gaussian_smooth(elt, sigma) for elt in x])
    else:
        assert x.ndim == 2
    return gaussian_filter(x, sigma)



def main_covariance(imgs_normalized):

    output = np.zeros_like(imgs_normalized[0,:,:])
    counts = np.zeros_like(imgs_normalized[0,:,:])

    window_size = (20, 20)
    stride_y = 16
    stride_x = 16

    yslices, xslices = get_slices(window_size, (stride_y, stride_x), imgs_normalized.shape[1:])

    for yslice in tqdm(yslices):
        for xslice in xslices:
            window = imgs_normalized[:,yslice,xslice]
            segmentation, _ = get_covariance_segmentation(window)
            output[yslice, xslice] += segmentation
            counts[yslice, xslice] += np.ones_like(segmentation)

    output = output / counts

    fig, axs = plt.subplots(1, 1, figsize=(10,10))
    axs.matshow(output)
    fig.savefig("/home/S-CV/images/test.png", dpi=400, bbox_inches='tight')

def main_affinities(imgs_normalized, offsets):
    
    return get_affinity_segmentation(imgs_normalized, offsets)


# if __name__ == '__main__':
#%%
path = "/mnt/efs/aimbl_2025/student_data/S-CV/pre/zarr/B02"
imgs = np.squeeze(zarr.open(str(path), mode="r"))
imgs = imgs[:,:,:]
# down = MaxPool2d(kernel_size=2)
# imgs = down(torch.tensor(imgs, dtype=torch.float32))
# imgs = np.array(imgs)

imgs_smoothed = gaussian_smooth(imgs, sigma=1)
assert np.shape(imgs) == np.shape(imgs_smoothed)
print("Image smoothed")

plt.imshow(imgs[150,:,:])
plt.title("Image")
plt.show()
plt.imshow(imgs_smoothed[150,:,:])
plt.title("Smoothed")
plt.show()

imgs_normalized = np.apply_along_axis(normalize, 0, imgs)
print("Image normalized")


#%%

offsets = [(1,0), (0,1),
           (2,0), (0,2),
            (3,0), (0,3),
            (5,0), (0,5),
            (9,0), (0,9),
            (12,0), (0,12)]
affinities = main_affinities(imgs_normalized, offsets)

#%%
fig, axs = plt.subplots(1, affinities.shape[0]+1)
for i, ax in enumerate(axs):
    ax.set_axis_off()
    if i > 0:
        ax.matshow(affinities[i-1,:,:])
        ax.set_title(offsets[i-1])
    else:
        ax.matshow(imgs_normalized[120,:,:])
        ax.set_title("image")








# %%

tau = 1e-10
bias = 0.03

pmin = 2
pmax = 98

sigma = 1

affinities_rescaled = sigmoid(affinities, tau, bias)
assert affinities_rescaled.shape == affinities.shape

#%%
idx = slice(0,len(offsets),1)

bias_1 = 0
bias_2 = 0
bias_3 = 0
bias_5 = 0
bias_9 = 0
bias_12 = 0

inpt_affinities = np.array([
        affinities_rescaled[0] + bias_1,
        affinities_rescaled[1] + bias_1,
        affinities_rescaled[2] + bias_2,
        affinities_rescaled[3] + bias_2,
        affinities_rescaled[4] + bias_3,
        affinities_rescaled[5] + bias_3,
        affinities_rescaled[6] + bias_5,
        affinities_rescaled[7] + bias_5,
        affinities_rescaled[8] + bias_9,
        affinities_rescaled[9] + bias_9,
        affinities_rescaled[10] + bias_12,
        affinities_rescaled[11] + bias_12
    ], dtype=np.float64)

components = mwatershed.agglom(
    affinities=inpt_affinities[idx],
    offsets=[list(offset) for offset in offsets[idx]]
)
print(f"Number of objects: {len(set(components.flatten()))}")

fig, ax = plt.subplots(1,1)
_=ax.matshow(components)

#%% 
inpt_affinities.shape

    # print(sigmoid(min_), sigmoid(max_))
# %%


idx = 2

plt.matshow(affinities[idx,:,:])
plt.colorbar()
plt.title(offsets[idx])
plt.show()

plt.matshow(affinities_rescaled[idx,:,:])
plt.colorbar()
plt.title(offsets[idx])
plt.show()



# %%

min_, max_ = np.percentile(affinities[idx,:,:].flatten(), (pmin, pmax))

plt.hist(affinities[idx,:,:].flatten(), bins=50)
plt.axvline(min_)
plt.axvline(max_)
plt.show()

plt.hist(get_percentile(affinities[idx,:,:], pmin, pmax).flatten(), bins=50)
plt.show()
# %%
