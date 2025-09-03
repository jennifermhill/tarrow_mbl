import numpy as np
from tqdm import tqdm

def get_affinity_segmentation(window, offsets):

    affinities = np.zeros(shape=(len(offsets), window.shape[1], window.shape[2]))

    for y in tqdm(range(window.shape[1])):
        for x in range(window.shape[2]):
            cur_values = window[:,y,x]
            for i, offset in enumerate(offsets):
                (dy, dx) = offset

                if y+dy < window.shape[1] and x+dx < window.shape[2]:
                    offset_values = window[:,y+dy,x+dx]
                    measurements = np.vstack((cur_values, offset_values))
                    assert measurements.shape == (2, window.shape[0]), f"Measurements has shape {measurements.shape}. Try transposing before vstack"
                    covariance = np.cov(measurements)[0,1]
                    affinities[i,y,x] = covariance
                else:
                    affinities[i,y,x] = 0

    return affinities