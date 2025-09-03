import numpy as np

def stack(image, stack_fn, flatten=True):
    (t, h, w) = np.shape(image)
    if flatten:
        output = np.zeros(shape=(h, w))
    else:
        output = np.zeros(shape=(t, h, w))
        
    for y in range(h):
        for x in range(w):
            assert image[:,y,x].ndim == 1
            val = stack_fn(image[:,y,x])
            if flatten:
                output[y,x] = val
            else:
                output[:,y,x] = val
    return output

def normalize(image):
    min = np.min(image)
    max = np.max(image)
    midpoint = (min + max)/2
    return (image - midpoint) / (max - midpoint + 1e-10)

def shade_by_cluster(clusterings, height, width):
    output = np.zeros(shape=(height,width))
    for i, cluster in enumerate(clusterings):
        y = int(i/width)
        output[y, i - width*y] = cluster
    return output

def reassign_cluster_ids(cov_matrix, cluster_assignments, boundary):

        output = np.zeros_like(cluster_assignments)

        cluster1_mat = np.triu(cov_matrix[:boundary,:boundary], k=1)
        cluster2_mat = np.triu(cov_matrix[boundary:,boundary:], k=1)
        cluster1_mat_flat = cluster1_mat.flatten()
        cluster2_mat_flat = cluster2_mat.flatten()

        mean_cov1 = np.mean(cluster1_mat_flat[np.nonzero(cluster1_mat_flat)])
        mean_cov2 = np.mean(cluster2_mat_flat[np.nonzero(cluster2_mat_flat)])

        # switch order of cluster IDs if necessary
        if mean_cov1 > mean_cov2:
            output[np.where(cluster_assignments==2)] = 0
        else:
            output[np.where(cluster_assignments==1)] = 0
            output[np.where(cluster_assignments==2)] = 1

        return output

def get_slices(window_size, strides, img_shape):
    (stride_y, stride_x) = strides
    yslices = list(slice(i*stride_y, i * stride_y + window_size[0], 1) for i in range((img_shape[0] - window_size[0]) // stride_y + 1))
    xslices = list(slice(i*stride_x, i * stride_x + window_size[1], 1) for i in range((img_shape[1] - window_size[1]) // stride_x + 1))

    if (img_shape[0] - window_size[0]) % stride_y != 0:
        max_i = (img_shape[0] - window_size[0]) // stride_y
        yslices.append(slice(max_i * stride_y + window_size[0], img_shape[0])) 

    if (img_shape[1] - window_size[1]) % stride_x != 0:
        max_i = (img_shape[1] - window_size[1]) // stride_x
        xslices.append(slice(max_i * stride_x + window_size[1], img_shape[1])) 

    return yslices, xslices

def get_time_measurements(window):
    
    (t, h, w) = np.shape(window)
    measurements = np.zeros(shape=(h*w, t))
    for i in range(t):
        measurements[:,i] = np.ravel(window[i,:,:])

    return measurements