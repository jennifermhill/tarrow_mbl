from image_modules import shade_by_cluster, reassign_cluster_ids, get_time_measurements
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
import numpy as np

def get_covariance_segmentation(window):
    
    (t, h, w) = np.shape(window)
    measurements = get_time_measurements(window)
    autocorr = np.cov(measurements)

    assert np.shape(measurements) == (h*w, t)
    assert np.shape(autocorr) == (h*w,h*w)

    # row distances will have size (h*w)_C_2 for pairwise comparisons
    row_distances = pdist(measurements, metric='cosine')
    
    isfinite = np.isfinite(row_distances)
    nonfinite_idx = np.where(~isfinite)
    finite_idx = np.where(isfinite)
    row_distances[nonfinite_idx] = np.max(row_distances[finite_idx])
    assert not np.any(np.isnan(row_distances))
    
    linkage_matrix = hierarchy.linkage(row_distances, method='ward')
    reordered_indices = hierarchy.leaves_list(linkage_matrix)
    reordered_matrix = autocorr[reordered_indices, :]
    reordered_matrix = reordered_matrix[:, reordered_indices]

    num_clusters = 2
    cluster_assignments = hierarchy.fcluster(linkage_matrix, num_clusters, "maxclust")
    cluster_assignments_reordered = cluster_assignments[reordered_indices]

    if not np.all(cluster_assignments_reordered[:-1] <= cluster_assignments_reordered[1:]):
        cluster_assignments[np.where(cluster_assignments==1)] = 3
        cluster_assignments[np.where(cluster_assignments==2)] = 1
        cluster_assignments[np.where(cluster_assignments==3)] = 2
        cluster_assignments_reordered = cluster_assignments[reordered_indices]

    idx_clust1 = np.where(cluster_assignments_reordered==1)[0]
    idx_boundary = np.max(idx_clust1)+1
    # make sure `cluster_assignments_reordered` is sorted
    assert np.all(cluster_assignments_reordered[:-1] <= cluster_assignments_reordered[1:]), cluster_assignments_reordered
    assert list(idx_clust1) == list(range(np.max(idx_clust1)+1))

    cluster_assignments = reassign_cluster_ids(reordered_matrix, cluster_assignments, idx_boundary)

    return shade_by_cluster(cluster_assignments, h, w), reordered_matrix