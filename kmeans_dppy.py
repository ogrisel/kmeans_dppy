# %%
import numba_dppy as dppy
import dpctl
import numpy as np


gpu = dpctl.select_gpu_device()
gpu.print_device_info()

# %%
rng = np.random.RandomState(42)
data = rng.normal(size=(1_000_000, 100)).astype(np.float32)
cluster_indices = np.zeros_like(data, dtype=np.int32)
centroids = data[rng.choice(np.arange(data.shape[0]), 100)]
data.nbytes / 1e6

# %%
# The following way to device allocate the USM arrays is currently not
# supported: it would make the numba_dppy kernel compiler raise a
# NotImplementedError.

# import dpctl.tensor as dpt
# 
# def convert_to_usm(data, buffer="shared"):
#     data_usm = dpt.usm_ndarray(data.shape, dtype=data.dtype, buffer=buffer)
#     data_usm.usm_data.copy_from_host(data.ravel().view("u1"))
#     return data_usm

# %%
import dpctl.memory as dpmem


def convert_to_usm(data, buffer="shared"):
    data_usm = dpmem.MemoryUSMShared(data.nbytes)
    data_usm.copy_from_host(data.ravel().view("u1"))
    return np.ndarray(data.shape, buffer=data_usm, dtype=data.dtype)


with dpctl.device_context(gpu):
    data = convert_to_usm(data)
    centroids = convert_to_usm(centroids)
    cluster_indices = convert_to_usm(cluster_indices)


# %%
data.base.sycl_device

# %%
def make_kernels(n_samples, n_features, n_centroids):
    @dppy.kernel
    def assign(data, cluster_indices, centroids):
        sample_idx = dppy.get_global_id(0)
        if sample_idx < n_samples:
            min_dist = -1
            for centroid_idx in range(n_centroids):
                dist = 0.0
                for feature_idx in range(n_features):
                    dist += (
                        data[sample_idx, feature_idx]
                        - centroids[centroid_idx, feature_idx]
                    ) ** 2
                if min_dist > dist or min_dist < 0:
                    min_dist = dist
                    cluster_indices[sample_idx] = centroid_idx

    return assign


# %%
with dpctl.device_context(gpu):
    assign = make_kernels(data.shape[0], data.shape[1], centroids.shape[0])


# %%
%%time
assign[data.shape[0], dppy.DEFAULT_LOCAL_SIZE](
    data, cluster_indices, centroids
)

# %%
cluster_indices

# %%
