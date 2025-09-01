from openpiv import pyprocess, validation, filters, tools
import zarr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = "/mnt/efs/aimbl_2025/student_data/S-CV/pre/zarr/B02"
img = np.squeeze(zarr.open(str(path), mode="r"))

for i in tqdm(range(80, 141)):
    frame_a = img[i][400:900, 350:575]
    frame_b = img[i+1][400:900, 350:575]
    max_val = max([np.max(frame_a), np.max(frame_b)])

    frame_a = np.array(frame_a, dtype=np.float64) / max_val
    frame_b = np.array(frame_b, dtype=np.float64) / max_val

    # fig, axs = plt.subplots(1, 3, figsize=(12,10))
    # axs[0].imshow(frame_a)
    # axs[1].imshow(frame_b)
    # axs[2].imshow(frame_b-frame_a)
    # plt.show()

    winsize = 7
    searchsize = 8
    overlap = 4
    dt = 1

    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method='peak2peak',
        correlation_method="circular"
    )

    invalid_mask = validation.sig2noise_val(
        sig2noise,
        threshold = 1.05,
    )
    invalid_mask = np.array(invalid_mask)
    print(f"Percent invalid: {round(10000*len(np.where(invalid_mask)[0])/np.size(invalid_mask))/100}%")

    u2, v2 = filters.replace_outliers(
        u0, v0,
        invalid_mask,
        method='localmean',
        max_iter=3,
        kernel_size=3,
    )

    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=searchsize,
        overlap=overlap,
    )

    fig, ax = plt.subplots(figsize=(8,8))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    tools.display_vector_field_from_arrays(
        x=x,
        y=y,
        u=u2,
        v=v2,
        ax=ax,
        on_img=True,
        image_name=frame_a
    )
    idx_str = str(i)
    if i <= 10:
        idx_str = "00" + idx_str
    elif i <= 100:
        idx_str = "0" + idx_str
    fig.savefig("/home/S-CV/images/B02/"+idx_str+".tif", bbox_inches='tight', transparent=True, pad_inches=0, dpi=400)
