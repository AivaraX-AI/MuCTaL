import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import GroupShuffleSplit

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from sklearn.utils import check_random_state


def sample_from_different_slides(n_samples, groups, random_state=None):
    rng = check_random_state(random_state)
    unique_groups = np.unique(groups)

    if len(unique_groups) < n_samples:
        print(
            f"Warning not enough groups ({len(unique_groups)}) to sample {n_samples} examples."
        )
        replace = True
    else:
        replace = False

    selected_groups = rng.choice(unique_groups, size=n_samples, replace=replace)
    sampled_indices = []

    for group in selected_groups:
        group_indices = np.where(groups == group)[0]
        sampled_indices.append(rng.choice(group_indices))

    return np.array(sampled_indices)


def plot_tile_grid(
    data,
    image_dir,
    clusters=["0"],
    feat="louvain_0.10",
    grid_size=(4, 4),
    figsize=(7, 7),
    save_figs=False,
    save_dir="",
    method="adata",  # if method is 'df' treat '
):

    # Get list of image files
    if method == "adata":
        adata_obs = data.obs
    else:
        adata_obs = data

    figs = []
    for i, cluster in enumerate(clusters):
        all_ex = adata_obs.loc[adata_obs[feat] == cluster].index.values
        print(len(all_ex), "for cluster", cluster)
        n_samples = grid_size[0] * grid_size[1]
        # Create a figure with subplots
        fig, axes = plt.subplots(
            nrows=grid_size[0],
            ncols=grid_size[1],
            layout="constrained",
            figsize=figsize,
        )

        axes = axes.flatten()

        # Plot each image in the grid
        j = 0
        use = []
        keep_slide = []
        max_tries = 6
        tries = 0
        while len(use) < n_samples:
            exs = all_ex[
                sample_from_different_slides(n_samples, adata_obs.loc[all_ex, "slide"])
            ]
            j = 0
            for j in range(0, n_samples):
                ex0 = exs[j]
                slide = adata_obs.loc[ex0, "slide"]
                ex = ex0.split(".jpg")[0] + ".jpg"
                img_path = image_dir.joinpath(slide, ex)
                if img_path.exists():
                    use.append(img_path)
                    keep_slide.append(slide)
            tries = tries + 1
        print(f"{tries} tries")
        for j in range(0, n_samples):
            img_path = use[j]
            ax = axes[j]
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off")  # Hide axes
            ax.set_title(keep_slide[j].split("-")[2])

        # Adjust layout and show the plot
        plt.suptitle(f"Cluster {cluster} examples ({feat} )")
        if save_figs:
            fn = f"cluster_{cluster}_examples_{feat}.png"
            pnfn = Path(save_dir, fn)
            print(pnfn)
            fig.savefig(pnfn)  # Save the plot to a file
            plt.close()  # Close the figure to free up memory
        else:
            # plt.tight_layout()
            figs.append(fig)
    return figs


def ds_img_from_wsi(wsi_fn, nchunks, ds, verbose=False):
    blank_detect = LabelWhiteSpaceHE(
        label_name="blank",
        proportion_threshold=0.99,
    )
    art_detect = LabelArtifactTileHE(label_name="artifact")
    wsi = HESlide(str(wsi_fn))
    x, y = wsi.shape
    tile_size = y // nchunks
    tot_tiles = (x // tile_size) * (y // tile_size)
    print(x, y, tile_size, tot_tiles)
    print("original slide shape:", x, y)
    dsx = x // ds
    dsy = y // ds
    print("ds shape:", dsx, dsy)
    print("tile size:", tile_size, "total tiles:", tot_tiles)
    blank_image = np.zeros((dsx, dsy, 3), np.uint8) + 255
    blank_tot = 0
    img_tot = 0
    for i, tile in enumerate(wsi.generate_tiles(shape=tile_size, pad=False)):
        blank_detect.apply(tile)
        if tile.labels["blank"] == False:
            # art_detect.apply(tile)
            # if tile.labels['artifact'] == False:
            if verbose:
                print("Loading tile %d" % i)
            im = np.array(tile.image)
            img_tot += 1
            xx, yy = tile.coords
            imds = cv2.resize(
                im, (tile_size // ds, tile_size // ds), interpolation=cv2.INTER_CUBIC
            )
            dsxx = xx // ds
            dsyy = yy // ds

            blank_image[
                dsxx : (dsxx + imds.shape[1]), dsyy : (dsyy + imds.shape[0])
            ] = imds
        else:
            blank_tot += 1
            if verbose:
                print("Tile %d detected as blank... skipping" % i)
    print("%d images loaded, %d detected as blank" % (img_tot, blank_tot))
    # blank_image = cv2.cvtColor(blank_image,cv2.COLOR_RGB2GRAY)
    return blank_image
