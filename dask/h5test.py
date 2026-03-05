import os
import dask
from dask.distributed import Client, LocalCluster
import dask.distributed
from pathml.core import HESlide
from pathml.preprocessing import Pipeline
from pathml.preprocessing import (
    TissueDetectionHE,
    LabelWhiteSpaceHE,
    LabelArtifactTileHE,
    StainNormalizationHE,
)
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import time

print("Inside python script")
if __name__ == "__main__":
    print("Inside __main__")
    scratch = os.getenv("SLURM_SCRATCH")
    sp = Path(scratch)
    print(sp)
    dask.config.set({"distributed.comm.timeouts.connect": "120s"})
    dask.config.set({"distributed.comm.timeouts.tcp": "120s"})
    dask.config.set(
        {"temporary_directory": str(sp.joinpath("dask-worker-space"))}
    )  # Crucial for reducing timeout issues
    cluster = LocalCluster(
        n_workers=4,  # 32 works well, 64 --> maxes things out and very slow on 1 node
        threads_per_worker=16,
        memory_limit="32g",
        dashboard_address=":8788",
    )  # This works to set dask port, connect locally with ssh tunnel:
    # ssh -N -f -L 8789:NODENAME.crc.pitt.edu:8789 bri8@htc.crc.pitt.edu and nav to http://127.0.0.1:8789/status

    client = Client(cluster)

    data = Path("/path/to/project/")
    results = Path("/ipath/to/results")
    fn = sp.joinpath("sample.svs")
    print(fn)
    print("exists", fn.exists())
    wsi = HESlide(fn)
    blank_detect = LabelWhiteSpaceHE(
        label_name="ignore", proportion_threshold=0.9
    )  # Thresh too low?
    art_detect = LabelArtifactTileHE(label_name="ignore")
    tissue_detect = TissueDetectionHE(
        mask_name="tissue", outer_contours_only=True, blur_ksize=21, threshold=20
    )
    normalize = StainNormalizationHE(
        target="normalize", stain_estimation_method="macenko"
    )
    print(client.dashboard_link, os.getenv("SLURMD_NODENAME"))

    a = time.time()
    pipeline = Pipeline([blank_detect, art_detect, tissue_detect, normalize])
    wsi.run(
        pipeline,
        tile_size=500,
        distributed=True,
        # overwrite_existing_tiles =True,
        client=client,
    )
    print(f"Total number of tiles extracted: {len(wsi.tiles)}")
    wsi.write(results.joinpath("test.h5path"))
    b = time.time()
    print((b - a) / 60, "minutes")
