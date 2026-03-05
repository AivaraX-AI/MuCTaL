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
