import numpy as np
import rasterio
from tqdm import tqdm
import geopandas as gpd
from scipy.ndimage import convolve
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def dilate(map_layer, iters):
    '''
    Function takes in a map_layer of predictions and dilates the positive predictions

    :param map_layer: input full of 0/1 binary predictions
    :param iters: Number of times to dilate
    :return: Dilated map_layer
    '''


    kernel = np.array([[0., 1., 0.],
                        [ 1., 1., 1.],
                        [ 0., 1., 0.]])

    for i in range(iters):
        map_layer = map_layer * 1.0
        map_layer = convolve(map_layer, kernel) > 0



    return map_layer


def erode(map_layer, iters):
    '''
    Function takes in a map_layer of predictions and erodes the positive predictions

    :param map_layer: input full of 0/1 binary predictions
    :param iters: Number of times to erode
    :return: Eroded map_layer
    '''

    map_layer = map_layer.astype(int)
    map_layer = ~map_layer

    kernel = np.array([[0., 1., 0.],
                        [ 1., 1., 1.],
                        [ 0., 1., 0.]])

    for i in range(iters):
        map_layer = map_layer * 1.0
        map_layer = convolve(map_layer, kernel) > 0

    map_layer = ~map_layer

    return map_layer


def interpolate_and_smooth(in_file):
    '''
    Function takes in a 3D imagery stack name, interpolates depthwise (temporally), and then smooths and saves the output.
    :param in_file: 3D imagery stack for interpolation and smoothing
    :return: Nothing; Saves an interpolated and smoothed output
    '''

    # Load imagery
    with rasterio.open(in_file, 'r') as src:
        img = src.read()
        meta = src.meta

    # Move temporal axis to last index
    img = np.transpose(img, (1, 2, 0))

    # Flatten
    img_flat = img.reshape(img.shape[0] * img.shape[1], img.shape[2])


    # Valid locations in original image
    valid_pixels = np.where((np.count_nonzero(img == meta['nodata'], axis=-1) +
                             np.count_nonzero(np.isnan(img), axis = -1)) < 0.25 * img.shape[-1])

    # Valid timeseries in flattened image
    img_flat_valid = img_flat[(np.count_nonzero(img_flat == meta['nodata'], axis=-1) +
                               np.count_nonzero(np.isnan(img_flat), axis=-1)) < 0.25 * img_flat.shape[-1]]

    # Interpolate depthwise for all pixels with valid data
    for ix in tqdm(range(img_flat_valid.shape[0])):
        valid_timesteps = np.argwhere(np.logical_and(img_flat_valid[ix] != meta['nodata'],
                                                     ~np.isnan(img_flat_valid[ix]))).flatten()
        # Interpolate if there are fewer valid timesteps than the overall number of timesteps
        if len(valid_timesteps) < img_flat_valid.shape[1]:
            f = interp1d(valid_timesteps, img_flat_valid[ix][valid_timesteps], kind='linear', fill_value='extrapolate')
            img_flat_valid[ix] = f(range(img_flat_valid.shape[1]))


    # Apply savgol filter
    #
    img_smoothed = np.zeros(img_flat_valid.shape)
    for ix in tqdm(range(img_flat_valid.shape[0])):
        img_smoothed[ix] = savgol_filter(img_flat_valid[ix], 7, 3)



    # Fill invalid pixel locations with 'nodata' value
    img.fill(meta['nodata'])
    # Reassign interpolated pixels to their original locations
    img[valid_pixels] = img_smoothed
    img = np.transpose(img, (2, 0, 1))

    out_file = in_file.replace('.tif', '') + '_interp_smoothed.tif'
    with rasterio.open(out_file, 'w', **meta) as dest:
        dest.write(img)


def vectorize(in_file):
    '''
    Load in a tif file with irrigation predictions and vectorize. Save polygonized output as geojson.

    :param in_file: tif file to be vectorized
    '''

    # Load imagery and meta data
    with rasterio.open(in_file, 'r+') as src:
        img = src.read()
        img[np.isnan(img)] = 0

        img = img.astype(np.uint16)

        transform = src.transform
        crs = src.crs

    # Filter groups of predictions
    out_array = rasterio.features.sieve(img[0], size = 16)

    # Polygonize
    polygons = rasterio.features.shapes(out_array, transform=transform)

    # Create list of geoms in dictionary format for output
    geoms = [
        {"properties": {"raster_val": v}, "geometry": s, "type": "feature"}
        for s, v in polygons if v == 1
    ]

    # Write out
    out_file = in_file.replace('.tif', '_polygonized.geojson')
    geoms_gdf = gpd.GeoDataFrame.from_features(geoms)
    geoms_gdf.crs = crs
    geoms_gdf.to_file(out_file, driver="GeoJSON")


if __name__ == '__main__':
    vectorize()

