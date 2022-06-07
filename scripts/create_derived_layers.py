import numpy as np
import rasterio
from tqdm import tqdm
import copy
from dl_training_main import get_args

def calculate_correlation_coefficient(args):
    '''
    Calculate correlation coefficient between an imagery and rainfall stacks. If calculating between EVI and rainfall,
    a shifted rainfall stack is loaded; if calculated between NDWI and rainfall, the actual rainfall timeseries
    is loaded.

    User must specify the full_region, cropped_region to identify the area in question.
    The user must then specify which layer to calculate the correlation coefficient for.
    '''

    # Specify region + layer
    full_region = 'catalonia'
    cropped_region = 'catalonia'
    layer = 'ndwi'


    if layer == 'ndwi':
        shift = 'noshift'
    elif layer == 'evi':
        shift = 'shifted'

    # Define img + chirps filenames
    img_file = f'{args.base_dir}/imagery/{full_region}/modis/stacked_and_cropped/' \
               f'utm_projection/{cropped_region}_{layer}_250m_interp_smoothed_reproj.tif'
    chirps_file = f'{args.base_dir}/chirps/resampled_nn/' \
                  f'{cropped_region}_chirps_2017_2020_resampled_250m_modis_dates_{shift}_mm.tif'

    # Load
    with rasterio.open(img_file, 'r') as src:
        evi_img = src.read()
        evi_meta = src.meta


    with rasterio.open(chirps_file, 'r') as src:
        chirps_img = src.read()
        chirps_meta = src.meta

    # Transpose and take valid pixels
    evi_img = np.transpose(evi_img, (1, 2, 0))
    chirps_img = np.transpose(chirps_img, (1, 2, 0))

    evi_chirps_corrcoeff_map = np.full([evi_img.shape[0], evi_img.shape[1], 1], np.nan).astype(np.float32)
    evi_annual_corrcoeff_map = np.full([evi_img.shape[0], evi_img.shape[1], 1], np.nan).astype(np.float32)

    valid_pixels = ~((evi_img == evi_meta['nodata']).any(axis=-1))
    evi_img_valid_flat = evi_img[valid_pixels]
    chirps_img_valid_flat = chirps_img[valid_pixels]

    # Create layers for output
    evi_chirps_corrcoeff_map_flat = np.zeros((evi_img_valid_flat.shape[0], 1)).astype(np.float32)
    evi_annual_corrcoeff_map_flat = np.zeros((evi_img_valid_flat.shape[0], 1)).astype(np.float32)



    for i in tqdm(range(evi_img_valid_flat.shape[0])):
        ## Calculate CHIRPS EVI correlation coefficient
        a = evi_img_valid_flat[i]
        b = chirps_img_valid_flat[i]

        # Calculated EVI
        evi_chirps_coeff = np.corrcoef(a, b)
        evi_chirps_corrcoeff_map_flat[i] = evi_chirps_coeff[0][1]

        ## Extract evi annual signals
        y1 = evi_img_valid_flat[i, 0:23]
        y2 = evi_img_valid_flat[i, 23:46]
        y3 = evi_img_valid_flat[i, 46:69]

        # Calculate annual correlation coefficients
        annual_correlation_matrix = np.corrcoef(np.array([y1, y2, y3]))
        mean_annual_correlation = np.mean([annual_correlation_matrix[1,0],
                                           annual_correlation_matrix[2,0],
                                           annual_correlation_matrix[2,1]])

        evi_annual_corrcoeff_map_flat[i] = mean_annual_correlation

    # Reconsitute outputs in original shape
    evi_chirps_corrcoeff_map[valid_pixels] = evi_chirps_corrcoeff_map_flat
    evi_annual_corrcoeff_map[valid_pixels] = evi_annual_corrcoeff_map_flat

    evi_chirps_corrcoeff_map = np.transpose(evi_chirps_corrcoeff_map, (2,0,1))
    evi_annual_corrcoeff_map = np.transpose(evi_annual_corrcoeff_map, (2,0,1))

    # Save out
    evi_chirps_corrcoef_out =f'{args.base_dir}/imagery/{full_region}/derived_imagery/' \
                              f'{layer}_{shift}_chirps_corrcoeff_{cropped_region}_250m.tif'
    annual_corrcoef_out =     f'{args.base_dir}/imagery/{full_region}/derived_imagery/' \
                              f'{layer}_annual_corrcoeff_{cropped_region}_250m.tif'

    chirps_meta['count'] = 1
    chirps_meta['dtype'] = 'float32'
    chirps_meta['nodata'] = 'NaN'

    with rasterio.open(evi_chirps_corrcoef_out, 'w', **chirps_meta) as dest:
        dest.write(evi_chirps_corrcoeff_map)

    with rasterio.open(annual_corrcoef_out, 'w', **chirps_meta) as dest:
        dest.write(evi_annual_corrcoeff_map)



def calculate_layer_values_at_lowest_chirps_timesteps(args):
    '''
    Calculates the layer value at the N timesteps with the lowest CHIRPS values. If calculating for EVI ,
    a shifted rainfall stack is loaded; if calculated for NDWI, the actual rainfall timeseries
    is loaded.

    User must specify the full_region, cropped_region to identify the area in question.
    The user must then specify which layer to calculate layer values for.

    '''

    full_region = 'catalonia'
    cropped_region = 'catalonia'
    layer = 'ndwi'

    # Define num timesteps to consider
    num_timesteps_for_min_rainfall = [12, 24, 36]


    if layer == 'ndwi':
        shift = 'noshift'
    elif layer == 'evi':
        shift = 'shifted'

    # Load images
    img_file = f'{args.base_dir}/imagery/{full_region}/modis/stacked_and_cropped/' \
               f'utm_projection/{cropped_region}_{layer}_250m_interp_smoothed_reproj.tif'
    chirps_file = f'{args.base_dir}/chirps/resampled_nn/' \
                  f'{cropped_region}_chirps_2017_2020_resampled_250m_modis_dates_{shift}_mm.tif'

    with rasterio.open(img_file, 'r') as src:
        evi_img = src.read()
        evi_meta = src.meta

    with rasterio.open(chirps_file, 'r') as src:
        chirps_img = src.read()

    # Transpose and take valid pixels
    evi_img = np.transpose(evi_img, (1, 2, 0))
    chirps_img = np.transpose(chirps_img, (1, 2, 0))

    valid_pixels = ~((evi_img == evi_meta['nodata']).any(axis=-1)) * ~((chirps_img == 0).all(axis = -1))

    evi_img_valid_flat = evi_img[valid_pixels]
    chirps_img_valid_flat = chirps_img[valid_pixels]

    mean_evi_at_min_rainfall_timesteps_flat = np.zeros((evi_img_valid_flat.shape[0],
                                                   len(num_timesteps_for_min_rainfall)))
    mean_evi_at_min_rainfall_timesteps_map = np.full([evi_img.shape[0], evi_img.shape[1],
                                                      len(num_timesteps_for_min_rainfall)],
                                                np.nan).astype(np.float32)

    max_evi_at_min_rainfall_timesteps_flat = np.zeros((evi_img_valid_flat.shape[0],
                                                        len(num_timesteps_for_min_rainfall)))
    max_evi_at_min_rainfall_timesteps_map = np.full([evi_img.shape[0], evi_img.shape[1],
                                                      len(num_timesteps_for_min_rainfall)],
                                                np.nan).astype(np.float32)

    for ix, num_timesteps in enumerate(num_timesteps_for_min_rainfall):
        # Find layer values at min chirps timesteps
        for jx in tqdm(range(len(evi_img_valid_flat))):
            min_chirps_indices = chirps_img_valid_flat[jx].argsort()[0:num_timesteps]
            mean_evi_at_min_rainfall_timesteps_flat[jx, ix] = np.mean(evi_img_valid_flat[jx][min_chirps_indices])
            max_evi_at_min_rainfall_timesteps_flat[jx, ix]  = np.max(evi_img_valid_flat[jx][min_chirps_indices])

    # Reconstitute images
    mean_evi_at_min_rainfall_timesteps_map[valid_pixels] = mean_evi_at_min_rainfall_timesteps_flat
    mean_evi_at_min_rainfall_timesteps_map = np.transpose(mean_evi_at_min_rainfall_timesteps_map, (2,0,1))

    max_evi_at_min_rainfall_timesteps_map[valid_pixels] = max_evi_at_min_rainfall_timesteps_flat
    max_evi_at_min_rainfall_timesteps_map = np.transpose(max_evi_at_min_rainfall_timesteps_map, (2,0,1))


    # Save out
    timestep_str = ''.join([f'{str(i)}_' for i in num_timesteps_for_min_rainfall])

    mean_out_file = f'{args.base_dir}/imagery/{full_region}/derived_imagery/' \
            f'{layer}_mean_at_min_{timestep_str}{shift}_chirps_timesteps_{cropped_region}_250m.tif'

    max_out_file = f'{args.base_dir}/imagery/{full_region}/derived_imagery/' \
            f'{layer}_max_at_min_{timestep_str}{shift}_chirps_timesteps_{cropped_region}_250m.tif'

    evi_meta['count'] = len(num_timesteps_for_min_rainfall)
    evi_meta['dtype'] = 'float32'

    with rasterio.open(mean_out_file, 'w', **evi_meta) as dest:
        dest.write(mean_evi_at_min_rainfall_timesteps_map)

    with rasterio.open(max_out_file, 'w', **evi_meta) as dest:
        dest.write(max_evi_at_min_rainfall_timesteps_map)


def calculate_high_low_evi_ratio(args):
    '''
    Calculate the ratio of EVI from high-to-low percentiles.

    User must specify the full_region, cropped_region to identify the area in question.
    The user must then specify which layer to calculate the ratio for.
    '''

    # Specify regions -- Currently only calculating this ratio for EVI
    full_region = 'california'
    cropped_region = 'fresno'
    layer = 'evi'

    img_file = f'{args.base_dir}/imagery/{full_region}/modis/stacked_and_cropped/'\
               f'utm_projection/{cropped_region}_{layer}_250m_interp_smoothed_reproj.tif'

    # Load image file
    with rasterio.open(img_file, 'r') as src:
        evi_img = src.read()
        evi_meta = src.meta

    # Transpose and take valid pixels
    evi_img = np.transpose(evi_img, (1, 2, 0))
    valid_pixels = ~((evi_img == evi_meta['nodata']).any(axis=-1))
    evi_img_valid_flat = evi_img[valid_pixels]

    # Define percentiles
    percentiles = [5, 10, 15, 20, 80, 85, 90, 95]

    # Define array for writing out
    evi_max_min_ratio_map =  np.full([evi_img.shape[0], evi_img.shape[1], int(len(percentiles)/2)],
                                                 np.nan).astype(np.float32)

    # Calulate the percentiles of valid imagery
    evi_max_min_ratio_flat = np.percentile(evi_img_valid_flat, percentiles, axis = 1)

    # Calculate the EVI percentile ratio
    for i in tqdm(range(int(len(percentiles)/2))):

        ratio =  np.divide(evi_max_min_ratio_flat[len(percentiles)-i-1], evi_max_min_ratio_flat[i])
        evi_max_min_ratio_map[valid_pixels, i] = ratio

    # Write out
    out_file = f'{args.base_dir}/imagery/{full_region}/derived_imagery/' \
               f'{layer}_max_min_ratio_at_percentiles_{cropped_region}_250m.tif'

    evi_meta['count'] = int(len(percentiles)/2)
    evi_meta['dtype'] = 'float32'
    evi_meta['nodata'] = 'NaN'
    evi_max_min_ratio_map = np.transpose(evi_max_min_ratio_map, (2,0,1))
    evi_max_min_ratio_map = np.clip(evi_max_min_ratio_map, 0, 10)

    with rasterio.open(out_file, 'w', **evi_meta) as dest:
        dest.write(evi_max_min_ratio_map)


def stack_layers(args):
    '''
    Load derived layers and stack them. User must specify the full_region, cropped_region to identify the
    area in question.

    DO NOT change the order of these layers in the out file -- model training + testing depends on it.
    '''

    # Define regions and folder
    full_region = 'catalonia'
    cropped_region = 'catalonia'

    full_region_folder =  f'{args.base_dir}/imagery/{full_region}'
    derived_folder = f'{full_region_folder}/derived_imagery/'

    # Define derived layer files
    srtm_layer = f'{full_region_folder}/srtm/{cropped_region}_srtm_slope_250m_cropped_reproj.tif'

    evi_chirps_corrcoef_map = f'{derived_folder}/evi_shifted_chirps_corrcoeff_{cropped_region}_250m.tif'
    ndwi_chirps_corrcoef_map = f'{derived_folder}/ndwi_noshift_chirps_corrcoeff_{cropped_region}_250m.tif'

    evi_annual_corrcoef_map = f'{derived_folder}/evi_annual_corrcoeff_{cropped_region}_250m.tif'
    ndwi_annual_corrcoef_map = f'{derived_folder}/ndwi_annual_corrcoeff_{cropped_region}_250m.tif'

    mean_evi_at_min_chirps_timesteps =  f'{derived_folder}/evi_mean_at_min_12_24_36_shifted_chirps_timesteps_' \
                                        f'{cropped_region}_250m.tif'
    max_evi_at_min_chirps_timesteps =  f'{derived_folder}/evi_max_at_min_12_24_36_shifted_chirps_timesteps_' \
                                       f'{cropped_region}_250m.tif'

    mean_ndwi_at_min_chirps_timesteps = f'{derived_folder}/ndwi_mean_at_min_12_24_36_noshift_chirps_timesteps_' \
                                        f'{cropped_region}_250m.tif'
    max_ndwi_at_min_chirps_timesteps = f'{derived_folder}/ndwi_max_at_min_12_24_36_noshift_chirps_timesteps_' \
                                       f'{cropped_region}_250m.tif'

    evi_max_min_ratio_percentiles = f'{derived_folder}/evi_max_min_ratio_at_percentiles_{cropped_region}_250m.tif'



    # Load layers
    print('Loading layers')
    with rasterio.open(srtm_layer, 'r') as src:
        srtm_img = src.read().astype(np.float32)
        srtm_img[np.where(srtm_img == src.meta['nodata'])] = np.nan

    with rasterio.open(evi_chirps_corrcoef_map, 'r') as src:
        evi_chirps_corrcoef_img = src.read()
        metadata = copy.deepcopy(src.meta)

    with rasterio.open(ndwi_chirps_corrcoef_map, 'r') as src:
        ndwi_chirps_corrcoef_img = src.read()

    with rasterio.open(evi_annual_corrcoef_map, 'r') as src:
        evi_annual_corrcoef_img = src.read()

    with rasterio.open(ndwi_annual_corrcoef_map, 'r') as src:
        ndwi_annual_corrcoef_img = src.read()

    with rasterio.open(mean_evi_at_min_chirps_timesteps, 'r') as src:
        mean_evi_at_min_chirps_img = src.read()

    with rasterio.open(max_evi_at_min_chirps_timesteps, 'r') as src:
        max_evi_at_min_chirps_img = src.read()

    with rasterio.open(mean_ndwi_at_min_chirps_timesteps, 'r') as src:
        mean_ndwi_at_min_chirps_img = src.read()

    with rasterio.open(max_ndwi_at_min_chirps_timesteps, 'r') as src:
        max_ndwi_at_min_chirps_img = src.read()

    with rasterio.open(evi_max_min_ratio_percentiles, 'r') as src:
        evi_max_min_ratio = src.read()

    # Stack layers
    image_stack = np.concatenate((srtm_img,

                                  evi_annual_corrcoef_img, evi_chirps_corrcoef_img,
                                  mean_evi_at_min_chirps_img,
                                  max_evi_at_min_chirps_img,
                                  evi_max_min_ratio,

                                  ndwi_annual_corrcoef_img, ndwi_chirps_corrcoef_img,
                                  mean_ndwi_at_min_chirps_img,
                                  max_ndwi_at_min_chirps_img,

                                  ),
                                 axis=0).astype(np.float32)


    # Write out
    image_stack_out = f'{full_region_folder}/stacked_images_for_classification/{cropped_region}_srtm_evi_ndwi.tif'

    metadata['count'] = image_stack.shape[0]
    metadata['nodata'] = 'NaN'

    with rasterio.open(image_stack_out, 'w', **metadata) as dest:
        dest.write(image_stack)


if __name__ == '__main__':
    args = get_args()


    calculate_layer_values_at_lowest_chirps_timesteps(args)
    stack_layers(args)

