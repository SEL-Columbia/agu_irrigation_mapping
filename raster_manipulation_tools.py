import numpy as np
import rasterio
import glob
from scipy.interpolate import CubicSpline
import datetime
from dateutil.rrule import rrule, MONTHLY
from tqdm import tqdm
import geopandas as gpd
from rasterio.mask import mask
import copy
from rasterio.warp import reproject, Resampling


def crop_tif(input_tif, input_shp_name):
    '''
    Function for cropping an input tif to a input shape file. The input tif is clipped to the first polygon in the
    shapefile -- only meant to be used when with shapefile with a single polygon.

    :param input_tif: Input tif filename.
    :param input_shp_name: Input shapefile filename.
    :return: Nothing; Saves cropped extent to file.
    '''

    polygon_shp = gpd.read_file(input_shp_name)


    with rasterio.open(input_tif, 'r') as src:
        meta = src.meta


        polygon_shp = polygon_shp.to_crs(meta['crs'])
        polygon_shp = polygon_shp['geometry'].iloc[0]
        img, out_transform = mask(src, [polygon_shp], crop=True)

    meta.update({"driver": "GTiff",
                 "height": img.shape[1],
                 "width": img.shape[2],
                 "transform": out_transform})

    print('Write out')
    out_file_name = input_tif.replace('.tif', '_cropped.tif')
    with rasterio.open(out_file_name, "w", **meta) as dest:
        dest.write(img)



def reproject_tif(source_tif, tif_with_dest_projection):
    '''
    Function for cropping an input tif to another tif with a different projection.

    :param source_tif: Input tif filename to be reprojected.
    :param tif_with_dest_projection: Tif with the desired projection.
    :return: Nothing; Saves reprojected image to file.
    '''
    with rasterio.open(source_tif, 'r') as src:
        src_meta = copy.deepcopy(src.meta)

        with rasterio.open(tif_with_dest_projection, 'r+') as proj_src:
            metadata = copy.deepcopy(proj_src.meta)
            metadata['count'] = src_meta['count']
            metadata['nodata'] = src_meta['nodata']
            metadata['dtype']  = src_meta['dtype']

            out_file = source_tif.replace('.tif', '_reproj.tif')
            with rasterio.open(out_file, 'w+', **metadata) as dest_tif:
                for i in range(1, metadata['count'] + 1):
                    reproject(source=rasterio.band(src, i),
                              destination=rasterio.band(dest_tif, i),
                              resampling=Resampling.nearest)

def create_ndwi_images(nir_file, swir_file):
    '''
    Function for creating a NDWI image from a NIR file and SWIR file.

    :param nir_file: Path to NIR file.
    :param swir_file: Path to SWIR file
    :return: Nothing; Saves SWIR to file.
    '''

    # Load images
    with rasterio.open(nir_file, 'r') as src:
        nir_img = src.read()
        meta = src.meta

    with rasterio.open(swir_file, 'r') as src:
        swir_img = src.read()

    # Transpose and clip
    nir_img = np.clip(np.transpose(nir_img, (1, 2, 0)), 0, np.inf)
    swir_img = np.clip(np.transpose(swir_img, (1,2,0)), 0, np.inf)

    invalid_perc = 0.25

    # Find valid pixels
    valid_pixels = np.where((np.count_nonzero(nir_img == meta['nodata'], axis=-1) < invalid_perc) *
                            (np.count_nonzero(swir_img == meta['nodata'], axis=-1) < invalid_perc) *
                            (np.count_nonzero(nir_img == 0, axis=-1) < invalid_perc))

    # Extract timeseries at valid pixel locations
    nir_img_flat = nir_img[valid_pixels]
    swir_img_flat= swir_img[valid_pixels]

    # Create flattend NDWI array
    ndwi_flat = np.nan_to_num(np.divide(np.subtract(nir_img_flat, swir_img_flat), np.add(nir_img_flat, swir_img_flat)),
                              nan=meta['nodata'])

    # Reconstitute NDWI layer, scaling by 10000
    ndwi = (np.ones(swir_img.shape)* meta['nodata']).astype(np.int16)
    ndwi[valid_pixels] = (10000*ndwi_flat).astype(np.int16)


    # Write out. Warning: this will only work if 'nir' is contained in the NIR file path name.
    out_file = nir_file.replace('_nir_', '_swir_')
    ndwi = np.transpose(ndwi, (2,0,1))

    with rasterio.open(out_file, 'w', **meta) as dest:
        dest.write(ndwi)



def reproject_500m_images(file_to_reproj, dest_file):
    '''
    Reproject 500m images (SWIR) to 250m (EVI, NIR).

    :param file_to_reproj: Path to 500m image to be reprojected.
    :param dest_file: Path to image containing 250m projection.
    :return: Nothing; saves reprojected image to file.
    '''


    out_file = file_to_reproj.replace('_500m', '_250m')

    with rasterio.open(dest_file, 'r') as src:
        metadata = copy.deepcopy(src.meta)

    with rasterio.open(file_to_reproj, 'r') as src:
        with rasterio.open(out_file, 'w', **metadata) as dest_tif:
            for ix in tqdm(range(1, src.meta['count']+1)):

                reproject(source=rasterio.band(src, ix),
                          destination=rasterio.band(dest_tif, ix),
                          warp_mem_limit=1000)


def stack_individual_layers(input_file_dir, out_file):
    '''
    Function for stacking (temporally/depthwise) successive imagery layers.
    :param input_file_dir: Directory containing input layers to be stacked.
    :param out_file: Output file name for stacked array.
    :return: Nothing; saves stacked image to file.
    '''

    input_files = glob.glob(input_file_dir + '/*.tif')
    src_meta = rasterio.open(input_files[0], 'r').meta

    # Currently set to 69 for 3 years of MODIS data (23 time samples per year).
    src_meta['count'] = 69

    out_array = np.zeros((69, src_meta['height'], src_meta['width'])).astype(src_meta['dtype'])

    # Stack and write out
    with rasterio.open(out_file, 'w', **src_meta) as dest:
        for ix, file in enumerate(input_files):
            print(ix)
            with rasterio.open(file, 'r') as src:
                out_array[ix] = src.read()
        print('Write out')
        dest.write(out_array)


def crop_and_stack_monthly_chirps_images(chirps_dir, shape_file, out_file):
    '''
    Function from cropping and stacking monthly CHIRPS images.

    :param chirps_dir: Directory containing the individual (for a single timestep) CHIRPS images.
    :param shape_file: Path to shapefile for cropping the CHIRPS images
    :param out_file: Path to the file for export.
    :return: Nothing; saves stacked image to file.
    '''

    # Load all saved CHIRPS files
    all_files = glob.glob(chirps_dir + '/*.tif')

    # Load shapefile and reproject to EPSG 4326 for CHIRPS, take 1st (hopefully only!) polygon in the shapefile.
    shapefile_mask = gpd.read_file(shape_file).to_crs('EPSG:4326')
    shapefile_mask = shapefile_mask['geometry'].iloc[0]

    print('Crop and save average monthly rainfall tif')
    rainfall_imagery_list = []
    for ix, file in enumerate(all_files):
        with rasterio.open(file, 'r') as src:
            masked_image, trans = mask(src, [shapefile_mask], crop=True, nodata = -3000)
            chirps_meta = src.meta
            rainfall_imagery_list.append(masked_image)

    chirps_meta.update({'count': '36', 'height': rainfall_imagery_list[0].shape[1],
                        'width': rainfall_imagery_list[0].shape[2], 'nodata': '-3000', 'transform': trans,
                        'dtype':'int16'})
    cropped_stacked_rainfall = np.concatenate(rainfall_imagery_list, axis=0)


    # Write out the average monthly rainfall tif
    with rasterio.open(out_file, 'w', **chirps_meta) as dest:
        dest.write(cropped_stacked_rainfall)


def load_chirps_and_resample(chirps_stack_file, destination_tif, modis_imgs_folder, out_file, shift):
    '''
    Function to load stacked chirps and resample spatially to a destination projection and temporally to MODIS dates.

    :param chirps_stack_file: CHIRPS rainfall stack to reproject. This is the file that is saved by the
    crop_and_stack_monthly_chirps_images() function.
    :param destination_tif: Path to file containing the destination spatial projection
    :param modis_imgs_folder: Directory containing MODIS images to take the DOYs for temporal interpolation.
    :param out_file: File to save the reprojected image.
    :param shift: Whether the resampled CHIRPS file should be shifted by a month.
    :return: Nothing; saves reprojected image to file.
    '''

    # Load files
    with rasterio.open(destination_tif, 'r') as src:
        dest_img = src.read()
        dest_meta = src.meta

    with rasterio.open(chirps_stack_file, 'r') as src:
        chirps_img = src.read()
        chirps_meta = src.meta

    # Pad for temporal interpolation
    rainfall_map_padded = np.concatenate((np.expand_dims(chirps_img[-1], 0), chirps_img), axis = 0)

    ## Interpolate rainfall to MODIS dates
    strt_dt = datetime.date(2017, 1, 15)
    end_dt = datetime.date(2019, 12, 15)

    # Find dates for the rainfall predictions
    start_doy = strt_dt.timetuple().tm_yday
    rainfall_ordinal_dates = [(dt.date() - strt_dt).days + start_doy for dt in rrule(MONTHLY, dtstart=strt_dt,
                                                                                     until=end_dt)]
    rainfall_ordinal_dates = np.insert(rainfall_ordinal_dates, 0, rainfall_ordinal_dates[0] - 31)


    # Return the dates for the MODIS images
    modis_dates = [i.split('_')[-1].replace('.tif', '').replace('doy', '') for i in
                   sorted(glob.glob(modis_imgs_folder + '/*.tif'))]  # starts September 14
    modis_dates = [(int(i[0:4]), int(i[4:7])) for i in modis_dates]

    modis_ordinal_dates = [(i[0] - 2017) * 365 + i[1] for i in modis_dates]
    interpolated_rainfall = np.zeros((69, chirps_img.shape[1], chirps_img.shape[2]))


    # Interpolate temporally using a cubic spline
    for i in tqdm(range(chirps_img.shape[1])):
        for j in range(chirps_img.shape[2]):
            cs = CubicSpline(rainfall_ordinal_dates, rainfall_map_padded[:,i,j])
            interpolated_rainfall[:, i, j] = cs(modis_ordinal_dates)

    # Reproject spatially
    resampled_rainfall = np.zeros((69, dest_img.shape[1], dest_img.shape[2]))

    print('Reprojecting')
    reproject(
        interpolated_rainfall,
        resampled_rainfall,
        src_transform=chirps_meta['transform'],
        src_crs = chirps_meta['crs'],
        dst_transform=dest_meta['transform'],
        dst_crs=dest_meta['crs'],
        resampling = Resampling.nearest
    )


    # Reconsitute image for saving
    resampled_rainfall = resampled_rainfall.astype(np.int16)
    resampled_rainfall = np.transpose(resampled_rainfall, (1,2,0))
    invalid_pixels = np.where(np.count_nonzero(resampled_rainfall == 0, axis=-1) == resampled_rainfall.shape[-1])
    resampled_rainfall[invalid_pixels] = chirps_meta['nodata']
    resampled_rainfall = np.transpose(resampled_rainfall, (2,0,1))

    # Shift the resampled rainfall image if desired.
    if shift:
        resampled_rainfall_shifted = np.zeros(resampled_rainfall.shape).astype(np.int16)
        resampled_rainfall_shifted[2::] = resampled_rainfall[0:-2]
        resampled_rainfall_shifted[0:2] = resampled_rainfall[-2::]

        resampled_rainfall = resampled_rainfall_shifted

    # Save out
    dest_meta['nodata'] = chirps_meta['nodata']

    with rasterio.open(out_file, 'w', **dest_meta) as dest:
        dest.write(resampled_rainfall)



if __name__ == '__main__':
    print('Run functions here')
