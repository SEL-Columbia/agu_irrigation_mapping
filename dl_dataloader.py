import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.mask import mask
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import pandas as pd
import seaborn as sns
import tensorflow as tf
import csv

class DataGenerator():
    'This selects and prepares training and testing data'
    def __init__(self, columns_to_use, dir_time):

        self.batch_size = 64

        self.input_tif = '/Volumes/sel_external/ethiopia_vegetation_detection/imagery/{}/stacked_images_for_classification/' \
                         '{}_srtm_evi_ndwi_amap.tif'
        self.pixels_within_shapefile_dict = {}

        catalonia_irrig_array, catalonia_noirrig_array, fresno_irrig_array, fresno_noirrig_array, \
        amhara_irrig_array, amhara_noirrig_array, uganda_irrig_array, uganda_noirrig_array = self.return_selected_pixels_for_training()

        print('Load Amhara')
        amhara_irrig_valid_pixels, amhara_noirrig_valid_pixels, amhara_standard_array = \
            self.return_pixel_data('ethiopia', 'amhara', amhara_irrig_array, amhara_noirrig_array, columns_to_use)

        print('Load Catalonia')
        catalonia_irrig_valid_pixels, catalonia_noirrig_valid_pixels, catalonia_standard_array = \
            self.return_pixel_data('catalonia', 'catalonia', catalonia_irrig_array, catalonia_noirrig_array, columns_to_use)

        print('Load Fresno')
        fresno_irrig_valid_pixels, fresno_noirrig_valid_pixels, fresno_standard_array = \
            self.return_pixel_data('california', 'fresno', fresno_irrig_array, fresno_noirrig_array, columns_to_use)

        # print('Load Uganda')
        # uganda_irrig_valid_pixels, uganda_noirrig_valid_pixels, uganda_standard_array = \
        #     self.return_pixel_data('uganda', 'uganda', uganda_irrig_array, uganda_noirrig_array, columns_to_use)

        train_val_amhara_irrig_array, self.test_amhara_irrig_array = train_test_split(amhara_irrig_valid_pixels, test_size=0.1)
        self.train_amhara_irrig_array, self.val_amhara_irrig_array = train_test_split(train_val_amhara_irrig_array, test_size=0.2)

        train_val_amhara_noirrig_array, self.test_amhara_noirrig_array = train_test_split(amhara_noirrig_valid_pixels, test_size=0.1)
        self.train_amhara_noirrig_array, self.val_amhara_noirrig_array = train_test_split(train_val_amhara_noirrig_array, test_size=0.2)

        train_val_catalonia_irrig_array, self.test_catalonia_irrig_array = train_test_split(catalonia_irrig_valid_pixels, test_size=0.1)
        self.train_catalonia_irrig_array, self.val_catalonia_irrig_array = train_test_split(train_val_catalonia_irrig_array, test_size=0.2)

        train_val_catalonia_noirrig_array, self.test_catalonia_noirrig_array = train_test_split(catalonia_noirrig_valid_pixels, test_size=0.1)
        self.train_catalonia_noirrig_array, self.val_catalonia_noirrig_array = train_test_split(train_val_catalonia_noirrig_array, test_size=0.2)

        train_val_fresno_irrig_array, self.test_fresno_irrig_array = train_test_split(fresno_irrig_valid_pixels, test_size=0.1)
        self.train_fresno_irrig_array,  self.val_fresno_irrig_array = train_test_split(train_val_fresno_irrig_array, test_size=0.2)

        train_val_fresno_noirrig_array, self.test_fresno_noirrig_array = train_test_split(fresno_noirrig_valid_pixels, test_size=0.1)
        self.train_fresno_noirrig_array, self.val_fresno_noirrig_array = train_test_split(train_val_fresno_noirrig_array, test_size=0.2)

        # Shuffle
        np.random.shuffle(self.train_amhara_irrig_array)
        np.random.shuffle(self.train_amhara_noirrig_array)
        np.random.shuffle(self.train_catalonia_irrig_array)
        np.random.shuffle(self.train_catalonia_noirrig_array)
        np.random.shuffle(self.train_fresno_irrig_array)
        np.random.shuffle(self.train_fresno_noirrig_array)

        # Calculate min pixels per region for balanced training
        self.min_amhara_pixels = np.min((len(self.train_amhara_irrig_array), len(self.train_amhara_noirrig_array)))
        self.min_catalonia_pixels = np.min((len(self.train_catalonia_irrig_array), len(self.train_catalonia_noirrig_array)))
        self.min_fresno_pixels = np.min((len(self.train_fresno_irrig_array), len(self.train_fresno_noirrig_array)))
        self.min_all_regions = np.min((self.min_amhara_pixels, self.min_catalonia_pixels, self.min_fresno_pixels))


        self.max_amhara_pixels = np.max((len(self.train_amhara_irrig_array), len(self.train_amhara_noirrig_array)))
        self.max_catalonia_pixels = np.max(
            (len(self.train_catalonia_irrig_array), len(self.train_catalonia_noirrig_array)))
        self.max_fresno_pixels = np.max((len(self.train_fresno_irrig_array), len(self.train_fresno_noirrig_array)))

        self.training_rate_dict = {}

        self.training_rate_dict['amhara'] = self.min_all_regions/self.max_amhara_pixels
        self.training_rate_dict['catalonia'] = self.min_all_regions/self.max_catalonia_pixels
        self.training_rate_dict['fresno'] = self.min_all_regions/self.max_fresno_pixels


        # Define training and validation arrays
        self.data_array_dict = {'train_amhara_irrig': self.train_amhara_irrig_array,
                                'train_amhara_noirrig': self.train_amhara_noirrig_array,
                                'train_catalonia_irrig': self.train_catalonia_irrig_array,
                                'train_catalonia_noirrig': self.train_catalonia_noirrig_array,
                                'train_fresno_irrig': self.train_fresno_irrig_array,
                                'train_fresno_noirrig': self.train_fresno_noirrig_array,

                                'val_amhara_irrig': self.val_amhara_irrig_array,
                                'val_amhara_noirrig': self.val_amhara_noirrig_array,
                                'val_catalonia_irrig': self.val_catalonia_irrig_array,
                                'val_catalonia_noirrig': self.val_catalonia_noirrig_array,
                                'val_fresno_irrig': self.val_fresno_irrig_array,
                                'val_fresno_noirrig': self.val_fresno_noirrig_array,

                                'test_amhara_irrig': self.test_amhara_irrig_array,
                                'test_amhara_noirrig': self.test_amhara_noirrig_array,
                                'test_catalonia_irrig': self.test_catalonia_irrig_array,
                                'test_catalonia_noirrig': self.test_catalonia_noirrig_array,
                                'test_fresno_irrig': self.test_fresno_irrig_array,
                                'test_fresno_noirrig': self.test_fresno_noirrig_array,
                                }

        self.standard_array_dict = {'amhara_standard_array': amhara_standard_array,
                                    'catalonia_standard_array': catalonia_standard_array,
                                    'fresno_standard_array': fresno_standard_array,
                                    # 'uganda_standard_array': uganda_standard_array
                                    }

        self.visual_truth_arrays = {'catalonia_irrig_vt': catalonia_irrig_array,
                                    'catalonia_noirrig_vt': catalonia_noirrig_array,
                                    'fresno_irrig_vt': fresno_irrig_array,
                                    'fresno_noirrig_vt': fresno_noirrig_array,
                                    'amhara_irrig_vt': amhara_irrig_array,
                                    'amhara_noirrig_vt': amhara_noirrig_array
                                    }

        self.save_norm_file = True
        if self.save_norm_file:
            dict_for_saving = {}
            for k, v in self.standard_array_dict.items():
                dict_for_saving[k] = v.tolist()
            out_df = pd.DataFrame.from_dict(dict_for_saving, orient='columns')
            out_df.to_csv(f'files_for_prediction/normalization_arrays/{dir_time}.csv')

    def return_pixel_data(self, full_region, cropped_region, irrig_raster, noirrig_raster, columns_to_use):
        # For return_polygon_pixels, see old_code.py

        input_tif = self.input_tif.format(full_region, cropped_region)

        lulc_map = self.load_lulc_pixels(cropped_region)


        with rasterio.open(input_tif, 'r') as src:
            img = np.transpose(src.read(), (1, 2, 0))

            norm_pixels_flat = img[np.where(lulc_map)]

            irrig_ts_flat = img[np.where(irrig_raster)]
            nonirrig_ts_flat = img[np.where(noirrig_raster)]


            pixels_within_shapefile = ~np.any(np.isnan(img), axis=-1)


        self.pixels_within_shapefile_dict[cropped_region] = pixels_within_shapefile


        # Clean NaNs
        norm_pixels_flat = norm_pixels_flat[~np.any(np.isnan(norm_pixels_flat), axis=1)]
        irrig_ts_flat = irrig_ts_flat[~np.any(np.isnan(irrig_ts_flat), axis=1)]
        nonirrig_ts_flat = nonirrig_ts_flat[~np.any(np.isnan(nonirrig_ts_flat), axis=1)]

        # Take only select columns
        norm_pixels_flat = norm_pixels_flat[:, columns_to_use]
        irrig_ts_flat = irrig_ts_flat[:, columns_to_use]
        nonirrig_ts_flat = nonirrig_ts_flat[:, columns_to_use]

        # Calculate mean + std for normalization
        band_means = np.nanmean(norm_pixels_flat, axis=0)
        band_std = np.nanstd(norm_pixels_flat, axis=0)

        standard_array = np.stack([band_means, band_std], axis=0)

        irrig_ts_flat = (irrig_ts_flat - band_means) / band_std
        nonirrig_ts_flat = (nonirrig_ts_flat - band_means) / band_std


        return irrig_ts_flat, nonirrig_ts_flat, standard_array

    def load_lulc_pixels(self, cropped_region):
        in_file = f'/Volumes/sel_external/ethiopia_vegetation_detection/copernicus_lc_maps/{cropped_region}_2018_250m_utm.tif'

        with rasterio.open(in_file, 'r') as src:
            lulc_map = src.read()

        valid_lulc_types = [40, 90]  # 40: Cultivated vegetation/agriculture; 90: Herbaceous wetland

        lulc_map_ag = np.isin(lulc_map, valid_lulc_types)

        return lulc_map_ag[0]


    def update_region(self, region):

        self.train_irrig_array = self.data_array_dict[f'train_{region}_irrig']
        self.train_noirrig_array = self.data_array_dict[f'train_{region}_noirrig']

        min_class_pixels = np.min((len(self.train_irrig_array), len(self.train_noirrig_array)))
        max_class_pixels = np.max((len(self.train_irrig_array), len(self.train_noirrig_array)))

        self.num_iters_per_region =  np.ceil(max_class_pixels/min_class_pixels).astype(int)


    def take_new_batch_for_iter(self, iter):


        # Calculate the minimum pixels per class
        min_class_pixels = np.min((len(self.train_irrig_array), len(self.train_noirrig_array)))

        # Find indices for sample within training data
        irrig_start_ix = np.remainder(iter*min_class_pixels, len(self.train_irrig_array))
        irrig_end_ix   = np.min((irrig_start_ix + min_class_pixels, len(self.train_irrig_array)))

        noirrig_start_ix = np.remainder(iter * min_class_pixels, len(self.train_noirrig_array))
        noirrig_end_ix   = np.min((noirrig_start_ix + min_class_pixels, len(self.train_noirrig_array)))

        training_sample_irrig   = self.train_irrig_array[irrig_start_ix:irrig_end_ix]
        training_sample_noirrig = self.train_noirrig_array[noirrig_start_ix:noirrig_end_ix]

        # Take min snum of samples -- this is necessary to balance training for the last batch of training
        min_samples = int(np.min((len(training_sample_irrig), len(training_sample_noirrig))))
        training_sample_irrig = training_sample_irrig[0:min_samples]
        training_sample_noirrig = training_sample_noirrig[0:min_samples]


        irrig_labels = np.concatenate((np.zeros((len(training_sample_irrig), 1)),
                                       np.ones((len(training_sample_irrig), 1))), axis = -1).astype(np.float32)

        nonirrig_labels = np.concatenate((np.ones((len(training_sample_noirrig), 1)),
                                         np.zeros((len(training_sample_noirrig), 1))), axis=-1).astype(np.float32)



        training_samples = np.concatenate((training_sample_irrig, training_sample_noirrig), axis=0)
        training_labels  = np.concatenate((irrig_labels, nonirrig_labels), axis = 0)

        # Shuffle
        p = np.random.permutation(len(training_samples))


        # Return tensorflow dataset
        train_ds = tf.data.Dataset.from_tensor_slices((training_samples[p], training_labels[p])).batch(self.batch_size)


        return train_ds


    def return_val_or_test_data(self, set_name, region):

        # Extract arrays
        irrig_array = self.data_array_dict[f'{set_name}_{region}_irrig']
        noirrig_array = self.data_array_dict[f'{set_name}_{region}_noirrig']


        # Create labels
        irrig_labels = np.concatenate((np.zeros((len(irrig_array), 1)),
                                       np.ones((len(irrig_array), 1))), axis=-1).astype(np.float32)

        noirrig_labels = np.concatenate((np.ones((len(noirrig_array), 1)),
                                         np.zeros((len(noirrig_array), 1))), axis=-1).astype(np.float32)

        # Place new val datasets in lists
        val_irrig_ds = tf.data.Dataset.from_tensor_slices((irrig_array, irrig_labels)).batch(self.batch_size)
        val_noirrig_ds = tf.data.Dataset.from_tensor_slices((noirrig_array, noirrig_labels)).batch(self.batch_size)


        return val_irrig_ds, val_noirrig_ds

    def find_valid_pixels_srtm_maxmin(self, full_region, cropped_region):

        input_tif = self.input_tif.format(full_region, cropped_region)

        with rasterio.open(input_tif, 'r') as src:
            img = src.read()

        maxmin_ix = 10

        srtm_layer_valid = img[0] < 800
        maxmin_layer = img[maxmin_ix] > 2


        valid_pixel_map = srtm_layer_valid * maxmin_layer

        return valid_pixel_map

    def return_selected_pixels_for_training(self):
        catalonia_irrig_file_names = [
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/catalonia/rasters/catalonia_irrig_all_min_10ha_clusterbypolygon.tif',
        ]

        catalonia_noirrig_file_names = [
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/catalonia/rasters/catalonia_noirrig_min_10ha_clusterbypolygon.tif',
        ]

        fresno_irrig_file_names = [
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/california/rasters/fresno_irrig_min_10ha_nclusters_5.tif'
        ]

        fresno_noirrig_file_names = [
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/california/rasters/fresno_noirrig_min_10ha_nclusters_5.tif'
        ]

        amhara_irrig_file_names = [
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/ethiopia/rasters/tana_irrig_min10ha_clusterbypolygon.tif',
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/ethiopia/rasters/amhara_irrig_min10ha_clusterbypolygon.tif',
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/ethiopia/rasters/koga_irrig_min10ha_clusterbypolygon.tif',
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/ethiopia/rasters/amhara_EA_polys_irrig_from_irrig_min_10ha_clusterbypolygon.tif',
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/ethiopia/rasters/amhara_EA_polys_irrig_from_potirrig_min_10ha_clusterbypolygon.tif',

        ]
        amhara_noirrig_file_names = [
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/ethiopia/rasters/tana_noirrig_min10ha_clusterbypolygon.tif',
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/ethiopia/rasters/amhara_noirrig_min10ha_clusterbypolygon.tif',
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/ethiopia/rasters/amhara_noirrig_extra_min10ha_clusterbypolygon.tif',
            '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/ethiopia/rasters/amhara_EA_polys_noirrig_from_noirrig_min_10ha_clusterbypolygon.tif',
        ]

        catalonia_irrig_array, catalonia_noirrig_array = self.load_valid_pixel_rasters(catalonia_irrig_file_names,
                                                                                       catalonia_noirrig_file_names,
                                                                                       'catalonia', 'catalonia')
        fresno_irrig_array, fresno_noirrig_array = self.load_valid_pixel_rasters(fresno_irrig_file_names,
                                                                                 fresno_noirrig_file_names,
                                                                                 'california', 'fresno')
        amhara_irrig_array, amhara_noirrig_array = self.load_valid_pixel_rasters(amhara_irrig_file_names,
                                                                                 amhara_noirrig_file_names,
                                                                                 'ethiopia', 'amhara')



        uganda_irrig_array = np.zeros((1,1))
        uganda_noirrig_array = np.zeros((1,1))

        return catalonia_irrig_array, catalonia_noirrig_array, fresno_irrig_array, fresno_noirrig_array, \
               amhara_irrig_array, amhara_noirrig_array, uganda_irrig_array, uganda_noirrig_array

    def load_valid_pixel_rasters(self, irrig_file_names, noirrig_file_names, full_region, cropped_region):

        irrig_arrays = []
        noirrig_arrays = []

        for file in irrig_file_names:
            irrig_arrays.append(rasterio.open(file, 'r').read())

        for file in noirrig_file_names:
            noirrig_arrays.append(rasterio.open(file, 'r').read())

        # Take valid pixel maps based on srtm + max-min EVI ratio
        valid_pixel_map = self.find_valid_pixels_srtm_maxmin(full_region, cropped_region)
        valid_pixels_lulc =  self.load_lulc_pixels(cropped_region)

        irrig_array_out = np.max(np.stack(irrig_arrays, axis=0), axis=0)[0] * valid_pixel_map * valid_pixels_lulc
        nonirrig_array_out = np.max(np.stack(noirrig_arrays, axis=0), axis=0)[0] * valid_pixel_map * valid_pixels_lulc


        return irrig_array_out, nonirrig_array_out

    def return_pixels_for_map_prediction(self, cropped_region, columns_to_use):

        if cropped_region == 'amhara':
            full_region = 'ethiopia'
        elif cropped_region == 'catalonia':
            full_region = 'catalonia'
        elif cropped_region == 'uganda':
            full_region = 'uganda'
        else:
            full_region = 'california'

        map_file = self.input_tif.format(full_region, cropped_region)

        with rasterio.open(map_file, 'r') as src:
            img = src.read()

        img = np.transpose(img, (1,2,0))

        valid_lulc_pixels = self.load_lulc_pixels(cropped_region)
        valid_srtm_maxmin_pixels = self.find_valid_pixels_srtm_maxmin(full_region, cropped_region)

        # Find valid pixels
        valid_pixel_map = (~(np.isnan(img).any(axis=-1)) * valid_lulc_pixels * valid_srtm_maxmin_pixels)
        valid_pixel_indices = np.where(valid_pixel_map)

        valid_pixels_for_pred = img[valid_pixel_indices]
        valid_pixels_for_pred = valid_pixels_for_pred[:, columns_to_use[0]]

        # Standardize
        standard_array = self.standard_array_dict[f'{cropped_region}_standard_array']

        # standard_array = standard_array[..., columns_to_use[0]]


        valid_pixels_for_pred = (valid_pixels_for_pred - standard_array[0]) / standard_array[1]

        valid_pixels_for_pred = np.expand_dims(valid_pixels_for_pred, axis=1)

        valid_pixels_ds = tf.data.Dataset.from_tensor_slices(valid_pixels_for_pred).batch(self.batch_size)

        return valid_pixels_ds, valid_pixel_indices