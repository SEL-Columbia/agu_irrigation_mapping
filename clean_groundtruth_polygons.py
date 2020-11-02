
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.mask import mask
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import FixedLocator, IndexLocator


def load_edwin_polygons():

    poly_file = '/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/points_for_vt_testing/points_from_edwin.geojson'
    poly = gpd.read_file(poly_file).to_crs('EPSG:32637')

    irrig_polys = poly[poly['Name'] == 'I'] #['geometry'].tolist()
    possible_irrig_polys = poly[poly['Name'] == 'I-N'] #['geometry'].tolist()
    non_irrig_polys = poly[poly['Name'] == 'N'] #['geometry'].tolist()

    return irrig_polys, possible_irrig_polys, non_irrig_polys


def cluster_and_plot_polys(ix, polys,  poly_file,  evi_src, chirps_src, n_kmeans_clusters, n_polys_to_consider_simul):


    evi_meta = evi_src.meta
    evi_polys = polys.to_crs(evi_meta['crs'])
    poly = [evi_polys['geometry'].iloc[jx] for jx in range(ix*n_polys_to_consider_simul, (ix+1)*n_polys_to_consider_simul)]

    chirps_img = np.transpose(chirps_src.read(), (1, 2, 0))
    chirps_meta = chirps_src.meta
    chirps_polys = polys.to_crs(chirps_meta['crs'])
    chirps_poly = [chirps_polys['geometry'].iloc[ix]]

    print(f'Mask image with polygon set {ix*n_polys_to_consider_simul}:{(ix+1)*n_polys_to_consider_simul}')
    masked_img, _ = mask(evi_src, poly)
    masked_chirps, _ = mask(chirps_src, chirps_poly, all_touched=True, nodata=chirps_meta['nodata'])
    valid_chirps_pixels = ~np.any(masked_chirps == chirps_meta['nodata'], axis=0)

    masked_img = np.transpose(masked_img, (1, 2, 0))
    valid_evi_pixels = ~np.any(masked_img == evi_meta['nodata'], axis=-1)

    num_valid_pixels = np.count_nonzero(valid_evi_pixels)

    if num_valid_pixels < 10:
        print('Not enough valid pixels in the polygon')
        return False, _




    print('Collect valid pixels')
    valid_pixel_ts = masked_img[valid_evi_pixels]
    valid_chirps_ts = chirps_img[valid_chirps_pixels]

    mean_chirps_ts_shifted = np.nanmean(valid_chirps_ts, axis=0)

    # PCA Transform + cluster
    n_pca_components = np.min((8, len(valid_pixel_ts)))

    # Initialize a PCA model and fit to the data
    print('Calculate PCA')
    pca = PCA(n_components=n_pca_components)

    principalComponents = pca.fit_transform(valid_pixel_ts)

    print('Cluster')
    kmeans_cluster = KMeans(n_clusters=n_kmeans_clusters).fit(principalComponents)
    cluster_predicts = kmeans_cluster.predict(principalComponents)
    cluster_centers = kmeans_cluster.cluster_centers_

    cluster_centers_ts = pca.inverse_transform(cluster_centers)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax2 = axes[0].twinx()

    colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'grape', 'dark turquoise', 'terracotta',
                   'salmon pink', 'evergreen'
                   ]
    cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
    sns.set_palette(sns.xkcd_palette(colors_xkcd))


    for i in range(n_kmeans_clusters):
        axes[0].plot(range(cluster_centers_ts.shape[1]), cluster_centers_ts[i],
                     label=f'Cluster {i}, {np.count_nonzero(cluster_predicts == i)} px.',
                     color=cmap[i + 1])

    ax2.plot(np.linspace(0, cluster_centers_ts.shape[1], len(mean_chirps_ts_shifted)), mean_chirps_ts_shifted,
             color='k', linestyle='-.', label='Mean shifted rainfall')

    ticknames = ['01/2017', '01/2018', '01/2019']
    minors = np.linspace(0, cluster_centers_ts.shape[1], 37)
    axes[0].set_xlabel('Month')
    axes[0].set_xticklabels(ticknames)
    axes[0].xaxis.set_major_locator(IndexLocator(cluster_centers_ts.shape[1]/3, 0))



    axes[0].xaxis.set_minor_locator(FixedLocator(minors))
    axes[0].tick_params(axis='x', which='minor', length=2)
    axes[0].tick_params(axis='x', which='major', length=4)

    axes[0].legend(loc='upper left')
    ax2.legend(loc='upper right')

    axes[0].set_ylabel('Mean EVI')
    ax2.set_ylabel('Mean Rainfall')

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # Reconstitute groundtruth at valid pixel locations
    valid_pixels_map = np.zeros((masked_img.shape[0], masked_img.shape[1], 1)) - 1
    cluster_predicts_for_plotting = cluster_predicts
    valid_pixels_map[valid_evi_pixels] = np.expand_dims(cluster_predicts_for_plotting, -1)

    cmap_im = colors.ListedColormap(sns.xkcd_palette(colors_xkcd)[0:n_kmeans_clusters + 1])
    bounds = np.arange(start=-1.5, stop=.5 + n_kmeans_clusters, step=1)
    # norm = colors.BoundaryNorm(bounds, cmap_im.N)

    valid_pixels_for_plotting = np.where(valid_pixels_map > -1)
    # Create plotting boundaries
    xmin = np.min(valid_pixels_for_plotting[0]) - 5
    xmax = np.max(valid_pixels_for_plotting[0]) + 5

    ymin = np.min(valid_pixels_for_plotting[1]) - 5
    ymax = np.max(valid_pixels_for_plotting[1]) + 5

    im = axes[1].imshow(valid_pixels_map[xmin:xmax, ymin:ymax, 0], interpolation='nearest', origin='upper',
                        cmap=cmap_im)
    xticks = axes[1].get_xticks()
    axes[1].set_xticklabels(np.round(np.linspace(start=xmin, stop=xmax, num=len(xticks))).astype(int))
    yticks = axes[1].get_yticks()
    axes[1].set_yticklabels(np.round(np.linspace(start=ymin, stop=ymax, num=len(yticks))).astype(int))

    fig.colorbar(im, cax=cax, orientation='vertical', boundaries=bounds, ticks=range(-1, n_kmeans_clusters + 1))

    file_name = poly_file.split('/')[-1].replace('.geojson', '')
    plt.suptitle(file_name)

    plt.show()

    return True, valid_pixels_map


def load_evi_and_chirps_pixel_timeseries(poly_file, full_region, cropped_region):

    evi_in_file = f'/Volumes/sel_external/ethiopia_vegetation_detection/imagery/{full_region}/modis/stacked_and_cropped/utm_projection/{cropped_region}_evi_250m_interp_smoothed_reproj.tif'
    # evi_in_file = '/Volumes/sel_external/ethiopia_vegetation_detection/imagery/ethiopia/sentinel/overlapping_with_visual_truth/clipped_extent_edwin_polys_east_37PEM_infilled.tif'

    chirps_in_file = f'/Volumes/sel_external/ethiopia_vegetation_detection/chirps/resampled_nn/{cropped_region}_chirps_2017_2020_resampled_250m_modis_dates_shifted_mm.tif'

    print('Load EVI + Mask')
    evi_src = rasterio.open(evi_in_file, 'r')
    evi_meta = evi_src.meta
    chirps_src = rasterio.open(chirps_in_file, 'r')

    min_ha = 10

    polys = gpd.read_file(poly_file).to_crs(evi_meta['crs'])

    polys = polys[(polys['geometry']).area/10000 > min_ha]

    n_polys_to_consider_simul = 3

    print(f'Number of polygons in shapefile: {len(polys)}')

    valid_pixel_list = []

    for ix in range(int(np.floor(len(polys)/n_polys_to_consider_simul))):

        poly_area_ha = np.sum([(polys['geometry'].iloc[jx]).area/10000 for jx in range(ix*n_polys_to_consider_simul,
                                                                                       (ix+1)*n_polys_to_consider_simul)])
        recluster = True
        n_kmeans_clusters = 5

        if poly_area_ha > min_ha:
            while recluster == True:
                valid_plot, valid_pixels_map = cluster_and_plot_polys(ix, polys, poly_file, evi_src, chirps_src,
                                                                      n_kmeans_clusters, n_polys_to_consider_simul)
                if valid_plot:
                    recluster_response = input('Do you want to recluster? (y/n)')
                    if recluster_response == 'y':
                        n_kmeans_clusters = int(input('How many clusters should be used? (integer)'))

                    if recluster_response == 'n':
                        recluster = False
                else:
                    recluster = False

            if valid_plot:
                valid_integer_input = False
                save_pixels = False

                try:
                    while not valid_integer_input:
                        valid_cluster_list = input('Which clusters should you save? (Enter integers separated by a space, a for all, or n for none)')

                        if valid_cluster_list == 'n':
                            valid_integer_input = True

                        elif valid_cluster_list == 'a':
                            input_list = range(n_kmeans_clusters)
                            valid_integer_input = True
                            save_pixels = True

                        else:
                            try:
                               input_list = [int(i) for i in valid_cluster_list.split(' ')]
                               valid_integer_input = True
                               save_pixels = True
                            except Exception as e:
                                print(e)
                                print('Improper input')
                except:
                    print('Improper input')

                plt.close()
                if save_pixels:
                    print('Saving pixels')

                    valid_pixel_list.append(np.where(np.isin(valid_pixels_map, input_list)))


    # Save selected pixels to raster
    valid_pixel_map_for_export = np.zeros(valid_pixels_map.shape).astype(np.int16)
    for pixels in valid_pixel_list:
        valid_pixel_map_for_export[pixels] = 1

    file_name = poly_file.split('/')[-1].replace('.geojson', '')

    valid_pixel_map_for_export = np.transpose(valid_pixel_map_for_export, (2,0,1))

    evi_meta['count'] = '1'
    evi_meta['dtype'] = 'int16'
    evi_meta['nodata'] = -3000


    out_file = f'/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/{full_region}/rasters/{file_name}_min{min_ha}ha_clusterbypolygon.tif'

    with rasterio.open(out_file, 'w', **evi_meta) as dest:
        dest.write(valid_pixel_map_for_export)

if __name__ == '__main__':

    full_region = 'ethiopia'
    cropped_region = 'amhara'


    poly_file = f'/Volumes/sel_external/ethiopia_vegetation_detection/groundtruth/sentinel_modis_testing/polygons/edwin_polys_32637.geojson'

    load_evi_and_chirps_pixel_timeseries(poly_file, full_region, cropped_region)
