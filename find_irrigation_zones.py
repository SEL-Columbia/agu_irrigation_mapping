import numpy as np
import shapely
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import fiona
from shapely.geometry import Point, Polygon
from math import ceil
from shapely.affinity import translate
from tqdm import tqdm
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio import Affine
from rasterio.enums import MergeAlg
from dl_training_main import get_args
import os

fiona.supported_drivers['KML'] = 'rw'
fiona.supported_drivers['LIBKML'] = 'rw'

def initial_plot(poly, center_dist=300, zone_size=300):
    '''
    Find initial locations of irrigation zones
    :param poly: Irrigation prediction polygon for which to divide into irrigation zones
    :param center_dist: Distance between initial irrigation zones (m)
    :param zone_size: How large each irrigation zone is (radius of a circle, m)
    :return:
    '''

    # Create list to hold irrigation zones
    center_list = []
    centroid = poly.centroid

    # Find box around irrigaiton prediction polygon
    box = poly.minimum_rotated_rectangle
    x, y = box.exterior.coords.xy

    min_x_delta = min(x) - centroid.x
    max_x_delta = max(x) - centroid.x

    min_y_delta = min(y) - centroid.y
    max_y_delta = max(y) - centroid.y

    leftmost_pt = centroid.x + ceil(np.divide(min_x_delta, center_dist)) * center_dist
    topmost_pt  = centroid.y + np.floor_divide(max_y_delta, center_dist) * center_dist

    rightmost_pt = centroid.x + np.floor_divide(max_x_delta, center_dist) * center_dist
    bottommost_pt = centroid.y + ceil(np.divide(min_y_delta, center_dist)) * center_dist

    # Popualte a grid of irrigation zones (circles with radius zone_size).
    for x_center in np.arange(leftmost_pt, rightmost_pt+center_dist, center_dist):
        for y_center in np.arange(topmost_pt, bottommost_pt-center_dist, -center_dist):
            point = Point(x_center, y_center).buffer(zone_size)
            center_list.append(point)

    # Return all irrig zones if they overlap a certain amount with the polygon predictions
    center_list = [i for i in center_list if i.intersection(poly).area > 30000]

    return center_list


def find_least_overlap_poly(poly, irrig_zone_list):
    '''
    Function for finding the irrigation zone with the least overlap with the irrigation prediction
    :param poly: Irrigaiton prediction poly
    :param irrig_zone_list: List of irrigation zones
    :return: Irrig zone list with least-overlapping polygon removed
    '''

    overlap_list = [poly.intersection(i).area for i in irrig_zone_list]
    least_overlap_poly_ix = np.argmin(overlap_list)
    irrig_zone_list.pop(least_overlap_poly_ix)

    return irrig_zone_list


def shift_polygons(poly, irrig_zones_list, shift):
    '''
    Function for shifting the irrigation zones to ascertain the max overlap with the irrigation prediction polygon

    :param poly: Irrigation prediction polygon
    :param irrig_zones_list: List of irrigaiton zones
    :param shift: How much the polygons should be shifted (in all cardinal directions, m)
    :return: List with the irrigation zones that have the largest overlap with the irrigation prediction polygon
    '''

    opt_zone_list = []

    # Enumerate through the irrig_zones, shift each one in all directions, and return the one with the largest
    # intersection with the prediction poly
    for ix, zone in enumerate(irrig_zones_list):

        shifted_left  = shapely.affinity.translate(zone, xoff=-shift)
        shifted_up    = shapely.affinity.translate(zone, yoff=shift)
        shifted_right = shapely.affinity.translate(zone, xoff=shift)
        shifted_down  = shapely.affinity.translate(zone, yoff=-shift)

        shifted_list = [zone, shifted_left, shifted_up, shifted_right, shifted_down]

        no_shift_area      = poly.intersection(unary_union(irrig_zones_list)).area
        shifted_left_area  = poly.intersection(unary_union(irrig_zones_list + [shifted_left])).area
        shifted_up_area    = poly.intersection(unary_union(irrig_zones_list + [shifted_up])).area
        shifted_right_area = poly.intersection(unary_union(irrig_zones_list + [shifted_right])).area
        shifted_down_area  = poly.intersection(unary_union(irrig_zones_list + [shifted_down])).area

        shifted_area_list = [no_shift_area, shifted_left_area, shifted_up_area, shifted_right_area, shifted_down_area]
        max_ix = np.argmax(shifted_area_list)

        opt_zone_list.append(shifted_list[max_ix])

    return opt_zone_list



def find_irrigation_zones(polygons_name, epsg_code):
    '''
    Function for finding the irrigation zones associated with an irrigation prediction polygons
    :param polygons_name: Filename for the irrigation prediction polygon
    :param epsg_code: EPSG code for converting to UTM coordinates
    :return: N/A; save irrigation zone polygons, irrigation zone centroids, and overlapping area between irrigation
    prediction polygons and the irrigation zones
    '''
    # Load file
    polys = gpd.read_file(polygons_name).to_crs(epsg_code)
    zone_size = 300

    # Create GDFs for export
    gdf_zone_export =  gpd.GeoDataFrame(columns=['irrig_polygon_ix', 'geometry'],
                                          crs=epsg_code)

    gdf_zone_centroid_export = gpd.GeoDataFrame(columns=['irrig_polygon_ix', 'geometry'],
                                         crs=epsg_code)

    gdf_polys_export = gpd.GeoDataFrame(columns=['irrig_polygon_ix', 'zone_centroid_x',
                                                  'zone_centroid_y', 'geometry'],
                                         crs=epsg_code)

    irrig_poly_list = []
    zone_geometry_list = []
    geometry_intersection_list = []
    zone_centroid_x = []
    zone_centroid_y = []

    # Loop through the irrigation prediction polygons and find the associated, best irrigation zones
    for ix in range(len(polys)):
        poly = polys['geometry'].iloc[ix]
        total_zones = np.ceil(poly.area / (np.pi*zone_size**2))



        polys_list = initial_plot(poly)
        zone_count = len(polys_list)

        pbar = tqdm(total=zone_count - total_zones)
        while zone_count > total_zones:
            pbar.update(1)

            polys_list = find_least_overlap_poly(poly, polys_list)
            zone_count = len(polys_list)

            shift_quantities = [100, 50, 25, 12.5, 5]

            for shift in shift_quantities:
                polys_list = shift_polygons(poly, polys_list, shift)


        for jx, zone in enumerate(polys_list):
            irrig_poly_list.append(ix)
            zone_geometry_list.append(zone)
            zone_centroid_x.append(zone.centroid.x)
            zone_centroid_y.append(zone.centroid.y)
            geometry_intersection_list.append(zone.intersection(poly))

    # Populate export GDFs
    gdf_zone_export['irrig_polygon_ix'] = irrig_poly_list
    gdf_zone_export['geometry'] = zone_geometry_list

    gdf_zone_centroid_export['irrig_polygon_ix'] = irrig_poly_list
    gdf_zone_centroid_export['geometry'] = [i.centroid for i in zone_geometry_list]

    gdf_polys_export['irrig_polygon_ix'] = irrig_poly_list
    gdf_polys_export['zone_centroid_x'] = zone_centroid_x
    gdf_polys_export['zone_centroid_y'] = zone_centroid_y
    gdf_polys_export['geometry'] = geometry_intersection_list

    # Save out
    dir_name = os.path.dirname(polygons_name)

    gdf_zone_export.to_file(f'{dir_name}/irrig_zones.geojson', driver='GeoJSON')
    gdf_zone_centroid_export.to_file(f'{dir_name}/irrig_zones_centroids.geojson', driver='GeoJSON')
    gdf_polys_export.to_file(f'{dir_name}/intersection_irrig_zones_and_polys.geojson', driver='GeoJSON')



def find_overlapping_area_save_new(polygons_name):
    '''
    Function for finding the unique area (i.e. pixels not counted twice) of intersection between the polygonized
    irrigation predictions and the irrigation zones
    :param polygons_name: Name of the file containing the polygonized irrigation predictions.
    :return: N/A. Overwrites irrigation zone geojson and polygonized prediction geojson with new column representing
    the unique overlapping area between the two sets of polygons.
    '''
    #
    # Find dir name and filenames of irrigation zones and the polygonized irrigation predictions
    dir_name = os.path.dirname(polygons_name)
    zones_filename = f'{dir_name}/irrig_zones.geojson'
    intersecting_polys_filename = f'{dir_name}/intersection_irrig_zones_and_polys.geojson'

    # Read in
    zones_full = gpd.read_file(zones_filename)
    intersecting_polys_full = gpd.read_file(intersecting_polys_filename)

    irrig_polygons = intersecting_polys_full['geometry']

    min_x = np.inf
    max_x = 0

    min_y = np.inf
    max_y = 0

    # Find boundary surrounding all polygons
    for poly in irrig_polygons:

        (poly_minx, poly_miny, poly_maxx, poly_maxy) = poly.bounds

        min_x = int(np.floor(np.min((min_x, poly_minx))))
        max_x = int(np.ceil(np.max((max_x, poly_maxx))))
        min_y = int(np.floor(np.min((min_y, poly_miny))))
        max_y = int(np.ceil(np.max((max_y, poly_maxy))))


    # Create an affine transformation
    trans = Affine(10, 0, min_x, 0, -10, max_y)

    out_x = int(np.ceil((max_x-min_x)/10))
    out_y = int(np.ceil((max_y-min_y)/10))

    # Rasterize the irrigation polygons preds
    out_array = rasterize(irrig_polygons, out_shape=(out_y, out_x),
                          fill=0, transform=trans, merge_alg=MergeAlg.add)

    # Take any pixel location that represents an overlap
    intersecting_pixels = np.argwhere(out_array > 1)

    # Create a list of all pixels that reside in an intersecting area saved as Shapely Points
    points_list = []
    for ix, pixel in enumerate(intersecting_pixels):
        points_list.append(Point(min_x + pixel[1]*10, max_y - pixel[0]*10))

    # Create spatial index for finding associated irrigation zones
    gdf = gpd.GeoDataFrame()
    gdf['geometry'] = points_list
    spatial_index = gdf.sindex

    points_dict = {}
    full_points_list = []

    # Create a mirror list of the irrig zones associated with every intersecting pixel
    for ix, poly in enumerate(irrig_polygons):

        possible_matches_index = list(spatial_index.intersection(poly.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_indices = possible_matches.within(poly)
        precise_matches = possible_matches[precise_indices]

        locs = list(precise_indices[precise_indices].index)
        full_points_list.append(locs)



    full_points_unique = np.unique([item for sublist in full_points_list for item in sublist])

    # Create a dictionary where every key is a point that resides in the intersection of two polys
    # Each value will be irrigation prediction polygon(s) that contain the intersecting point
    for point in full_points_unique:
        for ix, sub_list in enumerate(full_points_list):
            if point in sub_list:
                if point not in points_dict.keys():
                    points_dict[point] = [ix]
                else:
                    points_dict[point].append(ix)

    # Find total area for the predicted irrigation polys
    poly_area_list = [i.area for i in irrig_polygons]

    # Subtract intersecting area from total polygon predictions area calculations
    for key in points_dict.keys():
        poly_ixs = points_dict[key]
        for ix in poly_ixs:
           poly_area_list[ix] -= 100/len(poly_ixs)

    # Overwrite irrigation zone and polygon prediction files with additional unique intersection list
    intersecting_polys_full['poly_zone_unique_intersection_area_ha'] = [i/10000 for i in poly_area_list]
    zones_full['poly_zone_unique_intersection_area_ha'] = [i/10000 for i in poly_area_list]

    # Save out
    zones_full.to_file(zones_filename, driver='GeoJSON')
    intersecting_polys_full.to_file(intersecting_polys_filename, driver='GeoJSON')



def plot_polys(polys):
    '''
    Function for plotting polygson
    :param polys: List containing Shapely polygons
    :return: N/A; Displays plot.
    '''


    fig, ax = plt.subplots()
    for poly in polys:
        p = PolygonPatch(poly)
        ax.add_patch(p)
        ax.axis('scaled')
        ax.grid()

    plt.show()



if __name__ == '__main__':
    args = get_args()

    polygons_name  = f'{args.base_dir}/predicted_polys_from_dl_platform/north_gondar/polygonized_preds_min_3ha.kml'

    epsg_code = 'EPSG:32637'
    # find_irrigation_zones(polygons_name, epsg_code)


    find_overlapping_area_save_new(polygons_name)