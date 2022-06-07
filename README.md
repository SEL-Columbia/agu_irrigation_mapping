# agu_irrigation_mapping

This repository is associated with a presentation given to the AGU 2020 Fall Meeting. Slides for the presentation can be found [here](https://storage.googleapis.com/terry_phd_export/presentations/agu_presentation.pptx), while a further discussion of the approach and results can be found in Chapter 7 of the following [PhD dissertation](https://storage.googleapis.com/terry_phd_export/thesis/tconlon_phd_dissertation.pdf).

## Overview

This repository contains code for irrigation detection in sub-Saharan Africa. The project leverages agricultural
 administrative data in California, USA and Catalonia, Spain for the binary classification problem of predicting
  irrigation presence in both Ethiopia and Uganda. Here, the presence of vegetation growth in the dry season is taken
   as a proxy for the existence of irrigation. Derived imagery layers are used in concert with ground-truth
    (administrative polygons in California and Catalonia; hand-drawn polygons in Ethiopia; no data in Uganda) to
     train a deep neural network that predicts irrigation presence on a pixelwise basis. Currently all analysis in
      this repository is completed at a 250m spatial resolution.

## Repository structure 
```
agu_irrigation_mapping
├── LICENSE
├── README.md
├── data_and_outputs/
│   ├── abundance_maps/
│   ├── chirps/
│   ├── derived_layers/
│   ├── exports_for_energy_optimization/
│   ├── imagery/
│   ├── predictions/
│   ├── saved_endmembers/
│   ├── saved_rainfall_regions/
│   ├── shapefiles_and_templates/
│   ├── spectral_mixture_analysis/
├── scripts/
│   ├── clean_groundtruth_polygons.py
│   ├── create_derived_layers.py
│   ├── dl_dataloader.py
│   ├── dl_training_main.py
│   ├── find_irrigation_zones.py
│   ├── model.py
│   ├── params.yaml
│   ├── raster_manipulation_tools.py
│   ├── utils.py
```

## Root Level Repository Description

The `data_and_outputs/` folder is too large to include in this repository. It is instead made available [here](terry_phd_export/projects/ethiopia/agu_irrigation_mapping/data_and_outputs).  
* Download this folder
  (~28.5 GB in size, so think about saving it to an external storage unit), and then point the `base_dir` field in
   `params.yaml` to the corresponding directory location. The storage bucket contains all the derived layers
    required for training; all the raw images required for creating new derived layers; the vector and raster
     groundtruth files; Copernicus land-cover maps; CHIRPS rainfall prediction rasters; polygonized predictions
      made on the Descartes Labs platform at 10m spatial resolution; and administrative vector files. Lastly, and
       potentially most usefully, it also contains a pretrained model and normalization in case the user wants to
        skip the training process. 


The `scripts/` folder contains all files that are required to run the presented analysis.

* `clean_groundtruth_polygons.py`: This script cleans the groundtruth polygons for training. Here, the user
 specifies the vector file (GeoJSON, Shapefile, etc.) containing the polygons to be cleaned, along with the EVI and
  CHIRPS rainfall raster for the area. The script then iterates through the specified polygons and clusters the
   pixels within the polygons into a user-specified `k` clusters. The user then walks through a number of prompts
    and saves the pixels associated with each cluster that demonstrates the phenology in question (i.e. dry season
     vegetation growth). The output of this script is a raster containing all the pixels within each user-submitted
      groundtruth polygons that have been verified to contain a phenology that can be associated with one of the two
       class labels (irrigated, non-irrigated). 

* `create_derived_layers.py`: This script contains functions that create the derived feature layers that are used as
 inputs for the deep classification model. Each function saves a raster that contains the value of each derived
  product (e.g. EVI-CHIRPS correlation coefficient, mean NDWI value at lowest rainfall timesteps, etc.) on a
   pixelwise basis. 

* `dl_dataloader.py`: This script contains an custom object that returns derived feature layers over groundtruthed
 pixels for training of the deep learning classification model. 

* `dl_training_main.py`: Running this script will train the irrigation classifier. First, the script loads the
 training data using an object created with `dl_dataloader.py`. The user specifies the regions to train + validate
  over (some combination of Amhara, Catalonia, and Fresno) over a set number of epochs. Based on the specifications
   in `update_model_weights()`, model training weights will either be saved or ignored. Once a model is trained, it
    can be used to predict irrigation presence over entire regions. 

* `find_irrigation_zones.py`: This script takes polygonized irrigation predictions and split them into irrigation
 zones -- circles of 300m radius that overlap the polygonized area. Each irrigation prediction polygon is divided
  into the number of irrigation zones that would equal the area of the prediction polygon, rounding up to the nearest
   integer. In the irrigation zone location process, the zones are shifted around in order to overlap with the
    largest amount of the polygon prediction area; next, the unique area of intersection for each zone and its
     corresponding irrigation prediction polygon is calculated. Lastly, the irrigation zones, the zones' centroids
     , and the intersecting area of the zones and polygonized prediction are saved as `.geojson` files. 

* `params.yaml`: This file holds relevant repository paramters. At the moment, it contains only the path to a base
 directory that holds files necessary for model training and prediction. 

* `model.py`: This script contains the neural network (built with Keras model subclassing) that is trained and used
 for predictions. 

* `raster_manipulation_tools.py`: This script contains helper functions for manipulating rasters with rasterio. 

* `utils.py`: This script contains general utility functions for the repository, including those used to apply basic
 morphological processing to prediction maps, one for temporal interpolation and smoothing of imagery timeseries
  stacks, and one for polygonizing pixelwise predictions. 



