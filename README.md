irrigation_detection
==============================


# Repository structure 

This repository contains code for irrigation detection in sub-Saharan Africa. The project leverages agricultural administrative data in California, USA and Catalonia, Spain for the binary classification problem of predicting irrigation presence in both Ethiopia and Uganda. Here, the precense of vegetation growth in the dry season is taken as a proxy for the existence of irrigation. Derived imagery layers are used in concert with ground-truth (administrative polygons in California and Catalonia; hand-drawn polygons in Ethiopia; no data in Uganda) to train a deep neural network that predicts irrigation presence on a pixelwise basis. Currently all analysis in this repository is completed using MODIS imagery.

There are 7 scripts which make up the irrigation_detection repository. 

1. `clean_groundtruth_polygons.py`: This script cleans the groundtruth polygons for training. Here, the user specifies the vector file (GeoJSON, Shapefile, etc.) containing the polygons to be cleaned, along with the EVI and CHIRPS rainfall raster for the area. The script then iterates through the specified polygons and clusters the pixels within the polygons into a user-specified `k` clusters. The user then walks through a number of prompts and saves the pixels associated with each cluster that demonstrates the phenology in question (i.e. dry season vegetation growth). The output of this script is a raster containing all the pixels within each user-submitted groundtruth polygons that have been verfied to contain a phenology that can be associated with one of the two class labels (irrigated, non-irrigated). 

2. `create_derived_layers.py`: This script contains functions that create the derived feature layers that are used as inputs for the deep classification model. Each function saves a raster that contains the value of each derived product (e.g. EVI-CHIRPS correlation coefficient, mean NDWI value at lowest rainfall timesteps, etc.) on a pixelwise basis. 

3. `dl_dataloader.py`: This script contains an custom object that returns derived feature layers over groundtruthed pixels for training of the deep learning classification model. 

4. `dl_training_main.py`: Running this script will train the irrigation classifier. First, the script loads the training data using an object created with `dl_dataloader.py`. The user specifies the regions to train + validate over (some combination of Amhara, Catalonia, and Fresno) over a set number of epochs. Based on the specifications in `update_model_weights()`, model training weights will either be saved or ignored. Once a model is trained, it can be used to predict irrigation presence over entire regions. 

5. `model.py`: This script contains the neural network (built with Keras model subclassing) that is trained and used for predictions. 

6. `raster_manipulation_tools.py`: This script contains helper functions for manipulating rasters with rasterio. 

7. `utils.py`: This script contains general utility functions for the repository, including those used to apply basic morphological processing to prediction maps, one for temporal interpolation and smoothing of imagery timeseries stacks, and one for vectorizing pixelwise predictions. 

# Other information

* All derived layers and raw imagery will be made available at a public Google storage bucket.

* Currently, no environment file exists for this repository. The one required package specification is `tensorflow==2.2.0`.


