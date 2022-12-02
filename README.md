# Building Footprint Generation 
Building footprint generation is a complex task where we have to accuratlety predict the Building polygon from a given satellite RGB-PAN sharpen image.There has been numerous of papers and models availbale to do this task but still we have seen lot of irregularities in various models available.

# Objective
The objective of the repo is to create an in-house AI model with the help availble polygon data of buildings from database available either as opensource like MSFT or OSM or internal Ttom database. 
The repository helps in creating data and an AI model for predicting building polygons on satellite RGB images from given geojson file either from MSFT/OSM/TTom.


# Creation Process of the Data
The building Extraction is done in 2 stages 

## Data creation 

We used MSFT geojson and polygon area city wise to produce a zoom level 14 satellite images and with the help of geojson 
we can create mask images of the input. For model input we have both Satellite image in RGBA channel and mask images both saved in tif format

## DNN architecture and training
The network backbone we used is EfficientNet described here. Although we can have millions of data from multiple geojsons by MSFT, 
we found that an effective combination of supervised and unsupervised training yields the best results.We also used Augmentaion to 
rotate sheer and image processing techniques to create random data and make data as generic as possible.



