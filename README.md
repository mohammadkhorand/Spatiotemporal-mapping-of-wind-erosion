Spatiotemporal Mapping of Wind Erosion Using Machine Learning
Overview
This repository contains the code and data processing workflow used for modeling and mapping the spatiotemporal patterns of soil wind erosion using machine learning approaches. The study focuses on identifying erosion-prone areas and analyzing the spatial variability of wind erosion risk.

Machine learning algorithms were applied to environmental and climatic variables to estimate erosion intensity and generate predictive spatial maps. The workflow includes model training, cross‑validation, statistical comparison of models, and uncertainty analysis.

Repository Structure
ML-soil-wind-erosion.ipynb

Main notebook containing the machine learning workflow for wind erosion modeling and spatial prediction.

KFOLD.py

Implementation of k‑fold cross‑validation used for model evaluation and robustness assessment.

uncertainetymapRF.py

Script used to generate uncertainty maps from Random Forest model outputs.

wilcoxon test.py

Statistical comparison of model performances using the Wilcoxon signed-rank test.

Data files

Environmental variables and derived datasets used for model training and validation.

Methods
The modeling workflow includes the following steps:

Data preparation and preprocessing of environmental variables.
Training machine learning models for wind erosion prediction.
Model validation using k‑fold cross‑validation.
Statistical comparison of models using the Wilcoxon signed-rank test.
Generation of spatial prediction maps.
Uncertainty assessment of the Random Forest predictions.
Study Area
The analysis focuses on arid and semi‑arid environments where wind erosion is a dominant land degradation process. Spatial datasets were prepared within a GIS environment and integrated with machine learning models.

Requirements
The scripts were developed in Python. Main libraries include:

numpy
pandas
scikit-learn
scipy
matplotlib
geopandas (for spatial data processing)
Usage
Prepare the input datasets containing environmental predictors and erosion observations.
Run the modeling notebook ML-soil-wind-erosion.ipynb.
Use KFOLD.py for cross‑validation experiments.
Generate uncertainty maps using uncertainetymapRF.py.
Compare model performances using wilcoxon test.py.
Applications
The outputs of this workflow can support:

land degradation assessment
identification of wind erosion hotspots
environmental management and land-use planning
risk mapping in arid regions
Citation
If you use this repository in your research, please cite the related publication (when available).

Author
Mohammad Khorand

License
This repository is provided for research and academic purposes.
