# AI-LoadForecast-UniBuilding

**`Available online: 19 March 2025 / Author: Yu-Shin Hu, Kai-Yun Lo, and I-Yun Lisa Hsieh`**

This repository contains the code and models for the research paper [AI-Driven Short-Term Load Forecasting Enhanced by Clustering in Multi-Type University Buildings: Insights Across Building Types and Pandemic Phases](https://www.sciencedirect.com/science/article/pii/S2352710225006540)


## Installation
To set up the environment, run:

```
conda create --name aibld python=3.8.20
conda activate aibld
pip install -r requirements.txt
```

## Project Structure

### `modules/`: Custom Function Modules
- `bilstm.py`: BiLSTM model implementation  
- `data_processing.py`: Data processing functions  
- `kmeans.py`: K-Means clustering functions  
- `pca.py`: PCA dimensionality reduction  

### `model_weights/`: Pretrained Model Weights  
- `cluster_buildingdict.pkl`: Mapping of buildings to clusters  
- `seperate_bilstm_0.hdf5`-`seperate_bilstm_2.hdf5`: BiLSTM model weights for each cluster  

## Cite this work

If you found this work useful, please consider citing:

```bibtex
@article{HU2025112417,
title = {AI-Driven Short-Term Load Forecasting Enhanced by Clustering in Multi-Type University Buildings: Insights Across Building Types and Pandemic Phases},
journal = {Journal of Building Engineering},
pages = {112417},
year = {2025},
issn = {2352-7102},
doi = {https://doi.org/10.1016/j.jobe.2025.112417},
url = {https://www.sciencedirect.com/science/article/pii/S2352710225006540},
author = {Yu-Shin Hu and Kai-Yun Lo and I-Yun Lisa Hsieh}
}

```
