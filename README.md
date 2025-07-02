# STEP
This is the PyTorch implementation of the Semantic and Temporal Enhanced Prediction model (STEP) and helps readers to reproduce the results in the paper "**Next Location Prediction with Latent Location Semantics and Activity Time Inference**".


### Configurations
For TC datasets, the embedding dimensions of the proposed model are set to 48, while for MP, it's 24.  
The Transformer encoder consists of 2 layers, each with 4 attention heads and a dropout rate of 0.1.  
We train STEP for 15 epochs with a batch size of 128. 

### DataSet


### Hyperparameters
All hyperparameter settings are saved in the `.yml` files under the respective dataset folder under `saved_models/`. \
\
For example, `saved_models/TC/settings.yml` contains hyperparameter settings of STEP for Traffic Camera Dataset. 

### Run
- For STEP model:
  ```shell
  python ./model/run.py --dataset TC --dim 48
  ```

