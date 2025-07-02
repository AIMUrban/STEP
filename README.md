# STEP
This is the PyTorch implementation of the Semantic and Temporal Enhanced Prediction model (STEP) and helps readers to reproduce the results in the paper "**Next Location Prediction with Latent Location Semantics and Activity Time Inference**".


### Configurations
For TC datasets, the embedding dimensions of the proposed model are set to 48, while for MP, it's 24.  
The Transformer encoder consists of 2 layers, each with 4 attention heads and a dropout rate of 0.1.  
We train STEP for 15 epochs with a batch size of 128. 

### Datasets
The raw data utilized in this study originates from two publicly available datasets introduced in the papers [1][2].

Following the data processing definition outlined in the paper, we extracted user activity trajectories and constructed two CSV files: train.csv and test.csv. Each file contains three key attributes: `uid` (anonymized user ID), `t` (the recorded time of activity), and `l` (a unique identifier for each visited location).`

#### References

[1] Yu, F., Yan, H., Chen, R., Zhang, G., Liu, Y., Chen, M., & Li, Y. (2023). City-scale vehicle trajectory data from traffic camera videos. Scientific data, 10(1), 711.  
[2] Yabe, T., Tsubouchi, K., Shimizu, T., Sekimoto, Y., Sezaki, K., Moro, E., & Pentland, A. (2024). YJMob100K: City-scale and longitudinal dataset of anonymized human mobility trajectories. Scientific Data, 11(1), 397.

### Hyperparameters

All hyperparameter settings are saved in the `.yml` files under the respective dataset folder under `saved_models/`. \
\
For example, `saved_models/TC/settings.yml` contains hyperparameter settings of STEP for the Traffic Camera Dataset. 

### Run
- For STEP model:
  ```shell
  python ./model/run.py --dataset TC 
  ```
