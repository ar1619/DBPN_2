# Super Resolution Model Repository

This repository contains the implementation of our super resolution model, based on [1], including our training pipeline, hyperparameter tuning configurations and performance tests. The model has been trained on LST temperature data[2] for application to XCO2 data.  
Link to the paper [here](https://www.mdpi.com/2072-4292/17/9/1617).

---

## 📁 Repository Structure
<pre lang="markdown"> ```
├── data_distributions/ 
├── error_propagation/
├── experimentations/
├── figures/
├── Input/
├── model_performance/
├── noise_propagation/
├── output/
├── point_source_detection/
├── Result/
├── weights/
├── .gitignore 
├── base_networks.py 
├── create_dataset.py 
├── create_dataset.sh 
├── data_pwp.py 
├── data.py 
├── dataset.py 
├── eval_val.sh 
├── eval.pbs 
├── eval.py 
├── eval.sh 
├── exploration.ipynb 
├── main.py 
├── model.py 
├── README.md 
├── train_valid_split.py 
├── train_with_DIVK.pbs 
├── train.sh 
├── validate.pbs 
├── validate.py 
├── validate.sh 
``` </pre>
## Training the model

python -u main.py --upscale_factor 16 --patch_size 32 --lr 0.0002 --noise_level 0.01 --batchSize 4 --combination 2 --hr_train_dataset #path_to_training_dataset --hr_valid_dataset #path_to_validating_dataset

## 🛠 Configuration

You can modify the following arguments for training:

- Learning rate
- Upsampling rate
- Batch size
- Noise
- Number of epochs
- Dataset paths



## 🤝 Cite the paper

Rakotoharisoa, A., Cenci, S., & Arcucci, R. (2025). A High Resolution Spatially Consistent Global Dataset for CO2 Monitoring. Remote Sensing, 17(9), 1617.

---

[1]:Haris, M., Shakhnarovich, G., & Ukita, N. (2020). Deep back-projectinetworks for single image super-resolution. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(12), 4323-4337.

[2]:Wan, Z., Hook, S., & Hulley, G. (2015). MOD11C1 MODIS/Terra Land Surface Temperature/Emissivity Daily L3 Global 0.05 Deg CMG V006. (No Title).