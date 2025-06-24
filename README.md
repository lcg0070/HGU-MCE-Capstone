# HGU-MCE Post-Capstone, MIP   
# Enhancing U-Net for PCB Segmentation Using Hyperspectral Imaging in E-waste Recycling

1. **Research folders** : final version of software, thesis paper, presentation etc. 
2. **Tutorial**:  Github repository [Github](https://github.com/lcg0070/HGU-MCE-Capstone)
3. **Video Clip**: [Youtube](https://youtu.be/RDxqIfjPETQ)

---
## Tutorial (For Software-based Topics) 

### Dataset Details

The dataset includes:
- RGB images of 53 PCBs scanned with a high-resolution RGB camera (Teledyne Dalsa C4020).
- 53 hyperspectral data cubes of those PCBs scanned with Specim FX10 in the VNIR range.
- Two segmentation ground truth files: 'General' and 'Monoseg' for 4 classes of interest - 'others,' 'IC,' 'Capacitor,' and 'Connectors.'

### Data Access

To utilize the dataset, download it from the Rodare website: [Rodare](https://rodare.hzdr.de/record/2704), or from Zenodo: [Zenodo](https://zenodo.org/records/10617721).

### Requirements

To use the codes without errors, install the libraries listed in the Requirements.txt file. The codes require at least 1 GPU to run and handle the data.

### Code Usage  

---
1. Make directory to save the dataset(train, validation, test) for full_channel dataset and PCA processed dataset
2. Then using **HSI_cube_generation.ipynb, HSI_PCA_data_generation.ipynb** to generate cube dataset(640x640x214), pca dataset(640x640x3)
3. Use the **HSI_cube.ipynb and HSI_PCA_data_generation.ipynb** to train each of the different models in the models folder and display the results for every model.
   - HSI_cube.ipynb  
        - End2End_AttUnet.py
        - End2End_ResUnet.py
        - End2End_Unet.py
        - ResUnet.py
        - Unet.py
        - Unet_Attention.py
   - HSI_PCA.ipynb
        - ResUnet.py
        - Unet.py
        - Unet_Attention.py

4. In the end, will obtain 9 separate weight files—one for each configuration:
   - Full 214-channel input 
     - ResUNet 
     - UNet 
     - Attention UNet 
   - PCA-reduced 3-channel input 
     - ResUNet 
     - UNet 
     - Attention UNet 
   - 214-channel input with the SCRB module 
     - ResUNet + SCRB 
     - UNet + SCRB 
     - Attention UNet + SCRB
---


## License

The code is licensed under the Apache-2.0 license. Any further development and application using this work should be opened and shared with the community.

