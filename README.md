# HGU-MCE Post-Capstone, MIP   
# Enhancing U-Net for PCB Segmentation Using Hyperspectral Imaging in E-waste Recycling

1. **Research folders** : final version of software, thesis paper, presentation etc. 
2. **Tutorial**:  Github repository
3. **Video Clip:**  


---


## Tutorial (For Software-based Topics) 

Generate 640 × 640 images using the generation files (PCA and CUBE).
Then train using the HSI_cube and HSI_PCA files, save the resulting weights, and compare their performance.

### Requirements

To use the codes without errors, install the libraries listed in the Requirements.txt file. The codes require at least 1 GPU to run and handle the data.

### Dataset Details

The dataset includes:
- RGB images of 53 PCBs scanned with a high-resolution RGB camera (Teledyne Dalsa C4020).
- 53 hyperspectral data cubes of those PCBs scanned with Specim FX10 in the VNIR range.
- Two segmentation ground truth files: 'General' and 'Monoseg' for 4 classes of interest - 'others,' 'IC,' 'Capacitor,' and 'Connectors.'

### Data Access

To utilize the dataset, download it from the Rodare website: [Rodare](https://rodare.hzdr.de/record/2704), or from Zenodo: [Zenodo](https://zenodo.org/records/10617721).


## License

The code is licensed under the Apache-2.0 license. Any further development and application using this work should be opened and shared with the community.


[//]: # (## Video Clip)

[//]: # ()
[//]: # (Submit Video clip file &#40;~30sec&#41;.  This will be uploaded in my Lab's youtube. )

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (It should contain)

[//]: # ()
[//]: # (* Title page:  )

[//]: # (  * Course name, date, Research title, your name, advisor's name)

[//]: # (  * Institute &#40; School of Mechanical and Control Engineering&#41;)

[//]: # (* Overview of research:)

[//]: # (  * You can use PPT or Poster to explain overview of research)

[//]: # (* Result Video)

[//]: # (  * image, video that demonstrate results)
