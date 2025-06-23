
# Enhancing U-Net for PCB Segmentation Using Hyperspectral Imaging in E-waste Recycling




generation 파일들을 사용하여 640x640 이미지를 생성하고(PCA, CUBE) 
HSI_cube, HSI_PCA 파일을 사용하여 학습하는 weight를 저장후 성능을 비교한다.

## Dataset Details

The dataset includes:
- RGB images of 53 PCBs scanned with a high-resolution RGB camera (Teledyne Dalsa C4020).
- 53 hyperspectral data cubes of those PCBs scanned with Specim FX10 in the VNIR range.
- Two segmentation ground truth files: 'General' and 'Monoseg' for 4 classes of interest - 'others,' 'IC,' 'Capacitor,' and 'Connectors.'

## Data Access

To utilize the dataset, download it from the Rodare website: [Rodare](https://rodare.hzdr.de/record/2704), or from Zenodo: [Zenodo](https://zenodo.org/records/10617721).


## License

The code is licensed under the Apache-2.0 license. Any further development and application using this work should be opened and shared with the community.
