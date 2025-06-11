# ToothFairy2: Segmenting Maxillofacial Structures in CBCT Volumes

[[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bolelli_Segmenting_Maxillofacial_Structures_in_CBCT_Volumes_CVPR_2025_paper.pdf) [[Bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:SxlqJ7rfTE0J:scholar.google.com/&output=citation&scisdr=CgJJTysEELySm4m8Y0c:AAZF9b8AAAAAaEm6e0cq3jHuqDOuWOPA3ZDYmk8&scisig=AAZF9b8AAAAAaEm6e4MN6LfOEgd980rePIcsXa8&scisf=4&ct=citation&cd=-1&hl=en)

<figure>
 <img style="float: left" src="figs/CBCT.png" alt="Side view" width="70%">
 <figcaption><em>Caption</em></figcaption>
</figure>

ToothFairy2-CVPR is the official benchmark repository for the [ToothFairy2 dataset](https://ditto.ing.unimore.it/toothfairy2/), a large-scale, publicly available collection of CBCT scans with voxel-level 3D annotations for 42 maxillofacial structures, including teeth, jawbones, sinuses, and alveolar canals. This benchmark evaluates state-of-the-art segmentation methods—ranging from CNNs to transformers and Mamba-based models on maxillofacial anatomical regions, using the ToothFairy2 dataset for training and test.
All benchmarked methods are implemented within the [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet), with targeted architectures implemented by us in the [`benchmark_networks/nnunetv2/nets`](benchmark_networks/nnunetv2/nets) folder and their corresponding trainer classes in the [`benchmark_networks/nnunetv2/training/nnUNetTrainer`](benchmark_networks/nnunetv2/training/nnUNetTrainer) directory.