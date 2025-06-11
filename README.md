# ToothFairy2: Segmenting Maxillofacial Structures in CBCT Volumes

[[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bolelli_Segmenting_Maxillofacial_Structures_in_CBCT_Volumes_CVPR_2025_paper.pdf) [[Bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:SxlqJ7rfTE0J:scholar.google.com/&output=citation&scisdr=CgJJTysEELySm4m8Y0c:AAZF9b8AAAAAaEm6e0cq3jHuqDOuWOPA3ZDYmk8&scisig=AAZF9b8AAAAAaEm6e4MN6LfOEgd980rePIcsXa8&scisf=4&ct=citation&cd=-1&hl=en)

<figure>
 <img style="float: left" src="figs/CBCT.png" alt="Side view" width="100%">
 <figcaption><em>Caption</em></figcaption>
</figure>

Cone-beam computed tomography (CBCT) is a standard imaging modality in orofacial and dental practices, providing essential 3D volumetric imaging of anatomical structures, including jawbones, teeth, sinuses, and neurovascular canals. Accurately segmenting these structures is fundamental to numerous clinical applications, such as surgical planning and implant placement. However, manual segmentation of CBCT scans is time-intensive and requires expert input, creating a demand for automated solutions through deep learning. Effective development of such algorithms relies on access to large, well-annotated datasets, yet current datasets are often privately stored or limited in scope and considered structures, especially concerning 3D annotations. This paper proposes ToothFairy2, a comprehensive, publicly accessible CBCT dataset with voxel-level 3D annotations of 42 distinct classes corresponding to maxillofacial structures. We validate the dataset by benchmarking state-of-the-art neural network models, including convolutional, transformer-based, and hybrid Mamba-based architectures, to evaluate segmentation performance across complex anatomical regions. Our work also explores adaptations to the nnU-Net framework to optimize multi-class segmentation for maxillofacial anatomy. The proposed dataset provides a fundamental resource for advancing maxillofacial segmentation and supports future research in automated 3D image analysis in digital dentistry.

