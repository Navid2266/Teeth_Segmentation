# Teeth_Segmentation
Literature review, public datasets, model selection, testing, and evaluation for teeth segmentation

## introduction

Tooth segmentation serves as a critical prerequisite for clinical dental analysis and surgical procedures, enabling dentists to comprehensively assess oral conditions and diagnose pathologies. Panoramic X-rays are extensively used in dentistry to provide an inclusive view of the oral cavity, aiding in treatment planning for various dental conditions [2]. 

However, interpreting these images can be a laborious task, often diverting clinicians from vital clinical activities. The risk of misdiagnosis is substantial, especially since general practitioners may lack specialized training in radiology, and communication errors due to work exhaustion can exacerbate this. Over the past decade, deep learning has experienced significant advancements, with researchers introducing efficient models for automating and improving dental image analysis.

## Recent Models in Tooth Segmentation

In recent studies, the **U-Net** architecture has been widely employed for medical image segmentation, particularly in tooth segmentation. The blurred boundaries between teeth and the low contrast between teeth and alveolar bone in panoramic dental X-ray images make accurate tooth segmentation a challenging task [1]. Hou, S., et al. propose a dental panoramic X-ray image segmentation model using U-Net with context semantics and enhanced tooth boundaries. To improve feature extraction, U-Net often integrates pretrained backbones, such as **VGG16**, which have been trained on large-scale datasets, thus boosting segmentation accuracy even with limited dental imaging data.

Another effective model for tooth segmentation is **Mask R-CNN**. This region-based CNN, designed for object detection and instance segmentation, is particularly suited for tasks requiring precise boundary delineation at the tooth level. Built on the Faster R-CNN architecture, Mask R-CNN adds a branch for generating pixel-level masks for each detected object. Jader, G., et al. propose a segmentation system based on Mask R-CNN, achieving 99% accuracy on a 1500-image dataset using transfer learning strategies [2].

**SegNet**, another popular model reviewed by Liu, X., et al., is an encoder-decoder network specifically designed for pixel-level segmentation. SegNet utilizes a **VGG16**-based encoder to extract features and employs pooling indices from the encoder for upsampling the feature map in the decoder. This approach, rather than learning deconvolution filters, makes it particularly useful when medical data is scarce. However, the depooling process may lose some adjacent information during upsampling.

**MedSAM** (Medical Segment Anything Model) is a cutting-edge model that adapts the SAM (Segment Anything Model) framework for medical image segmentation. It uses few-shot learning to accurately segment anatomical structures, such as teeth, with minimal supervision. MedSAM is a valuable tool for automating and improving dental diagnostics by handling complex medical imaging tasks.

Finally, **ResNet** (Residual Network), an advanced CNN, uses deep architectures and innovative skip connections to solve the vanishing gradient problem [4]. Unlike earlier models like AlexNet and GoogleNet, ResNet architectures can have many more layers, such as **ResNet-50** and **ResNet-101**, which have been used in recent teeth segmentation studies. Meşeci, E., et al. compared ResNet-50 and ResNet-101 for tooth segmentation and achieved precision of 96.42% and an F1-score of 93.97% for tooth numbering using ResNet-101 [5].

## Dataset

Although there are many limitations in accessing public medical datasets, there are some valuable x-ray teeth image datasets that can be beneficial for training segmentation models:

1. **Hamamci, I.E., et al.** published a valuable dataset consisting of 1,005 panoramic X-rays, which is partitioned into training, validation, and testing subsets, comprising 705, 50, and 250 images, respectively [6].
2. **Helli, S., and A. Hamamcı** shared a dataset that includes original X-ray images and their corresponding ground truth masks for training purposes. This dataset contains 116 instances for each image and its mask [7].
3. **Wang, X., et al.** also published an X-ray image dataset [8].

For the purpose of training models in this project, all models were primarily tested on the **Helli, S., and A. Hamamcı** dataset, which includes custom ground truth masks for segmentation tasks. for testing models download the dataset and locate it in /Data/images.

You can download the dataset used in this project from [LINK](https://data.mendeley.com/datasets/hxt48yk462/1).

# Selected models, testing, and evaluation 

This repository contains 4 model implementations for segmenting teeth from dental images, supporting advanced analysis in dental health applications.

## Directory Structure

```plaintext
Teeth-Segmentation-project/
│
├── Data/                   # Directory for storing dataset files
├── Mask-RCNN/              # Folder containing Mask-RCNN implementation
│   └── Mask-R-CNN.ipynb    # Main notebook for Mask-RCNN model training and segmentation
├── MedSam/                 # Folder containing MedSAM model implementation
│   └── MedSam_model.ipynb  # Main notebook for training the MedSAM model
├── SegNet-VGG16/           # Folder containing SegNet model with VGG16 backbone
│   └── Segnet_model.ipynb  # Main notebook for SegNet model training and segmentation
├── U-Net_VGG16/                  # Folder containing VGG16-based U-Net segmentation model
│   └── U-Net.ipynb         # Main notebook for U-Net model training and segmentation
└── README.md               # Project README file
```

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Navid2266/Teeth_Segmentation.git
    ```
2. Navigate into the project directory:
    ```bash
    cd Teeth_Segmentation
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Model Descriptions

- **MedSAM**: Uses MedSAM to segment teeth from dental images.
- **U-Net**: Utilizes U-Net architecture with VGG16 as the backbone.
- **SegNet-VGG16**: Utilizes SegNet architecture with VGG16 as the backbone.
- **Mask-RCNN**: Implements Mask R-CNN for instance segmentation.

## MedSAM Model

In this part, the MedSAM model to segment teeth from medical images is implemented, enabling further analysis for dental health applications. The code provided requires pre-trained model checkpoints and a new function has been added to ensure compatibility with .dcm (DICOM) files.

### Teeth Segmentation using YOLO and MedSam Model

This repository provides a complete solution for dental image segmentation using a fine-tuned YOLO object detection model and the MedSam model. The process is designed for automatic segmentation of individual teeth, using YOLO for detection and MedSam for detailed segmentation.

### 1. Fine-Tuned YOLO Model for Automatic Segmentation
The YOLO model has been fine-tuned to automatically detect and segment individual teeth in dental images. This eliminates the need for manual annotation, making the segmentation process faster and more efficient.

- The main YOLO object detection model is located in `YOLO-Model.py`.
- To test the YOLO model, run the `test-yolo.ipynb` notebook. This notebook will evaluate the model's performance in detecting bounding boxes around the teeth.

### 2.Download the Model Checkpoints

Ensure that the pre-trained MedSAM model checkpoints are downloaded. To automatically download the checkpoints, uncomment the relevant line in the script.

### 3.Testing the YOLO Model

If you would like to test only the YOLO model for tooth detection (without segmentation), use the `test-yolo.ipynb` notebook. This will give you the bounding boxes for the detected teeth.

### 4. Automated Segmentation with MedSam

Once the teeth have been detected using YOLO, the MedSam model is used to automatically perform segmentation on each individual tooth. This allows for a fully automated segmentation process.

To use the automated segmentation workflow:

1. Navigate to the `MedSam` folder:
   ```bash
   cd MedSam

2. Open and run the `Automated_MedSam.ipynb` notebook to segment the teeth detected by YOLO automatically.

### 5.Using Annotated Files with MedSam

If you already have manually created YOLO annotation files (bounding boxes), you can still use the previous approach for segmentation:
- Run MedSam.ipynb as before to use the YOLO bounding boxes for segmentation.

### Evaluation

The model’s segmentation accuracy is measured by calculating the Dice score between the binarized model output and the ground truth mask and getting 84.2% accuracy.

![Screenshot](https://github.com/Navid2266/Teeth_Segmentation/raw/main/MedSam/Results/Screenshot%202024-11-11%20174904.png)


## U-Net Model

This part contains an implementation of a U-Net model with a VGG16 backbone for segmenting teeth from dental images. The model combines U-Net’s encoder-decoder structure with VGG16’s pre-trained feature extraction, making it suitable for accurate segmentation in dental health applications.

1. Navigate to the `VGG16` folder.
    ```bash
    cd VGG16
    ```
2. Open `U-Net.ipynb` in Jupyter Notebook to access the main model training and segmentation file.

3. **Preprocess the Images and Masks**
   images and masks are resized and converted to single-channel format by prepare_images.py and masks_prepare.py. This is necessary for consistency with the U-Net model input.

4.**Data Augmentation**
    Data augmentation is applied to increase the variety in the dataset. Due to GPU limitations, augmentation is done once prior to training.

5.**Load the VGG16 Weights**
    The VGG16 layers are loaded and frozen in the model, serving as the encoder in the U-Net structure. This leverages VGG16’s pre-trained feature extraction capabilities.

6.**Run the Segmentation**
    The U-Net model with VGG16 is trained for only 20 epochs using 60 images, which is insufficient to achieve the desired level of accuracy. This setup is primarily intended for reporting the model's performance in the project. For improved results, especially when using a system with a proper GPU, the number of epochs and batch size should be increased..

### Evaluation

The model’s segmentation accuracy was 86.9% by the dice score.

![Screenshot](https://github.com/Navid2266/Teeth_Segmentation/raw/main/U-Net_VGG16/Screenshot%202024-11-12%20122711.png)
![Screenshot](https://github.com/Navid2266/Teeth_Segmentation/raw/main/U-Net_VGG16/Screenshot%202024-11-12%20122739.png)

## SegNe model

The SegNet model with a VGG16 backbone is designed for teeth segmentation in dental images. By utilizing VGG16's pre-trained layers as an encoder, SegNet effectively captures image features, while its decoder reconstructs the segmentation map for precise tooth boundaries. SegNet is more memory-efficient than U-Net with its encoder-decoder structure but lacks skip connections, making it less precise for highly detailed tasks, as seen in the results.

1. Navigate to the `SegNet-VGG16` folder.
    ```bash
    cd SegNet-VGG16
    ```
2. Open `Segnet_model.ipynb` in Jupyter Notebook to access the main model training and segmentation file.

3. **Run the model**
    The SegNet model shares a structure similar to U-Net, so there is no need for further settings. Due to its simpler architecture, the SegNet model requires less training time. However, under the same training settings (number of epochs and input samples), its results were less precise compared to the U-Net model.

### Evaluation

The model's average dice score was 80.2% which is 6.7% less than the U-Net model.

![Screenshot](https://github.com/Navid2266/Teeth_Segmentation/raw/main/SegNet-VGG16/Screenshot%202024-11-12%20122621.png)


## Mask R-CNN model

Mask R-CNN is a robust model for teeth segmentation that combines object detection and instance segmentation, enabling precise identification and segmentation of individual teeth in dental images. This model's potential is highlighted by the work of Jader, G., et al., who achieved up to 98% accuracy [2], demonstrating its high effectiveness for detailed segmentation tasks. However, Mask R-CNN's high computational cost requires a powerful GPU for efficient training. Despite the time invested, I was unable to test the model with my dataset due to library incompatibilities and the lack of adequate GPU resources. With further configuration, though, this model could become usable for application.

1. Navigate to the `Mask-RCNN` folder.
    ```bash
    cd Mask-RCNN
    ```
2. Open `Mask-R-CNN.ipynb` in Jupyter Notebook to access the main model training and segmentation file.

3. **Run the model**
    Open the notebook to configure and run the model for teeth segmentation. Note that the model requires further configuration to work effectively on this specific dataset.

### Evaluation
While I couldn’t run the model on my dataset, published results [2] indicate that Mask R-CNN can achieve up to 99% accuracy, highlighting its precision and suitability for high-quality dental image segmentation.

## Conclusion
In this project, I evaluated four models for teeth segmentation: U-Net, SegNet, MedSAM, and Mask R-CNN. Each model has unique strengths and trade-offs in terms of accuracy, computational cost, and flexibility.

U-Net, the most widely used model in medical image segmentation, performed well even with a small dataset and fewer epochs, achieving a Dice score of 86.7%. This highlights U-Net’s robustness in producing accurate segmentations with limited data. SegNet, a simpler model that omits skip connections, provided a faster training time—approximately 10 minutes less than U-Net under the same training configuration—although it achieved a slightly lower accuracy of 80.2%. This reduced complexity makes SegNet a suitable option when computational efficiency is a priority.

MedSAM, on the other hand, offered a Dice score of 84.2% without the need for training, making it a convenient choice when quick deployment is needed. However, the model’s parameters are not accessible for fine-tuning, which limits its adaptability to specific datasets. Lastly, Mask R-CNN, was not tested on my dataset due to resource constraints and library incompatibilities, but it has shown excellent performance in the literature for teeth segmentation tasks up to 99% accuracy.

Overall, the results highlight that while U-Net remains a reliable choice for accurate segmentation, each model brings unique advantages depending on the project requirements, such as computational resources, training flexibility, and dataset size.

## references
1.	Hou, S., et al., Teeth U-Net: A segmentation model of dental panoramic X-ray images for context semantics and contrast enhancement. Computers in Biology and Medicine, 2023. 152: p. 106296.
2.	Jader, G., et al. Deep instance segmentation of teeth in panoramic X-ray images. in 2018 31st SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI). 2018. IEEE.
3.	Liu, X., et al., A review of deep-learning-based medical image segmentation methods. Sustainability, 2021. 13(3): p. 1224.
4.	Umer, F., S. Habib, and N. Adnan, Application of deep learning in teeth identification tasks on panoramic radiographs. Dentomaxillofacial Radiology, 2022. 51(5): p. 20210504.
5.	Meşeci, E., et al., Tooth Detection and Numbering with Instance Segmentation in Panoramic Radiographs. 2021.
6.	Hamamci, I.E., et al., DENTEX: an abnormal tooth detection with dental enumeration and diagnosis benchmark for panoramic X-rays. arXiv preprint arXiv:2305.19112, 2023.
7.	Helli, S. and A. Hamamcı, Tooth instance segmentation on panoramic dental radiographs using u-nets and morphological processing. Düzce Üniversitesi Bilim ve Teknoloji Dergisi, 2022. 10(1): p. 39-50.
8.	Wang, X., et al., Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation. Neurocomputing, 2023. 540: p. 126208.


