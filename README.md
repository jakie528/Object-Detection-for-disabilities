# Yolo5/8 in Indoor and outdoor usage

Contributors: Ziqi Liao, Xinmiao Xiong, Pin chun Lu, Yi Wen Chen

[Indoor: Electronic Device Detection](#indoor-electronic-device-detection)

[Outdoor: Wildfires project](#outdoor-wildfires-project)

# Indoor: Electronic Device Detection

## Background

This section serves as our base case for comparing with the outdoor Wildfires project. To understand how YOLO usage differs between indoor and outdoor scenarios, we have tested indoor scenarios with fixed-angle cameras and first-person views. We have used YOLOv5 and YOLOv8 for this purpose.

## About YOLO
YOLO (You Only Look Once) is a state-of-the-art, real-time object detection algorithm. It treats object detection as a regression problem, using a single neural network to predict bounding boxes and class probabilities directly from full images in one evaluation. This unified architecture enables end-to-end training and real-time speeds while maintaining high average precision.
We implement the object detection in the YOLOv5 and YOLOv8 model for comparison.


## Dataset Preparation
YOLO models are typically pre-trained on the `COCO dataset`, a large-scale object detection, segmentation, and captioning dataset. COCO stands for Common Objects in Context and includes over 330K images with 80 object categories, providing a robust foundation for training object detection models.
- [OpenMMLab](https://github.com/open-mmlab): Our detection code uses [MMEngine](https://github.com/open-mmlab/mmengine) and our model is built upon [MMYOLO](https://github.com/open-mmlab/mmyolo).

## Environment Set Up
We utilized Google Colab and its NVIDIA Tesla T4 GPUs. GPUs are built with CUDA that can accelerate our data processing tasks and enhance model performance and efficiency. This setup accelarate the hourly computation into minutes and ensures optimal execution.
- Available here with Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19zZqunRLepQalh7REmbcu7dlfxJ4yVL0?usp=sharing)

### Set up YOLOv5 model
- Setup from the Ultralytics, Command: `git clone https://github.com/ultralytics/yolov5.git`,  and installing the dependencies from `requirements.txt` file.
- Execute the YOLOv5 object detection model with 'yolov5s.pt' pretrained weights to identify objects in images and videos. `python detect.py --weights yolov5s.pt --img 416 --conf 0.4 --source /content/test_yolo.jpeg`
  - Run the `detect.py`.
  - Adjust the image resolution to `416 x 416 pixels` to get a good mix of speed and accuracy.The size must be a multiple of 32. The choice of image size can affect both the speed of the detection process and its accuracy.[1]
  - Set a confidence `threshold at 0.4` to ignore detections that were not very certain.

### Set up the YOLOv8
- Installs PyTorch, Torchvision, and specific versions of the Ultralytics packages necessary for running YOLOv8.
```
!pip install torch torchvision
!pip install -U ultralytics
!pip install yolov5
!pip install ultralytics==8.0.196
```

**Pre-training on COCO:** This involves training the YOLO model on the COCO dataset before it is fine-tuned for specific tasks. This is the one that we have used in the stage of base case.

~~**Fine-tuning:** After pre-training, YOLO models are often fine-tuned on specific datasets tailored to particular use cases, such as indoor object recognition or wildfire detection. This step adjusts the model's weights to better detect objects relevant to the specific application.~~

## Model Training and Result
We selected four diverse samples and a video that would mimic real-world variability. we utilized pre-trained models from Ultralytics. Samples and a video as sources are put in the pre-trained model in the YOLOv5 and YOLOv8 separately. Then we conducted comparative performance analysis of YOLOv5 and YOLOv8 in Object Detection.

In each sample picture, we first count the total number of objects that are actually there and called as "Total Objects". Next, we see how many objects the model identifies and called them as "Detected Objects". We also count how many of these detected objects are coorectly identified, nameing them "Correctly Identified Objects". To measure how accurate the model is, we use two rates: the correctness rate and the detection rate. 
 1. `correctness rate (%) = Correctly Identified Objects / Total Objects in the Photo * 100`
 2. `detection rate(%) = Detected Objects / Total Objects in the Photo * 100`

 These calculations help us understand how well the model works in identifying and detecting objects in photos.
```
|- ..
|- yolo5/
|- sample_data/
|- sample1.jpg
|- sample2.jpg
|- sample3.jpg
|- sample4.jpg
|- video1.MOV
```


### Photos - 4 samples

| Model  | Sample   | Total Objects | Detected Objects | Correctly Identified Objects | Correctness Rate (%) | Detection Rate (%) |
|--------|----------|---------------|------------------|-----------------|----------------------|--------------------|
| YOLOv5 | Sample 1 | 4             | 5                | 4               | 100                  | 125                |
| YOLOv8 | Sample 1 | 4             | 4                | 3               | 75                   | 100                |
| YOLOv5 | Sample 2 | 3             | 2                | 2               | 100                  | 66.67              |
| YOLOv8 | Sample 2 | 3             | 3                | 2               | 66.67                | 100                |
| YOLOv5 | Sample 3 | 5             | 3                | 2               | 40                   | 60                 |
| YOLOv8 | Sample 3 | 5             | 6                | 3               | 60                   | 120                |
| YOLOv5 | Sample 4 | 5             | 3                | 3               | 100                  | 60                 |
| YOLOv8 | Sample 4 | 5             | 4                | 2               | 40                   | 80                 |

Table 1: Comparative Performance Analysis of YOLOv5 and YOLOv8 in Object Detection.

The analysis evaluates YOLOv5 and YOLOv8's object detection performance across four samples, using metrics like Correctness Rate and Detection Rate. YOLOv5 showed higher accuracy in correctly identifying objects, with particularly strong performance in Samples 1, 2, and 4. In contrast, YOLOv8 detected more objects overall but had more false positives, leading to a lower correctness rate. This suggests YOLOv5 might be better for applications needing precise identification, while YOLOv8 could be useful where capturing as many objects as possible is crucial. Optimizing these models through better training and parameter tuning can enhance their application-specific performance.

![](/assets/sample1_2.jpg)

![](/assets/sample3_4.jpg)
Figure 1: Comparison of object detection by YOLOv5 and YOLOv8 across four samples, highlighting differences in sensitivity and precision.

<video src="/assets/video1yolov5.mp4" width="320" height="240" controls></video>

<video src="/assets/videoyolov8.avi" width="320" height="240" controls></video>

In our analysis of video using YOLOv5 and YOLOv8, the focus is on detecting cellphones. YOLOv5 processes frames quickly (10-20 milliseconds per frame) but often misidentifies objects. For example, it confuses computer screens with TVs and laptops with books or keyboards. On the other hand, YOLOv8, although slightly slower (10-30 milliseconds per frame), demonstrates greater accuracy in detecting a broader range of objects, including cellphones and laptops. Both models incorrectly identifies EarPods as mice. This shows that while YOLOv5 and YOLOv8 excel in detecting trained objects like cellphones, they still mislabel items outside their training parameters.

## Findings

- YOLOv5 provides higher accuracy but does not detect everything in the frame.
- YOLOv8 provides lower accuracy but detects everything in the frame.
- Both YOLOv5 and YOLOv8 perform well in indoor detection when the object is at a certain distance and has the right angle with the camera.
- In first-person view detection, the performance is not as good due to two main reasons:
  - The distance is different from the COCO dataset used for YOLO pre-training.
  - The person wearing the recording glasses is moving, and the movement and changing angles might blur the frame.


# Outdoor: Wildfires project

## Background
Wildfires are increasing around the globe in frequency, severity, and duration, heightening the need to understand the health effects of wildfire exposure. The risk of wildfires grows in extremely dry conditions, such as drought, heat waves and during high winds.
With climate change leading to warmer temperatures and drier conditions and the increasing urbanization of rural areas, the fire season is starting earlier and ending later. Wildfire events are getting more extreme in terms of acres burned, duration and intensity, and they can disrupt transportation, communications, water supply, and power and gas services.


## Motivation
- A comprehensive sensing framework is needed to assess wildfire risks more accurately and enable strategic decisions
  - Commonly used sensors enable (temperature, humidity, particle, etc.) short-range detection.
  - Long-range detection (~1km-10km) can be enabled using cameras.
- Challenges in edge-devices:
  - Resource-constraint devices.
  - Communication is costly.

## Setup and Demo
[SmokeyNet](https://arxiv.org/pdf/2112.08598): Our classification model is built upon [SmokeyNet] framework. Please refer to their work of managing the environment.
[Smoke Detection with YOLO] Please refer to the config folder and mmyolo's documentation for running train/val configs. Using mmyolo's inference demo for testing some images.
[Smoke Classification of TileToImage] Please refer to the train_script for training. And then load the ckpt file for inference. Checkpoint file can be provided through email link.

## Methodology (method, data, evaluation metrics)
If applicable, highlight how it differs from prior work (new experiments, new methods, etc)
Dataset: www.hpwren.ucsd.edu/HPWREN-FIgLib
Reference paper: FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real-Time Wildland Fire Smoke Detection (arxiv.org)
Method: Object Detection + TileToImage binary classification
Difference: We make the inference model on Jetson and to estimate the energy consumption.
Visual: the gif in presentation slide
Demos: on Github

## Models Evaluated

- **ResNet34**: This is a 34-layer Residual Network model, commonly used in image recognition tasks for its efficiency and accuracy.
- **YOLOv8**: The latest version of the You Only Look Once model, YOLOv8 is designed for enhanced real-time object detection.
- **ResNet+ViT**: A hybrid architecture that combines the Residual Network (ResNet) and Vision Transformer (ViT) to capitalize on both their strengths in processing image data.


## Performance Metrics

- **Paras (Parameters)**: Represents the total number of trainable parameters within the model. This metric indicates the model's complexity and memory requirements.
- **Time**: Measures the time required for one iteration during training, impacting the overall training speed.
- **Acc (Accuracy)**: Denotes the percentage of correct predictions made by the model on the test dataset.
- **F1 Score**: A metric that considers both precision and recall to provide a balance between the two by calculating their harmonic mean.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. High precision relates to a low rate of false positives.


## Detailed Performance Table

| Model        | Parameters | Time per Iteration | Accuracy (%) | F1 Score | Precision (%) |
|--------------|------------|--------------------|--------------|----------|---------------|
| ResNet34     | 22.3M      | 50.4 ms/iter       | 75.5         | 73.2     | 82.0          |
| YOLOv8       | 43.6M      | 80.3 ms/iter       | 68.0         | 45.3     | 87            |
| ResNet+ViT   | 60.2M      | 65.2 ms/iter       | 81           | 76       | 89            |

## Summary

This table highlights the trade-offs between model complexity, performance, and training efficiency. It provides insights into which model might be most appropriate based on the specific requirements and constraints of the task at hand.

## Discussion
The detection model is based on YOLOV8. It can produce good false positive rate but missing too many real signals. And it requires more time to compute the possible boxes. --
ResNet+ViT enable the model to learn the spatial information and can produce better acc at a higher inference speed.
## Reference of Dataset
The main dataset is [HPWREN](https://www.hpwren.ucsd.edu/FIgLib).
The mini set is [AI-Humankind](https://public.roboflow.com/object-detection/wildfire-smoke).

# Reference
[1] ultralytics/yolov5: v7.0 - YOLOv5 SOTA Realtime Instance Segmentation. (2022, November 23). Zenodo. https://doi.org/10.5281/zenodo.7347926


<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>


