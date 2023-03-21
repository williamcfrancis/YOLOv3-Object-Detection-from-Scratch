# YOLOv3 Object Detection from scratch

## Overview
This is an implementation of YOLO (You Only Look Once), a fast, real-time object detection algorithm that is widely used in the field of computer vision. It is capable of detecting multiple objects in an image and assigning them semantic labels based on their class. The following image is an example of the output of an object detection model:
![image](https://user-images.githubusercontent.com/38180831/206100696-b4db529c-63e1-4c31-bcfb-348c8b3f5722.png)

Here, the different colors indicate different object classes.

If you want to learn more about YOLO, here are some useful resources:
[Original YOLO paper](https://arxiv.org/pdf/1506.02640.pdf) |
[Intuitive Explanation](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006) |
[YOLO Video Tutorial](https://www.youtube.com/watch?v=9s_FpMpdYW8&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=30) |
[Mean Average Precision](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173) |
[Intersection over Union](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection)

## Data
The data for this project consists of 10,000 street scene images and their corresponding labels. The images are 128x128x3 in dimension, and the labels include the semantic class and bounding box for each object in the image. Note that a small portion of these labels may be noisy, and the size of the training set is not large, so it may not be possible to learn a highly robust object detector.

To use the data, download the images from [here](https://drive.google.com/file/d/1cTxFrQ9o4dUKqY0WH2eUNtyRjtuSfWmd/view?usp=sharing) and the labels from this repository as labels.npz.

## Preprocessing
Before training the model, the labels must be converted into a ground truth matrix with dimension $8 \times 8 \times 8$. This is done as follows:
![image](https://user-images.githubusercontent.com/38180831/206101640-2f40d6d0-1311-4fce-b78d-54d51711ecef.png)

* The image is divided into $8 \times 8$ grid cells, with each cell representing a 16x16 patch in the original image.
* For simplicity, only one anchor box is used, with the same size as the grid cell. If the center of an object falls within a grid cell, that cell is responsible for detecting the object.
* Each anchor has 8 channels: Pr(Objectness), $x$, $y$, $w$, $h$, P(class=pedestrian), P(class=traffic light), and P(class=car
* Pr(Objectness) is the probability that this anchor is an object rather than background. "1" indicates an object, and "0" indicates background.
* The $x$ and $y$ coordinates represent the center of the bounding box relative to the bounds of the grid cell, and $w$ and $h$ are the width and height of the bounding box relative to the width and height of the image.
* The final three channels are for the semantic labels of the object, and are encoded using one-hot coding.
* If the anchor does not have an object (Pr=0), the values for channels 2-8 are not assigned, as they will not be used during training.
* The dimensions are ordered as (channels, x, y).

## Model Architecture
This model takes input with dimension of $128 \times 128 \times 3$ and outputs an activation with dimension of $8 \times 8 \times 8$.
![image](https://user-images.githubusercontent.com/38180831/206103922-3b1aa7ea-cbbd-4d9d-8f28-9e3b358599c8.png)

| Layer | Hyperparameters |
| :-: | :-: |
| conv1 | Kernel size $= 4 \times 4 \times 32$, stride $=2$, pad $=1$. Followed by BatchNorm and ReLU. |
| conv2 | Kernel size $= 4 \times 4 \times 64$, stride $=2$, pad $=1$. Followed by BatchNorm and ReLU. |
| conv3 | Kernel size $= 4 \times 4 \times 128$, stride $=2$, pad $=1$. Followed by BatchNorm and ReLU. |
| conv4 | Kernel size $= 4 \times 4 \times 256$, stride $=2$, pad $=1$. Followed by BatchNorm and ReLU. |
| conv5 | Kernel size $= 4 \times 4 \times 512$, stride $=2$, pad $=1$. Followed by BatchNorm and ReLU. |
| conv6 | Kernel size $= 4 \times 4 \times 1024$, stride $=2$, pad $=1$. Followed by BatchNorm and ReLU. |
| transposed_conv7 | Kernel size $= 4 \times 4 \times 256$, stride $=2$, pad $=1$. Followed by BatchNorm and ReLU. |
| transposed_conv8 | Kernel size $= 4 \times 4 \times 64$, stride $=2$, pad $=1$. Followed by BatchNorm and ReLU. |
| conv9 | Kernel size $= 3 \times 3 \times 8$, stride $=1$, pad $=1$. |

## Training Details
During training, the localization and classification errors are optimized jointly. The loss function is shown as below.  $i$ indicates number of grid cells and $j$ indicates number of anchor boxes at each grid cell. In our
case, there is only one anchor box at each grid cell and $B = 1$.
![image](https://user-images.githubusercontent.com/38180831/206102984-cf70e6d4-5c99-4c16-8161-9a928cb717b5.png)

* In our case there is only one anchor box at each grid, hence $B = 1$.
* $S^2 =$ total number of grid cells.
* $\mathbb{1}_{ij}^\text{obj} = 1$ if an object appears in grid cell $i$ and 0 otherwise.
* $\hat{C}_i =$ Box confidence score $=$ Pr(box contains object) $\times$ IoU
* IoU $=$ Intersection over union between the predicted and the ground truth.
* $\hat{p}_i(c) =$ conditional class probability of class $c$ in cell $i$.

$\lambda_\text{coord}$ and and $\lambda_\text{no obj}$ are two hyperparameters for coordinate predictions and non-objectness classification. We set $\lambda_\text{coord} = 5$ and and $\lambda_\text{no obj} = 0.5$.

Each grid cell predicts 1 bounding box, confidence score for those boxes and class conditional probabilities.

The confidence Score reflects the degree of confidence that the box contains an object and how accurate the box is. If no object exists in the cell then the confidence score should be 0 else the confidence score should be equal to the IOU between the predicted box and the ground truth box.

During training, I set a learning rate of 10e-3 using Adam optimizer with default beta 1 and beta 2. I also visualize the loss over training iterations. Based on the loss visualization, I train the model for 20 epochs.

## Post-Processing
During inference, the network is going to predict lots of overlapping redundant bounding boxes. To eliminate the redundant boxes, there are basically two steps:

1. Get rid of predicted boxes with low objectness probability (Pr $< 0.6$).
2. For each class, calculate the IoU for all the bounding boxes and cluster boxes with IoU > 0.5 as a group. For each group, find the one with highest Pr and suppress the other boxes. This is referred as non-max suppression.

<div><img src="https://github.com/LukasZhornyak/CIS680_files/raw/main/HW2/fig2_3.png"/></div>

To evaluate the performance of the YOLO implementation, I compute the mean Average Precision (mAP) of inference. Predicted bounding boxes are a match with ground truth bounding boxes if they share the same label and have an IoU with the ground truth bounding box of greater than 0.5. These matches can be used to calculate a precision/recall curve for each class. The Average Precision for a class is the area under this curve. The mean of these Average Precision values over all the classes in inference gives the mean Average Precision of the network.

<div><img src="https://github.com/LukasZhornyak/CIS680_files/raw/main/HW2/fig2_4.png"/></div>

## Results
#### Training Loss
![image](https://user-images.githubusercontent.com/38180831/206104182-61baa546-5485-4390-96f2-ffe70e8785cf.png)

#### Validation Loss
![image](https://user-images.githubusercontent.com/38180831/206104297-82ed6c83-3029-4c34-bbc4-0a4348f3f9f9.png)

#### Plot of the mean Average Precision over training
![image](https://user-images.githubusercontent.com/38180831/206104391-031df0aa-d905-41f8-8ff8-a27c9c11591b.png)

#### Raw output
![image](https://user-images.githubusercontent.com/38180831/206104889-8f716565-c3f5-4d7d-be56-5fbc909d1575.png)


#### Output after confidence thresholding and Non Maximum Suppression (NMS)
![image](https://user-images.githubusercontent.com/38180831/206104921-1705d29e-05ea-4b5d-ae72-dd2cdfd39dce.png)

#### Classwise mean Average Precision curves
![image](https://user-images.githubusercontent.com/38180831/206105074-1b426ee6-6c98-4a50-aafa-1a14f3767997.png)
![image](https://user-images.githubusercontent.com/38180831/206105128-5987a178-efa6-41f8-a9b4-75310db1cdec.png)
![image](https://user-images.githubusercontent.com/38180831/206105196-9faf2101-b976-4cae-9341-fa999771ba00.png)



