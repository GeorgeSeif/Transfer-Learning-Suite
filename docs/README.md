# Transfer Learning Suite in Keras

## News

## Description
This repository serves as a Transfer Learning Suite. The goal is to easily be able to perform transfer learning using any built-in Keras image classification model! 
**Any suggestions to improve this repository or any new features you would like to see are welcome!**

You can also check out my [Semantic Segmentation Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite).

## Models
All of the Keras built in models are made available:

| Model  | Size  |  Top-1 Accuracy  |  Top-5 Accuracy  |  Parameters  |  Depth  |
| -------------     | ------------- | -------------| ------------- | ------------- | ------------- |
| VGG16    | 528 MB    |  0.715    | 0.901    | 138,357,544    | 23    |
| VGG19    | 549 MB    |  0.727    | 0.910    | 143,667,240    | 26    |
| ResNet50    |  99 MB    | 0.759    | 0.929    | 25,636,712    |  168    |
| Xception    |  88 MB    | 0.790    | 0.945    | 22,910,480     | 126    |
| InceptionV3    | 92 MB    | 0.788    | 0.944    | 23,851,784    |  159    |
| InceptionResNetV2    | 215 MB    |  0.804    | 0.953   | 55,873,736    |  572    |
| MobileNet    | 17 MB    | 0.665    | 0.871    | 4,253,864    | 88    |
| DenseNet121    | 33 MB    | 0.745    | 0.918    | 8,062,504    | 121    |
| DenseNet169    | 57 MB    | 0.759    | 0.928    | 14,307,880    |  169    |
| DenseNet201    | 80 MB    | 0.770    | 0.933    | 20,242,984    |  201    |
| NASNetMobile    | 21 MB    | NA    | NA    | 5,326,716    |  NA    |
| NASNetLarge    | 342 MB    | NA    | NA    | 88,949,818    |  NA    |


## Files and Directories


- **main.py:** Training and Prediction mode

- **utils.py:** Helper utility functions

- **checkpoints:** Checkpoint files for each epoch during training

- **Predictions:** Prediction results

## Installation
This project has the following dependencies:

- Numpy `sudo pip install numpy`

- OpenCV Python `sudo apt-get install python-opencv`

- TensorFlow `sudo pip install --upgrade tensorflow-gpu`

- Keras `sudo pip install keras` 

## Usage
The only thing you have to do to get started is set up the folders in the following structure:

    ├── "dataset_name"                   
    |   ├── train
    |   |   ├── class_1_images
    |   |   ├── class_2_images
    |   |   ├── class_X_images
    |   |   ├── .....
    |   ├── val
    |   |   ├── class_1_images
    |   |   ├── class_2_images
    |   |   ├── class_X_images
    |   |   ├── .....
    |   ├── test
    |   |   ├── class_1_images
    |   |   ├── class_2_images
    |   |   ├── class_X_images
    |   |   ├── .....

Then you can simply run `main.py`! Check out the optional command line arguments:

```
usage: main.py [-h] [--num_epochs NUM_EPOCHS] [--mode MODE] [--image IMAGE]
               [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
               [--resize_height RESIZE_HEIGHT] [--resize_width RESIZE_WIDTH]
               [--batch_size BATCH_SIZE] [--dropout DROPOUT] [--h_flip H_FLIP]
               [--v_flip V_FLIP] [--rotation ROTATION] [--zoom ZOOM]
               [--shear SHEAR] [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --mode MODE           Select "train", or "predict" mode. Note that for
                        prediction mode you have to specify an image to run
                        the model on.
  --image IMAGE         The image you want to predict on. Only valid in
                        "predict" mode.
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --resize_height RESIZE_HEIGHT
                        Height of cropped input image to network
  --resize_width RESIZE_WIDTH
                        Width of cropped input image to network
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --dropout DROPOUT     Dropout ratio
  --h_flip H_FLIP       Whether to randomly flip the image horizontally for
                        data augmentation
  --v_flip V_FLIP       Whether to randomly flip the image vertically for data
                        augmentation
  --rotation ROTATION   Whether to randomly rotate the image for data
                        augmentation
  --zoom ZOOM           Whether to randomly zoom in for data augmentation
  --shear SHEAR         Whether to randomly shear in for data augmentation
  --model MODEL         Your pre-trained classification model of choice

```
    
