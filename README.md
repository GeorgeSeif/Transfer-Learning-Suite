# Transfer Learning Suite in Keras

## News

## Description
This repository serves as a Trasnfer Learning Suite. The goal is to easily be able to implement, train, and test new Transfer Learning Classification models! 
**Any suggestions to improve this repository, including any new segmentation models you would like to see are welcome!**

## Models
The following models are currently made available:




## Files and Directories


- **main.py:** Training and Prediction mode

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
    

## Results

These are some **sample results** for the Cats and Dogs dataset with 2 classes.
