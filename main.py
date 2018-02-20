from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout
from keras import backend as K

from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import matplotlib.pyplot as plt
import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--is_training', type=str2bool, default=True, help='Whether we are training or testing')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="Pets", help='Dataset you are using.')
parser.add_argument('--resize_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout ratio')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=None, help='Whether to randomly zoom in for data augmentation')
parser.add_argument('--model', type=str, default="VGG16")
args = parser.parse_args()


BATCH_SIZE = args.batch_size
WIDTH = args.resize_width
HEIGHT = args.resize_height
FC_LAYERS = [1024, 1024]
TRAIN_DIR = args.dataset + "/train/"
VAL_DIR = args.dataset + "/val/"

preprocessing_function = None
base_model = None

def get_subfolders(dataset):
    subfolders = os.listdir(dataset)
    return subfolders

def get_num_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def build_finetune_model(base_model, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in FC_LAYERS:
        x = Dense(fc, activation='relu')(x) # New FC layer, random init
        x = Dropout(args.dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x) # New softmax layer
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()



# Prepare the model
if args.model == "VGG16":
    from keras.applications.vgg16 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "VGG19":
    from keras.applications.vgg19 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "ResNet50":
    from keras.applications.resnet50 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionV3":
    from keras.applications.inception_v3 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "Xception":
    from keras.applications.xception import preprocess_input
    preprocessing_function = preprocess_input
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionResNetV2":
    from keras.applications.inceptionresnetv2 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "MobileNet":
    from keras.applications.mobilenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet121":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet169":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet201":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetLarge":
    from keras.applications.nasnet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetMobile":
    from keras.applications.nasnet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))


# Prepare data generators
train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocessing_function,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
val_datagen = ImageDataGenerator(
  preprocessing_function=preprocessing_function,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)


train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)

validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)

class_list = get_subfolders(TRAIN_DIR)

finetune_model = build_finetune_model(base_model, len(class_list))

adam = Adam(lr=0.0001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

num_train_images = get_num_files(TRAIN_DIR)
num_val_images = get_num_files(VAL_DIR)


history = finetune_model.fit_generator(train_generator, epochs=args.num_epochs, workers=8, steps_per_epoch=num_train_images, 
    validation_data=validation_generator, validation_steps=num_val_images, class_weight='auto', shuffle=True)


plot_training(history)