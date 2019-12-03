import numpy as np
import cv2
from keras.layers import Input
from keras.models import Model, save_model
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from load_dataset import load_folder_dataset, load_flowers_dataset
from model import build_autoencoder
from visualize import visualize, show_image

# Training dataset
# http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
IMAGES_NAME = "datasets/flowers/102flowers.tgz"
#IMAGES_NAME = "datasets/cocoon"


X = load_flowers_dataset(IMAGES_NAME)

# Center images by 0
X = X.astype('float32')/255.0 - 0.5

# Dataset splitting
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

IMG_SHAPE = X.shape[1:]
CODE_SIZE = 64
encoder, decoder = build_autoencoder(IMG_SHAPE, CODE_SIZE)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
print(autoencoder.summary())

# Model training
history = autoencoder.fit(x=X_train, y=X_train, epochs=20, validation_data=[X_test, X_test])

# Save Model
autoencoder.save('models/flowers/flowers.h5')


