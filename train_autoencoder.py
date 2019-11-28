import numpy as np
import cv2
from keras.layers import Input
from keras.models import Model, save_model
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from load_dataset import load_lfw_dataset, load_image
from model import build_autoencoder
from visualize import visualize, show_image

# Training dataset
# http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
ATTRS_NAME = "datasets/lfw/lfw_attributes.txt"

# http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
IMAGES_NAME = "datasets/lfw/lfw-deepfunneled.tgz"

# http://vis-www.cs.umass.edu/lfw/lfw.tgz
RAW_IMAGES_NAME = "datasets/lfw/lfw.tgz"


X,attr = load_lfw_dataset(RAW_IMAGES_NAME, IMAGES_NAME, ATTRS_NAME, use_raw=True, dimx=32, dimy=32)

# Center images by 0
X = X.astype('float32')/255.0 - 0.5

# Dataset splitting
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

IMG_SHAPE = X.shape[1:]
encoder, decoder = build_autoencoder(IMG_SHAPE, 1000)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
print(autoencoder.summary())

# Model training
history = autoencoder.fit(x=X_train, y=X_train, epochs=15, validation_data=[X_test, X_test])

# Testing
tst_img = load_image("puppy.jpg")
tst_img = tst_img.astype('float32')/255.0 - 0.5
tst_img = cv2.resize(tst_img, (32, 32))
visualize(tst_img,encoder,decoder)

# Save Model
#autoencoder.save('models/faces.h5')
#save_model(autoencoder,"faces.h5")


