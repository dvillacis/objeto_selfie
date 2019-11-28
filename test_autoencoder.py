
import cv2
from visualize import visualize
from load_dataset import load_image
from keras.models import load_model

# Load autoencoder model
autoencoder = load_model('models/faces/faces.h5')
encoder = load_model('models/faces/encoder_faces.h5')
decoder = load_model('models/faces/decoder_faces.h5')
print(autoencoder.summary())

# Testing
tst_img = load_image("test_images/puppy.jpg")
tst_img = tst_img.astype('float32')/255.0 - 0.5
tst_img = cv2.resize(tst_img, (32, 32))
visualize(tst_img,encoder,decoder)