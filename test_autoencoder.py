
from visualize import load_image
from keras.models import load_model

# Load autoencoder model
autoencoder = load_model('models/faces.h5')

# Testing
tst_img = load_image('test_images/puppy.jpg')
visualize(tst_img,encoder,decoder)