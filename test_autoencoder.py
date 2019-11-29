
import cv2
from visualize import visualize, show_image
from load_dataset import load_image
from keras.models import load_model
import matplotlib.pyplot as plt

# Load autoencoder model
autoencoder_faces = load_model('models/faces/faces.h5')
autoencoder_flowers = load_model('models/flowers/flowers.h5')
print(autoencoder_faces.summary())
print(autoencoder_flowers.summary())

# Testing
tst_img = load_image("test_images/puppy2.jpg")
tst_img = tst_img.astype('float32')/255.0 - 0.5
tst_img_32 = cv2.resize(tst_img, (32, 32))
tst_img_64 = cv2.resize(tst_img, (64, 64))
out1 = autoencoder_faces.predict(tst_img_32[None])[0]
out2 = autoencoder_flowers.predict(tst_img_64[None])[0]
#visualize(tst_img,encoder,decoder)

plt.subplot(2,2,1)
plt.title("Original")
show_image(tst_img_32)

plt.subplot(2,2,2)
plt.title("Reconstructed")
show_image(out1)

plt.subplot(2,2,3)
#plt.title("Original")
show_image(tst_img_64)

plt.subplot(2,2,4)
#plt.title("Reconstructed")
show_image(out2)
plt.show()