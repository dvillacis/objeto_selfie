import cv2
import numpy as np
import configparser
from keras.models import load_model

def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def load_keras_model(model_path):
    m = load_model(model_path)
    print(m.summary())
    return m


def main(config):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("OSelfie v2.0",cv2.WINDOW_NORMAL)
    WIDTH = int(cam.get(3))
    HEIGHT = int(cam.get(4))

    MODEL_PATH = 'models/cocoon/cocoon.h5'
    ae = load_keras_model(MODEL_PATH)

    while True:
        ret,frame = cam.read()
        frame = cv2.flip(frame,1)
    
        if not ret:
            break
        k = cv2.waitKey(1)
        K = k%256

        if K == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        # Modification step
        tst = frame.astype('float32')/255.0 - 0.5
        tst = cv2.resize(tst, (32, 32))
        out = ae.predict(tst[None])[0]
        cv2.imshow("OSelfie v2.0",cv2.resize(out,(WIDTH,HEIGHT)))
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = read_config('oselfie.config')
    main(config)
    