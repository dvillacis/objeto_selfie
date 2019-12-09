import cv2
import numpy as np
import configparser
import sched, time
from datetime import datetime
from keras.models import load_model
from utils import load_image, save_image

def fuse_patch(frame,patch,original,x,y,w,h):
    (p,q,_) = original.shape
    (m,n,_) = frame.shape
    vis = np.zeros(frame.shape,np.uint8)
    vis[:m,:n,:3] = frame
    vis[y:y+h,x:x+w,:3] = patch[:,:,:3]
    vis[:p,:q] = original[:,:,:3]
    return vis

def reconstruct_patch(patch,ae):
    (m,n,_) = patch.shape
    tst = patch.astype('float32')/255.0 - 0.5
    out = np.zeros(patch.shape)
    if tst.shape[0] > 0 and tst.shape[1] > 0:
        tst = cv2.resize(tst, (32, 32))
        out = ae.predict(tst[None])[0]
        out = (out + 0.5)*250.0
        cv2.normalize(out,out,0,255,cv2.NORM_MINMAX)
        out = cv2.resize(out,(n,m))
    return out

def detect_faces(frame,face_model_path,ae,offset,original):
    face_cascade = cv2.CascadeClassifier(face_model_path)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
        x = x-offset
        y = y-offset
        w = w+offset
        h = h+offset
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        recon_patch = reconstruct_patch(frame[y:y+h,x:x+w],ae)
        frame = fuse_patch(frame,recon_patch,original,x,y,w,h)
    return frame

def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def load_keras_model(model_path):
    m = load_model(model_path)
    #print(m.summary())
    return m

def setup_model(model_path, model_names):
    i = int(np.random.randint(low=0,high=len(model_names)))
    model_name = model_names[i]
    encoder_path = model_path+'/'+model_name+'/'+model_name+'.h5'
    ae = load_keras_model(encoder_path)
    original = load_image('test_images/'+model_name+'.jpg')
    original = cv2.resize(original,(50,50))
    return ae,original


def main(config):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("For the time being",cv2.WINDOW_NORMAL)
    WIDTH = int(cam.get(3))
    HEIGHT = int(cam.get(4))

    OFFSET = 30

    MODEL_PATH = 'models'
    MODEL_NAMES = ['faces','cocoon','flowers']
    OUT_IMAGES_PATH = 'output'

    
    ae,original = setup_model(MODEL_PATH,MODEL_NAMES)

    DETECT_FLAG = False

    FRAME_COUNTER_MAX = 100
    frame_counter = 1
    
    while True:
        ret,frame = cam.read()
        frame = cv2.flip(frame,1)

        # Check face detection
        out = frame
        if DETECT_FLAG == True:
            out = detect_faces(frame,"face_detection_model/haarcascade_frontalface_default.xml",ae,OFFSET,original)
    
        if not ret:
            break
        k = cv2.waitKey(1)
        K = k%256

        if K == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif K == 32 or K == 13:
            # SPACE pressed
            save_image(out,OUT_IMAGES_PATH,datetime.now().strftime("%Y%m%d_%H_%M_%S"))
        elif K == 100:
            if DETECT_FLAG == False:
                DETECT_FLAG = True
            else:
                DETECT_FLAG = False

        if frame_counter > FRAME_COUNTER_MAX and DETECT_FLAG == True:
            ae,original = setup_model(MODEL_PATH,MODEL_NAMES)
            frame_counter = 1
        frame_counter += 1
        cv2.imshow("For the time being",cv2.resize(out,(WIDTH,HEIGHT)))
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = read_config('oselfie.config')
    main(config)
    