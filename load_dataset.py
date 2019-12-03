import pandas as pd
import numpy as np
import cv2
import os
import tarfile
import tqdm
from PIL import Image
from utils import decode_image_from_raw_bytes

def load_image(img_path):
    img = Image.open(img_path)
    img.load()
    img = np.asarray(img, dtype=np.uint8)
    return img

def load_lfw_dataset(raw_images_name, images_name, attrs_name,use_raw=False, dx=80, dy=90, dimx=45, dimy=45):

    # Read attributes
    df_attrs = pd.read_csv(attrs_name, sep='\t', skiprows=1)
    df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns=df_attrs.columns[1:])
    imgs_with_attrs = set(map(tuple,df_attrs[["person", "imagenum"]].values))

    # Read photos
    all_photos=[]
    photo_ids=[]

    with tarfile.open(raw_images_name if use_raw else images_name) as f:
        for m in tqdm.tqdm(f.getmembers()):
            # Only process image files from the compressed data
            if m.isfile() and m.name.endswith(".jpg"):
                img = decode_image_from_raw_bytes(f.extractfile(m).read())

                # Crop faces and resize it
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))

                # Parse person and append it to the collected data
                fname = os.path.split(m.name)[-1]
                fname_splitted = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(fname_splitted[:-1])
                photo_number = int(fname_splitted[-1])
                if (person_id, photo_number) in imgs_with_attrs:
                    all_photos.append(img)
                    photo_ids.append({'person':person_id, 'imagenum': photo_number})
    
    photo_ids = pd.DataFrame(photo_ids)
    all_photos = np.stack(all_photos).astype('uint8')

    # Preserve photo_ids order
    all_attrs = photo_ids.merge(df_attrs, on=('person', 'imagenum')).drop(["person", "imagenum"], axis=1)
    return all_photos, all_attrs

def load_flowers_dataset(images_name):
    # Read photos
    all_photos=[]
    with tarfile.open(images_name) as f:
        for m in tqdm.tqdm(f.getmembers()):
            if m.isfile() and m.name.endswith(".jpg"):
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                #img = img[80:-80, 90:-90]
                img = cv2.resize(img,(32,32))
                all_photos.append(img)
    all_photos = np.stack(all_photos).astype('uint8')
    return all_photos

def load_folder_dataset(dataset_path):
    all_photos=[]
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            img = load_image(dataset_path+'/'+filename)
            all_photos.append(img)
    all_photos = np.stack(all_photos).astype('uint8')
    return all_photos



