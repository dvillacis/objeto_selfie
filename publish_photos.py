
import os, uuid
from google.cloud import storage

def upload_photo(bucket_name, source_file_name, destination_photo_name):
    storage_client = storage.Client()
    bucket = storage_client(bucket_name)
    photo = bucket.blob(destination_photo_name)
    photo.upload_from_filename(source_file_name)
    print(f'File {source_file_name} uploaded to {destination_photo_name}')
    return True


BUCKET_NAME = ''
PHOTO_DIR = ''

for filename in os.listdir(PHOTO_DIR):
    if filename.endswith('.png'):
        flag = upload_photo(BUCKET_NAME,filename,uuid.uuid4())
        if flag == True:
            os.remove(filename)