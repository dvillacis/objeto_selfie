
import argparse
import numpy as np
from PIL import Image
import cv2
from load_dataset import load_image

def parse():
    parser = argparse.ArgumentParser(description='Generating synthetic dataset from input image')
    parser.add_argument('--img_path', help='Path for the original image', required=True)
    parser.add_argument('--dataset_path', help='Path for the generated images', required=True)
    parser.add_argument('--out_img_size', help='Path for the generated images', default=32)
    parser.add_argument('--dataset_size', help='Number of generated images', required=True)
    return parser.parse_args()

def main(args):
    m = args.out_img_size
    orig = load_image(args.img_path)
    orig = Image.fromarray(cv2.resize(orig,(m,m)))
    for k in range(int(args.dataset_size)):
        rotation_angle = np.random.randint(low=0,high=5) # Generate an angle at random
        rot_img = orig.rotate(rotation_angle)
        rot_img.save(args.dataset_path+'/'+str(k)+'.jpg')
        print(k)

if __name__ == "__main__":
    args = parse()
    main(args)