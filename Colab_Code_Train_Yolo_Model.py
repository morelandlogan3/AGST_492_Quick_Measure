#This is the code I used to train the Yolo model. I used the Roboflow dataset that I created and downloaded it in the Yolo format. I then split the dataset into train, valid, and test sets and created a data.yaml file for the Yolo model. Finally, I trained the model using the ultralytics library.
#!pip install ultralytics roboflow # uncomment if you need to install these libraries.

from roboflow import Roboflow
rf = Roboflow(api_key="APIKey") #add api key
project = rf.workspace("logans-workspace").project("agst_492") #worspace name and project name insert
dataset = project.version(1).download("yolov11")

import os, shutil

for split in ['train', 'valid', 'test']:
    split_path = f'{dataset.location}/{split}'
    images_path = f'{split_path}/images'
    labels_path = f'{split_path}/labels'

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for file in os.listdir(split_path):
        if file.endswith('.jpg') or file.endswith('.png'):
            shutil.move(f'{split_path}/{file}', f'{images_path}/{file}')
        elif file.endswith('.txt') and file != '_darknet.labels':
            shutil.move(f'{split_path}/{file}', f'{labels_path}/{file}')

print("Done!")

import os, shutil, random

base     = dataset.location
src_imgs = f'{base}/export/images'
src_lbls = f'{base}/export/labels'

# Create split folders
for split in ['train', 'valid', 'test']:
    os.makedirs(f'{base}/{split}/images', exist_ok=True)
    os.makedirs(f'{base}/{split}/labels', exist_ok=True)

# Get all image files and shuffle
all_images = [f for f in os.listdir(src_imgs) if f.endswith('.jpg')]
random.seed(42)
random.shuffle(all_images)

# 80% train, 10% valid, 10% test
n        = len(all_images)
n_train  = int(n * 0.80)
n_valid  = int(n * 0.10)

splits = {
    'train': all_images[:n_train],
    'valid': all_images[n_train:n_train + n_valid],
    'test':  all_images[n_train + n_valid:]
}

for split, files in splits.items():
    for fname in files:
        stem = os.path.splitext(fname)[0]
        # Move image
        shutil.copy(f'{src_imgs}/{fname}',        f'{base}/{split}/images/{fname}')
        # Move matching label if it exists
        lbl = f'{stem}.txt'
        if os.path.exists(f'{src_lbls}/{lbl}'):
            shutil.copy(f'{src_lbls}/{lbl}',      f'{base}/{split}/labels/{lbl}')
    print(f'{split}: {len(files)} images')

print('Done!')

import yaml

data = {
    'path': base,
    'train': 'train/images',
    'val':   'valid/images',
    'test':  'test/images',
    'nc':    2,
    'names': ['quarter', 'straw']
}
with open(f'{base}/data.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
print("data.yaml created!")

from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
model.train(
    data=f'{base}/data.yaml',
    epochs=50,
    imgsz=640
)
