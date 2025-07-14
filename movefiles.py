import os
import shutil
import random

sourcefolder = "veo3videos"
destfolder = "train/veo3"

os.makedirs(destfolder, exist_ok = True)

files = os.listdir(sourcefolder)

random.shuffle(files)

filesmove = int(len(files) * 0.8)

for file in files[:filesmove]:
    src_path = os.path.join(sourcefolder,file)
    dest_path = os.path.join(destfolder,file)
    shutil.move(src_path,dest_path)
