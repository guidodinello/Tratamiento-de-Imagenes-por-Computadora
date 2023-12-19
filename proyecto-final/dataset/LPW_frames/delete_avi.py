import subprocess
import os
import glob

folder_path = 'LPW'
for subfolder in glob.glob(os.path.join(folder_path, '*')):
    if not os.path.isdir(subfolder):
        continue
    # delete the video files in each subfolder
    for video in glob.glob(os.path.join(subfolder, '*.avi')):
        print(f"Deleting {video}")
        os.remove(video)
        