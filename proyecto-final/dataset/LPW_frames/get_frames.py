import os
import subprocess
from pathlib import Path


def extract_frames(video_path, output_folder):
    subprocess.run(['ffmpeg', '-i', video_path, os.path.join(output_folder, 'frame_%d.png')])

folder_path = 'LPW' 
for subfolder in Path(folder_path).glob('*'):
    if not subfolder.is_dir():
        print(f'Skipping {subfolder.name}')
        continue

    # Iterate through the video files in each subfolder
    for video in subfolder.glob('*.avi'):
        
        video_name = os.path.splitext(os.path.basename(video))[0]
        
        # Create new folder for video frames
        output_folder = os.path.join(subfolder, video_name + '_frames')
        os.makedirs(output_folder, exist_ok=True)

        # Extract all frames and save them in the new folder
        extract_frames(video, output_folder)
