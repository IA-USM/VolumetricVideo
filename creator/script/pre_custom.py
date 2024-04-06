# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os 
import cv2 
import glob 
import tqdm 
import shutil
import sys 
import argparse
sys.path.append(".")
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
from pathlib import Path
import ffmpeg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def extractframes(videopath, startframe=0, endframe=300, w=1080):
    video = ffmpeg.input(videopath)
    resize = video.filter('scale', w, -1)
    outpath = videopath.replace(".mp4", "")
    os.makedirs(outpath, exist_ok=True)

    video.output(os.path.join(outpath,"%d.png")).run()

def preparecolmapfolders(folder, offset=0, extension=".png"):
    savedir = os.path.join(folder, "colmap_" + str(offset))
    os.makedirs(savedir, exist_ok=True)
    savedir = os.path.join(savedir, "input")
    os.makedirs(savedir, exist_ok=True)
    cameras = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    for cam in cameras:
        if "colmap" in cam:
            continue
        imagepath = os.path.join(folder, cam, str(offset) + extension)
        folder = folder.replace("\\", "/")
        imagesavepath = os.path.join(savedir, cam + extension)

        shutil.copy(imagepath, imagesavepath)

def rename_frames(folder, elements):
    for element in elements:
        frames = glob.glob(os.path.join(folder, element, "*.png"))
        frames += glob.glob(os.path.join(folder, element, "*.jpg"))
        frames.sort()

        for i, frame in enumerate(frames):
            newname = os.path.join(folder, element, str(i) + frame[-4:])
            if frame == newname:
                continue
            os.rename(frame, newname)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=1, type=int)
    parser.add_argument("--endframe", default=60, type=int)
    parser.add_argument("--colmap_path", default="colmap", type=str)
    parser.add_argument("--resize_width", default=1080, type=int)

    args = parser.parse_args()
    videopath = args.videopath

    startframe = args.startframe
    endframe = args.endframe


    if startframe >= endframe:
        print("start frame must smaller than end frame")
        quit()
    if not os.path.exists(videopath):
        print("path does not exist")
        quit()
    
    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    # 1- Prepare frames
    videoslist = glob.glob(videopath + "*.mp4")
    extension = ".png"
    if len(videoslist) == 0:
        elements = os.listdir(videopath)
        images_sample = glob.glob(os.path.join(videopath, elements[0], "*.png"))
        images_sample += glob.glob(os.path.join(videopath, elements[0], "*.jpg"))
        if len(images_sample) == 0:
            print("no videos or images found")
            quit()
        print("found images, renaming them to 0.png, 1.png, ...")
        #rename_frames(videopath, elements)
        extension = images_sample[0][-4:]

    else:
        for v in tqdm.tqdm(videoslist):
            print(f"start extracting {endframe-startframe} frames from videos")
            extractframes(v, startframe=startframe, endframe=endframe, w= args.resize_width)
            pass
    
    # 2- Create colmap folders for each frame, add images
    print("start preparing colmap image input")
    for offset in range(startframe, endframe):
        preparecolmapfolders(videopath, offset, extension=extension)
    
    # 3 - Run mapper on the first frame
    getcolmapsinglen3d(videopath, startframe, colmap_path=args.colmap_path, manual=False)

    # 4- Run colmap per-frame, use the poses from first frame for all
    for offset in range(startframe+1, endframe):
        getcolmapsinglen3d(videopath, offset, colmap_path=args.colmap_path, manual=True, startframe=startframe)