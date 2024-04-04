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

    resize.output(os.path.join(outpath,"%d.png")).run()

def preparecolmapfolders(folder, videos_list, offset=0):
    savedir = os.path.join(folder, "colmap_" + str(offset))
    os.makedirs(savedir, exist_ok=True)
    savedir = os.path.join(savedir, "input")
    os.makedirs(savedir, exist_ok=True)
    cameras = [video.replace(".mp4", "") for video in videos_list]

    for cam in cameras:
        imagepath = os.path.join(folder, cam, str(offset) + ".png")
        folder = folder.replace("\\", "/")
        imagesavepath = os.path.join(savedir, Path(cam).stem + ".png")

        shutil.copy(imagepath, imagesavepath)

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
        print("path not exist")
        quit()
    
    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    # 1- Extract frames
    print(f"start extracting {endframe-startframe} frames from videos")
    videoslist = glob.glob(videopath + "*.mp4")
    for v in tqdm.tqdm(videoslist):
        extractframes(v, startframe=startframe, endframe=endframe, w= args.resize_width)

    # 2- Create colmap folders for each frame, add images
    print("start preparing colmap image input")
    for offset in range(startframe, endframe):
        preparecolmapfolders(videopath, videoslist, offset)

    # 3- Run colmap per-frame
    for offset in range(startframe, endframe):
        getcolmapsinglen3d(videopath, offset, colmap_path=args.colmap_path, manual=False)