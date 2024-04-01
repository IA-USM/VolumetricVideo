
import sys 
sys.path.append("./thirdparty/gaussian_splatting")

import torch
from thirdparty.gaussian_splatting.scene import Scene
from tqdm import tqdm
import warnings

from helper_train import getmodel
from thirdparty.gaussian_splatting.helper3dg import gettestparse
from thirdparty.gaussian_splatting.arguments import ModelParams

from thirdparty.gaussian_splatting.gaussian_to_unity import save_frame
from thirdparty.gaussian_splatting.gaussian_to_unity.converter import get_order
from thirdparty.gaussian_splatting.gaussian_to_unity.utils import create_one_file, create_deleted_mask

from thirdparty.gaussian_splatting.utils.system_utils import searchForMaxIteration

warnings.filterwarnings("ignore")

import os
import time

def convert_set(gaussians, args, time_range=None, prev_order = None):
    
    if args.unity_export_path == "":
        save_path = os.path.join(args.model_path, "unity_format/")
    else:
        save_path = args.unity_export_path
    
    os.makedirs(save_path, exist_ok=True)
    
    if prev_order is None:
        order = get_order(gaussians.get_xyz)
    else:
        order = prev_order

    deleted = os.path.join(save_path, "scene_deleted.bytes")
    if os.path.exists(deleted):
        print("Deleted buffer found, using it to prune points.")
        mask = create_deleted_mask(deleted)
        gaussians.prune_points_order(mask, order)
        order = get_order(gaussians.get_xyz) # Reorder the points after pruning
    
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()
    
    time1 = time.time()

    if time_range==None:    
        step_init = 0
        step_end = args.duration
    else:
        step_init = time_range[0]
        step_end = time_range[1]

    for idx in tqdm(range(step_init, step_end, args.save_interval)):    
        save_frame(idx, idx/(args.duration), gaussians, order, save_path, args)
        
    splat_count = gaussians.get_xyz.cpu().numpy().shape[0]
    chunk_count = (splat_count+args.chunk_size-1) // args.chunk_size
    
    if (step_end == args.duration):
        create_one_file(save_path, splat_count=splat_count, chunk_count=chunk_count, frame_time=args.save_interval/args.fps, args=args)

    time2= time.time()

    print("FPS:", (step_end- step_init)/(time2-time1))

    return order


def run_conversion(dataset : ModelParams, iteration: int, 
                   rgbfunction="rgbv1", args=None, time_range=None, prev_order=None):
    
    with torch.no_grad():
        
        print("use model {}".format(dataset.model))
        GaussianModel = getmodel(dataset.model)
        gaussians = GaussianModel(dataset.sh_degree, rgbfunction)

        gaussians.load_ply(os.path.join(dataset.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(iteration),
                                                           "point_cloud.ply"))
        order = convert_set(gaussians, args, time_range=time_range, prev_order=prev_order)
    
    return order

if __name__ == "__main__":
    args, model_extract, pp_extract, multiview =gettestparse()
    args.scale = [4,-4,4]
    args.pos_offset = [0,0,0]
    args.rot_offset = [0,0,0]
    args.save_interval = 2
    args.save_name = "birth.v3d"
    args.audio_path = "D:/spacetime-entrenados/birth/audio.wav"

    iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))

    run_conversion(model_extract, iteration, rgbfunction=args.rgbfunction, args=args)