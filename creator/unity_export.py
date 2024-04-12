
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
from thirdparty.gaussian_splatting.gaussian_to_unity.utils import create_one_file_sections, create_deleted_mask

from thirdparty.gaussian_splatting.utils.system_utils import searchForMaxIteration

warnings.filterwarnings("ignore")

import os
import time

def convert_set(gaussians, args, time_range=None, prev_order = None, max_splat_count=0, last=False):
    print(f"Splat count {gaussians.get_xyz.shape[0]}")
    
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
    duration = args.section_size
    
    for idx in tqdm(range(0, duration, args.save_interval)):
        save_frame(idx, idx/duration, gaussians, order, save_path, args)
        
    splat_count = gaussians.get_xyz.cpu().numpy().shape[0]
    
    max_splat_count = max(max_splat_count, splat_count)
    chunk_count = (max_splat_count+args.chunk_size-1) // args.chunk_size
    
    # args.duration
    if (last):
        create_one_file_sections(save_path, max_splat_count=max_splat_count, max_chunk_count=chunk_count, frame_time=args.save_interval/args.fps, args=args)
    
    time2= time.time()

    print("FPS:", (duration)/(time2-time1))

    return order, splat_count

def remove_outliers(xyz, threshold=3.5):
    median = torch.median(xyz, dim=0).values
    abs_deviation = torch.abs(xyz - median)
    mad = torch.median(abs_deviation, dim=0).values
    mask = abs_deviation / mad < threshold
    xyz_filtered = xyz[mask.all(dim=1)]

    return xyz_filtered

def edit(gaussians):
    xyz = gaussians.get_xyz

    new_xyz = remove_outliers(xyz)
    center = torch.mean(new_xyz, dim=0)

    new_xyz = xyz - center
    distances = torch.norm(new_xyz, dim=-1)
    prune_mask =  (distances > 25.0).squeeze()
    gaussians.prune_points_no_training(prune_mask)

def run_conversion(dataset : ModelParams, iteration: int, 
                   rgbfunction="rgbv1", args=None, prev_order=None, max_splat_count=0, time_range=None, last=False):
    
    with torch.no_grad():
        
        print("use model {}".format(dataset.model))
        GaussianModel = getmodel(dataset.model)
        gaussians = GaussianModel(dataset.sh_degree, rgbfunction)

        gaussians.load_ply(os.path.join(dataset.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(iteration),
                                                           "point_cloud.ply"))
        
        edit(gaussians)
        order, splat_count = convert_set(gaussians, args, prev_order=prev_order, max_splat_count=max_splat_count, time_range=time_range, last=last)
    
    return order, splat_count

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