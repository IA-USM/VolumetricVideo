
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

class UnityExporter:

    def __init__(self):
        self.max_splat_count = 0
        self.environment = None

    def convert_set(self, gaussians, args, last=False):
        print(f"Splat count {gaussians.get_xyz.shape[0]}")
        
        if args.unity_export_path == "":
            save_path = os.path.join(args.model_path, "unity_format/")
        else:
            save_path = args.unity_export_path
        
        os.makedirs(save_path, exist_ok=True)
        
        order = get_order(gaussians.get_xyz)

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
            save_frame(idx, idx/(duration+args.section_overlap), gaussians, order, save_path, args)
            
        splat_count = gaussians.get_xyz.cpu().numpy().shape[0]
        
        self.max_splat_count = max(self.max_splat_count, splat_count)
        chunk_count = (self.max_splat_count+args.chunk_size-1) // args.chunk_size
        
        if (last):
            create_one_file_sections(save_path, max_splat_count=self.max_splat_count, max_chunk_count=chunk_count, frame_time=args.save_interval/args.fps, args=args)
        
        time2= time.time()

        print("FPS:", (duration)/(time2-time1))

        if splat_count > self.max_splat_count:
            self.max_splat_count = splat_count

    def remove_outliers(self, xyz, threshold=3.5):
        median = torch.median(xyz, dim=0).values
        abs_deviation = torch.abs(xyz - median)
        mad = torch.median(abs_deviation, dim=0).values
        mask = abs_deviation / mad < threshold
        xyz_filtered = xyz[mask.all(dim=1)]

        return xyz_filtered

    def edit(self, gaussians, args):
        if args.edit_shape == "cube":
            prune_mask = self.cube(gaussians)
        elif args.edit_shape == "sphere":
            prune_mask = self.sphere(gaussians)
        elif args.edit_shape == "cilinder":
            prune_mask = self.cilinder(gaussians)
        else:
            return

        gaussians.prune_points_no_training(~prune_mask) 

    def cilinder(self, gaussians, center = [2.02,1.13,293.5], radius = 5, y1 = -5, y2 = 5):
        xyz = gaussians.get_xyz

        center = torch.tensor(center).to(xyz.device)
        distances_xz = torch.norm(xyz[:, [0, 2]] - center[[0, 2]], dim=-1)
        prune_mask =  (distances_xz < radius).squeeze()

        prune_mask = prune_mask & (xyz[:,1] > center[1] + y1) & (xyz[:,1] < center[1] + y2)

        return prune_mask
    
    def cube(self, gaussians, x_range = [-18, 25], y_range = [-6, 12]):
        xyz = gaussians.get_xyz
                
        # NOTE: SuperSplat is x-inverted and y-inverted
        prune_mask_y =  (xyz[:,1] > y_range[0]) & (xyz[:,1] < y_range[1])
        prune_mask_x =  (xyz[:,0] > x_range[0]) & (xyz[:,0] < x_range[1])
        
        prune_mask = prune_mask_x & prune_mask_y
        prune_mask = prune_mask.squeeze()

        return prune_mask
        

    def sphere(self, gaussians, center = [2.02,1.13,293.5], radius = 5):

        xyz = gaussians.get_xyz

        # NOTE: SuperSplat is x-inverted and y-inverted
        center = torch.tensor(center).to(xyz.device)        
        distances = torch.norm(xyz - center, dim=-1)
        prune_mask =  (distances < radius).squeeze()
        
        return prune_mask

    def mix_environment(self, gaussians, environment, args):

        center = [2, 0, 19.6]
        radius = 9.65
        y1 = -10
        y2 = 10

        dynamic_mask = ~self.cilinder(gaussians, center = center, radius = radius, y1 = y1, y2 = y2)
        env_mask = self.cilinder(environment, center = center, radius = radius, y1 = y1, y2 = y2 )

        gaussians.blend_points(environment, dynamic_mask, env_mask)

    def run_conversion(self, dataset : ModelParams, iteration: int, rgbfunction="rgbv1", args=None, last=False):
        
        with torch.no_grad():
            
            print("use model {}".format(dataset.model))
            GaussianModel = getmodel(dataset.model)
            gaussians = GaussianModel(dataset.sh_degree, rgbfunction)

            gaussians.load_ply(os.path.join(dataset.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(iteration),
                                                            "point_cloud.ply"))
            
            if self.environment==None and args.static_enviroment:
                self.environment = GaussianModel(dataset.sh_degree, rgbfunction)
                self.environment.load_ply(os.path.join(dataset.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(iteration),
                                                            "point_cloud.ply"))
                motion = self.environment._motion
                self.environment._motion = motion * 0
                
                self.edit(self.environment, args)
            
            self.edit(gaussians, args)

            if args.static_enviroment and self.environment is not None:
                self.mix_environment(gaussians, self.environment, args)

            self.convert_set(gaussians, args, last=last)

           