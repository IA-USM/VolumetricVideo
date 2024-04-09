import sys
sys.path.append("DeformationPyramid")
sys.path.append("thirdparty/gaussian_splatting")


import torch
import torch.nn as nn
BCE = nn.BCELoss()

import torch.optim as optim


from DeformationPyramid.utils.benchmark_utils import setup_seed

import numpy as np

from DeformationPyramid.model.nets import Deformation_Pyramid
from DeformationPyramid.model.loss import compute_truncated_chamfer_distance

from easydict import EasyDict as edict

from scene.dataset_readers import fetchPly



setup_seed(0)


class PointMatcher:
    def __init__(self, samples = 3000):
        self.config = {
            "gpu_mode": True,

            "iters": 500,
            "lr": 0.01,
            "max_break_count": 15,
            "break_threshold_ratio": 0.001,

            "samples": samples,

            "motion_type": "Sim3",
            "rotation_format": "euler",

            "m": 9,
            "k0": -8,
            "depth": 3,
            "width": 128,
            "act_fn": "relu",

            "w_reg": 0,
            "w_ldmk": 0,
            "w_cd": 0.1
        }

        self.config = edict(self.config)

        if self.config.gpu_mode:
            self.config.device = torch.cuda.current_device()
        else:
            self.config.device = torch.device('cpu')
        
        self.NDP = Deformation_Pyramid(depth=self.config.depth,
                              width=self.config.width,
                              device=self.config.device,
                              k0=self.config.k0,
                              m=self.config.m,
                              nonrigidity_est=self.config.w_reg > 0,
                              rotation_format=self.config.rotation_format,
                              motion=self.config.motion_type)

    
    def match(self, source, target):
        
        assert source.shape[0] > self.config.samples, "Source point cloud must have more points than the number of samples"
        assert target.shape[0] > self.config.samples, "Target point cloud must have more points than the number of samples"

        # Load points
        source_sample_idx = np.random.choice(source.shape[0], self.config.samples, replace=False)
        target_sample_idx = np.random.choice(target.shape[0], self.config.samples, replace=False)

        source_sample = source[source_sample_idx]
        target_sample = target[target_sample_idx]

        # To device
        source, target, source_sample, target_sample = map( lambda x: torch.from_numpy(x).to(self.config.device).float(), [source, target, source_sample, target_sample] )

        # Cancel global translation
        src_mean = source_sample.mean(dim=0, keepdims=True)
        tgt_mean = target_sample.mean(dim=0, keepdims=True)
        source_sample = source_sample - src_mean
        target_sample = target_sample - tgt_mean

        s_sample = source_sample
        t_sample = target_sample

        for level in range(self.NDP.n_hierarchy):

            """freeze non-optimized level"""
            self.NDP.gradient_setup(optimized_level=level)

            optimizer = optim.Adam(self.NDP.pyramid[level].parameters(), lr=self.config.lr)

            break_counter = 0
            loss_prev = 1e+6

            """optimize current level"""
            for iter in range(self.config.iters):


                s_sample_warped, data = self.NDP.warp(s_sample, max_level=level, min_level=level)

                loss = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=1e+9)


                if level > 0 and self.config.w_reg > 0:
                    nonrigidity = data[level][1]
                    target = torch.zeros_like(nonrigidity)
                    reg_loss = BCE(nonrigidity, target)
                    loss = loss + self.config.w_reg * reg_loss


                # early stop
                if loss.item() < 1e-4:
                    break
                if abs(loss_prev - loss.item()) < loss_prev * self.config.break_threshold_ratio:
                    break_counter += 1
                if break_counter >= self.config.max_break_count:
                    break
                loss_prev = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            # Use warped points for next level
            s_sample = s_sample_warped.detach()

        # Warp original point cloud
        self.NDP.gradient_setup(optimized_level=-1)
        source = source - src_mean
        warped_points, data = self.NDP.warp(source)
        warped_points = warped_points.detach().cpu().numpy()

        return warped_points + tgt_mean[0].cpu().numpy()
    

if __name__ == "__main__":
    matcher = PointMatcher()

    source_pcd = fetchPly("/media/pablo/Nuevo_vol/capturas/cocina/output/colmap_1/sparse/0/points3D_total.ply")
    target_pcd = fetchPly("/media/pablo/Nuevo_vol/capturas/cocina/output/colmap_10/sparse/0/points3D_total.ply")

    warped = matcher.match(source_pcd.points, target_pcd.points)

    warped_selection = np.random.choice(warped.shape[0], 500, replace=False)
    warped = warped[warped_selection]
    source_selection = np.random.choice(source_pcd.points.shape[0], 500, replace=False)
    source_pts = source_pcd.points[source_selection]
    target_selection = np.random.choice(target_pcd.points.shape[0], 500, replace=False)
    target_pts = target_pcd.points[target_selection]

    #3d plot
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(warped[:,0], warped[:,1], warped[:,2], c='r', marker='o')
    ax.scatter(target_pts[:,0], target_pts[:,1], target_pts[:,2], c='b', marker='o')
    plt.show()
    
    pass
