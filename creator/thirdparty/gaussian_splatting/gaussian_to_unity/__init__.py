import torch
import time 
from scene.ourslite import GaussianModel
import numpy as np
from helper_train import trbfunction
from gaussian_to_unity.converter import gaussian_timestep_to_unity, gaussian_static_data_to_unity
from gaussian_to_unity.utils import square_centered01, sigmoid, normalize_swizzle_rotation

def sigmoid(v):
    return 1.0 / (1.0 + torch.exp(-v))

def square_centered01(x):
    x -= 0.5
    x *= x * torch.sign(x)
    return x * 2.0 + 0.5


def save_frame(idx, timestamp, pc : GaussianModel, order_indexes, basepath="output", args=None):

    shs = None # Si
    scales_final = pc._scaling # bien

    
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    
    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
    trbfdistanceoffset = timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = trbfunction(trbfdistance)
    

    pointopacity = square_centered01(sigmoid(pc._opacity))
    opacities_final = pointopacity * trbfoutput
    pc.trbfoutput = trbfoutput

    #opacities_final = pc._opacity
    tforpoly = trbfdistanceoffset.detach()
    # Means
    means3D = pc.get_xyz
    means3d_final = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    
    rotations_final = pc.get_rotation(tforpoly)
    #rotations_final =  pc._rotation + tforpoly * pc._omega

    colors_final = pc.get_features(tforpoly)
    
    if idx==0:
        gaussian_static_data_to_unity(pc.get_xyz.shape[0], scales_final, rotations_final, colors_final, 
                                      shs, opacities_final, order_indexes, args= args, basepath=basepath)
    
    gaussian_timestep_to_unity(means3d_final, scales_final, rotations_final, colors_final, opacities_final, order_indexes, debug=False, 
                               args=args, basepath=basepath, idx=idx)