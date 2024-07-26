from thirdparty.gaussian_splatting.helper3dg import gettestparse
from thirdparty.gaussian_splatting.utils.system_utils import searchForMaxIteration
import glob
import os
from helper_train import getmodel
import torch


if __name__ == "__main__":
    args, model_extract, pp_extract, multiview =gettestparse()

    model_extract.model = "ours_lite"
    model_extract.loader = "immersiveud"
    model_extract.resolution = 2

    all_sections = glob.glob(args.model_path[:-2] + "_*")
    sorted_sections = sorted(all_sections, key=lambda x: int(x.split("_")[-1]))
    

    for idx, section in enumerate(sorted_sections):
        
        time_range = [idx*args.section_size, (idx+1)*args.section_size]
        
        iteration = searchForMaxIteration(os.path.join(section, "point_cloud"))
        
        model_extract.model_path = section
        
        GaussianModel = getmodel(model_extract.model)
        gaussians = GaussianModel(model_extract.sh_degree, args.rgbfunction)

        gaussians.load_ply(os.path.join(model_extract.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(iteration),
                                                           "point_cloud.ply"))
        
        xyz = gaussians.get_xyz
        center = torch.mean(xyz, dim=0)
        xyz = xyz - center
        distances = torch.norm(xyz, dim=-1)
        prune_mask =  (distances > 250.0).squeeze()
        gaussians.prune_points_no_training(prune_mask)

        gaussians.save_ply(os.path.join(model_extract.model_path,
                                        "point_cloud",
                                        "iteration_" + str(iteration),
                                        "point_cloud_edit.ply"))