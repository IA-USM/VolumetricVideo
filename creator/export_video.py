from thirdparty.gaussian_splatting.helper3dg import gettestparse
from unity_export import UnityExporter
from thirdparty.gaussian_splatting.utils.system_utils import searchForMaxIteration
import glob
import os
import torch
from helper_train import getmodel, trbfunction
import cv2
import numpy as np
import ffmpeg
from tqdm import tqdm

# from print_ranges.py
min_thresholds = {
    "_features_dc": -2,
    "_features_rest": -1,
    "_scaling": -13,
    "_rotation": -1,
    "_opacity": -6,
    "_colors": -7
}

max_thresholds = {
    "_features_dc": 4,
    "_features_rest": 1,    
    "_scaling": 3,
    "_rotation": 2,
    "_opacity": 12,
    "_colors": 7
}

def to_numpy_img(tensor):
    grid_sidelen = int(tensor.shape[0] ** 0.5)
    attr_tensor = tensor.reshape((grid_sidelen, grid_sidelen, -1))
    attr_numpy = attr_tensor.detach().cpu().numpy()
    return attr_numpy

def create_frame_image(pc, timestamp, xyz_min=-1, xyz_max=1):
    ##
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    
    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
    trbfdistanceoffset = timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = trbfunction(trbfdistance)
    
    pointopacity = pc._opacity
    
    tforpoly = trbfdistanceoffset.detach()
    means3D = pc.get_xyz
    means3d_final = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    
    rotations_final = pc.get_rotation_raw(tforpoly)
    colors_final = pc.get_features(tforpoly)
    opacities_final = pointopacity * trbfoutput

    # xyz 
    xyz = to_numpy_img(means3d_final)
    xyz = ((xyz - xyz_min) / (xyz_max - xyz_min))
    xyz = (xyz.clip(0, 1) * 65535).astype(np.uint16)

    # colors
    colors = to_numpy_img(colors_final)
    colors = ((colors - min_thresholds["_colors"]) / (max_thresholds["_colors"] - min_thresholds["_colors"])).clip(0,1) * 65535
    colors = colors.astype(np.uint16)

    # rotations
    rotations = to_numpy_img(rotations_final)
    rotations = ((rotations - min_thresholds["_rotation"]) / (max_thresholds["_rotation"] - min_thresholds["_rotation"])).clip(0,1) * 65535
    rotations = rotations.astype(np.uint16)

    # opacities
    opacities = to_numpy_img(opacities_final)
    opacities = ((opacities - min_thresholds["_opacity"]) / (max_thresholds["_opacity"] - min_thresholds["_opacity"])).clip(0,1) * 65535
    opacities = opacities.astype(np.uint16)

    # scales
    scales = to_numpy_img(pc._scaling)
    scales = ((scales - min_thresholds["_scaling"]) / (max_thresholds["_scaling"] - min_thresholds["_scaling"])).clip(0,1) * 65535
    scales = scales.astype(np.uint16)

    alpha_ones = np.ones((scales.shape[0], scales.shape[1], 1), dtype=np.uint16) * 65535

    # join all
    panel1 = np.concatenate((colors, opacities), axis=2)
    panel2 = rotations
    panel3 = np.concatenate((scales, alpha_ones), axis=2)
    panel4 = np.concatenate((xyz, alpha_ones), axis=2)

    return np.concatenate((panel1, panel2, panel3, panel4), axis=1)


if __name__ == "__main__":
    args, model_extract, pp_extract, multiview =gettestparse()

    model_extract.model = "ours_lite"
    model_extract.loader = "immersiveud"
    model_extract.resolution = 2

    all_sections = glob.glob(args.model_path[:-2] + "_*")
    sorted_sections = sorted(all_sections, key=lambda x: int(x.split("_")[-1]))
    
    args, model_extract, pp_extract, multiview =gettestparse()
    outpath = "compressed"
    os.makedirs(outpath, exist_ok=True)
    #args.audio_path = "D:/spacetime-entrenados/birth/audio.wav"
    args.fps= 30
    args.section_overlap = 0
    rgbfunction="rgbv1"

    xyz_min = 0
    xyz_max = 0

    sorted_sections = sorted_sections[:-1]

    for idx, section in enumerate(sorted_sections):
                
        time_range = [idx*args.section_size, (idx+1)*args.section_size]

        print(f" ----- EXPORTING SECTION {idx}-----")
        print(f"Time range: {time_range}")
        
        iteration = searchForMaxIteration(os.path.join(section, "point_cloud"))
        #iteration = 2000
        
        model_extract.model_path = section       
        

        with torch.no_grad():
            print("use model {}".format(model_extract.model))
            GaussianModel = getmodel(model_extract.model)
            gaussians = GaussianModel(model_extract.sh_degree, rgbfunction)

            gaussians.load_ply(os.path.join(model_extract.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(iteration),
                                                            "point_cloud.ply"))
            if xyz_min == 0 and xyz_max == 0:
                xyz = gaussians.get_xyz.detach().cpu().numpy()
                xyz_min = np.min(xyz)
                xyz_max = np.max(xyz)
                "write min max in a txt file"
                with open(os.path.join(outpath, "min_max.txt"), "w") as f:
                    f.write(f"{xyz_min},{xyz_max}")
            
            t = [frame/args.section_size for frame in range(0, args.section_size)]

            for i, timestamp in tqdm(enumerate(t), total=len(t)):
                frame_image = create_frame_image(gaussians, timestamp, xyz_min, xyz_max)
                cv2.imwrite(os.path.join(outpath, f"frame_{str(idx*args.section_size + i).zfill(5)}.png"), frame_image)
    
    print("Encoding video...")
    # encode results in a video
    # -x265-params lossless=1 for lossless
    (
        ffmpeg
        .input(os.path.join(outpath, "frame_%05d.png"), framerate=args.fps)
        .output(os.path.join(outpath ,"scene.mkv"), pix_fmt="yuv420p10le",
                vcodec="libx265", preset="medium", crf=5, tune="grain")
        .run()
    )