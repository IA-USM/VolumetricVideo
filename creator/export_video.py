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

# alternative: delete splats with norm rotation -> 0
def normalize_swizzle_rotation(wxyz):
    normalized = np.divide(wxyz,np.clip(np.linalg.norm(wxyz, axis=1).reshape(-1, 1), 1e-3, 1-1e-3))
    normalized = np.roll(normalized, -1)
    return normalized

def pack_smallest_3_rotation(q):
    abs_q = np.abs(q)
    index = np.argmax(abs_q, axis=1)
    n = q.shape[1]
    rolled_indices = np.zeros((q.shape[0], n), dtype=np.int32)

    rolled_indices[index == 0, :] = [1, 2, 3, 0]
    rolled_indices[index == 1, :] = [0, 2, 3, 1]
    rolled_indices[index == 2, :] = [0, 1, 3, 2]
    rolled_indices[index == 3, :] = [0, 1, 2, 3]

    q_rolled = q[np.arange(q.shape[0])[:, np.newaxis], rolled_indices]
    signs = np.sign(q_rolled[:, 3])
    three = q_rolled[:, :3] * signs[:, np.newaxis]
    three = (three * np.sqrt(2)) * 0.5 + 0.5
    index = index / 3.0

    return np.column_stack((three, index))

def to_numpy_img(tensor):
    grid_sidelen = int(tensor.shape[0] ** 0.5)
    attr_tensor = tensor.reshape((grid_sidelen, grid_sidelen, -1))
    if (type(attr_tensor) == torch.Tensor):
        attr_numpy = attr_tensor.detach().cpu().numpy()
        return attr_numpy
    return attr_tensor
    

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
    q1q2, q3q4 = np.split(rotations, 2, axis=2)
    last_q1q2 = q1q2[:, :, -1]
    last_q3q4 = q3q4[:, :, -1]    
    q1q2_1 = np.concatenate((q1q2, last_q1q2[:, :, np.newaxis]), axis=2)
    q3q4_1 = np.concatenate((q3q4, last_q3q4[:, :, np.newaxis]), axis=2)
    
    # scales
    scales = to_numpy_img(pc._scaling)
    scales = ((scales - min_thresholds["_scaling"]) / (max_thresholds["_scaling"] - min_thresholds["_scaling"])).clip(0,1) * 65535
    scales = scales.astype(np.uint16)

    # opacities
    opacities = to_numpy_img(opacities_final)
    opacities = ((opacities - min_thresholds["_opacity"]) / (max_thresholds["_opacity"] - min_thresholds["_opacity"])).clip(0,1) * 65535
    opacities = opacities.astype(np.uint16)

    

    ones = np.ones((xyz.shape[0], xyz.shape[1], 1), dtype=np.uint16) * 65535
    opacities_3 = np.concatenate((opacities, opacities, opacities), axis=2)
    rot_part1 = q1q2_1
    rot_part2 = q3q4_1

    panel1 = np.concatenate((xyz, colors, opacities_3), axis=1)
    panel2 = np.concatenate((rot_part1, rot_part2, scales), axis=1)
    frame_image = np.concatenate((panel1, panel2), axis=0)

    if frame_image.shape[0] % 4 != 0:
        frame_image = cv2.resize(frame_image, (frame_image.shape[1]//4*4, frame_image.shape[0]//4*4), interpolation=cv2.INTER_NEAREST)
    
    return frame_image


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
                vcodec="libx265", preset="medium", crf=5)
        .run()
    )