from thirdparty.gaussian_splatting.helper3dg import gettestparse
from unity_export import run_conversion
from thirdparty.gaussian_splatting.utils.system_utils import searchForMaxIteration
import glob
import os

if __name__ == "__main__":
    args, model_extract, pp_extract, multiview =gettestparse()

    model_extract.model = "ours_lite"
    model_extract.loader = "immersiveud"
    model_extract.resolution = 2

    all_sections = glob.glob(args.model_path[:-2] + "_*")
    
    args, model_extract, pp_extract, multiview =gettestparse()
    args.scale = [4,-4,4]
    args.pos_offset = [0,0,0]
    args.rot_offset = [0,0,0]
    args.save_interval = 2
    args.save_name = "test.v3d"
    args.audio_path = "D:/spacetime-entrenados/birth/audio.wav"
    
    for idx, section in enumerate(sorted(all_sections)):
        
        time_range = [idx*args.section_size, (idx+1)*args.section_size]

        print(f" ----- EXPORTING SECTION {idx}-----")
        print(f"Time range: {time_range}")
        
        iteration = searchForMaxIteration(os.path.join(section, "point_cloud"))
        
        model_extract.model_path = section
        
        if idx == 0:
            prev_order = None

        prev_order = run_conversion(model_extract, iteration, 
                       rgbfunction=args.rgbfunction, time_range=time_range, args=args, prev_order=prev_order)