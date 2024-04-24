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
    sorted_sections = sorted(all_sections, key=lambda x: int(x.split("_")[-1]))
    
    args, model_extract, pp_extract, multiview =gettestparse()
    args.scale = [1,-1,1]
    args.pos_offset = [0,0.7,-120]
    args.rot_offset = [0,0,0]
    args.save_interval = 1
    args.save_name = "ballett.v3d"
    args.audio_path = "D:/spacetime-entrenados/birth/audio.wav"
    args.dynamic_others = True
    args.fps= 30
    args.section_overlap = 0 
    
    outpath = "test"

    prev_order = None
    max_splat_count = 0
    
    assert args.section_size % args.save_interval == 0
    for idx, section in enumerate(sorted_sections):
        
        if idx==0:
            continue
        
        time_range = [idx*args.section_size, (idx+1)*args.section_size]

        print(f" ----- EXPORTING SECTION {idx}-----")
        print(f"Time range: {time_range}")
        
        iteration = searchForMaxIteration(os.path.join(section, "point_cloud"))
        #iteration = 2000
        
        model_extract.model_path = section       
        
        section_outpath = os.path.join(outpath, f"section_{idx}")

        args.unity_export_path = section_outpath
        if not os.path.exists(section_outpath):
            os.makedirs(section_outpath)

        prev_order, splat_count = run_conversion(model_extract, iteration, 
                    rgbfunction=args.rgbfunction, args=args, prev_order=None, 
                    max_splat_count=max_splat_count, time_range=time_range, last = (idx == len(sorted_sections)-1))
        
        if splat_count > max_splat_count:
            max_splat_count = splat_count