#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
from random import randint
import sys 
import json


import shutil
import logging

sys.path.append("./thirdparty/gaussian_splatting")
sys.path.append("./deep-image-matching/src")
sys.path.append("../creator")



from thirdparty.gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


def getparser():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) #we put more parameters in optimization params, just for convenience.
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6029)
    parser.add_argument('--debug_from', type=int, default=-2)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10000, 12000, 25_000, 30_000])
    parser.add_argument("--test_iterations", default=-1, type=int)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--densify", type=int, default=1, help="densify =1, we control points on N3d dataset")
    parser.add_argument("--duration", type=int, default=5, help="5 debug , 50 used")
    parser.add_argument("--basicfunction", type=str, default = "gaussian")
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v2")
    parser.add_argument("--configpath", type=str, default = "None")
    parser.add_argument("--section_idx", type=int, default = 0)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
    else:
        raise ValueError("config file not exist or not provided")

    return args, lp.extract(args), op.extract(args), pp.extract(args)

def getrenderparts(render_pkg):
    return render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]




def gettestparse():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--duration", default=50, type=int)
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v3")
    parser.add_argument("--valloader", type=str, default = "colmap")
    parser.add_argument("--configpath", type=str, default = "1")

    parser.add_argument("--quiet", action="store_true")

    # Unity conversion
    parser.add_argument("--save_interval", default=1, type=int)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    
    parser.add_argument("--pos-format", type=str, default="Norm11")
    parser.add_argument("--scale_format", type=str, default="Norm11")
    parser.add_argument("--sh_format", type=str, default="Norm6")
    parser.add_argument("--col_format", type=str, default="Norm8x4")

    parser.add_argument("--save_name", type=str, default="scene.v3d")
    parser.add_argument("--dynamic_others", action="store_true", default=False)
    parser.add_argument("--dynamic_color", action="store_true", default=True)
    parser.add_argument("--include_shs", action="store_true", default=False)
    
    parser.add_argument("--pos_offset", type=list, default=[0,0,0.35])
    parser.add_argument("--rot_offset", type=list, default=[0,180,0])
    parser.add_argument("--scale", type=list, default=[0.55,0.55,0.55])
    
    parser.add_argument("--audio_path", type=str)

    parser.add_argument("--unity_export_path", type=str, default ="")

    parser.add_argument("--edit_shape", default="cube", type=str) # Sphere, Cube, Cylinder, None
    parser.add_argument("--static_enviroment", action="store_true", default=True)

    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    # configpath
    safe_state(args.quiet)
    
    multiview = True if args.valloader.endswith("mv") else False

    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
        print("args: " + str(args))
        
    return args, model.extract(args), pipeline.extract(args), multiview

def check_database_valid(path):
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(path, "images.bin"))
    assert os.path.exists(os.path.join(path, "cameras.bin"))
    assert os.path.exists(os.path.join(path, "points3D.bin"))
    assert os.path.exists(os.path.join(os.path.dirname(path), "input.db"))

def getcolmapsinglen3d(output_path, offset, colmap_path="colmap", manual=True, startframe=0, feature_matcher = "sift"):
    
    folder = os.path.join(output_path, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted", "sparse")

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    ## Feature extraction

    if feature_matcher == "sift":

        feat_extracton_cmd = colmap_path + f" feature_extractor --database_path {dbfile} --image_path {inputimagefolder} + \
            --ImageReader.camera_model " + "OPENCV" + " \
            --ImageReader.single_camera 1 \
            --SiftExtraction.use_gpu 1"
        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            exit(exit_code)

        # Feature matching
        feat_matching_cmd = colmap_path + " exhaustive_matcher \
            --database_path " + dbfile + "\
            --SiftMatching.use_gpu 1"
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            exit(exit_code)
    
    elif feature_matcher == "superpoint":
        cmd = "python deep-image-matching/main.py --images " + inputimagefolder + " --pipeline superpoint+lightglue" + \
        " --config deep-image-matching/config/superpoint+lightglue.yaml" + " --camera_options deep-image-matching/config/cameras.yaml"
        dbfile_sp = os.path.join(folder, "results_superpoint+lightglue_matching_lowres_quality_high", "database.db")
        distortedmodel_sp = os.path.join(folder, "results_superpoint+lightglue_matching_lowres_quality_high", "reconstruction")

        exit_code = os.system(cmd)
        if exit_code != 0:
            exit(exit_code)

        out_dir = os.path.join(distortedmodel, "0")
        os.makedirs(out_dir, exist_ok=True)

        shutil.copy(dbfile_sp, dbfile)
        for f in os.listdir(distortedmodel_sp):
            shutil.move(os.path.join(distortedmodel_sp, f), out_dir) # distortedmodel
        
        #if not os.path.exists(os.path.join(distortedmodel, "0")):
        #    first_rec = os.listdir(distortedmodel)[0]
        #    shutil.move(os.path.join(distortedmodel, first_rec), os.path.join(distortedmodel, "0"))

    
    if manual:
        # Copy reconstruction from first frame
        source_folder = os.path.join(folder.replace(f"colmap_{offset}", f"colmap_{startframe}"), os.path.join("distorted","sparse","0"))
        if not os.path.exists(manualinputfolder):
            shutil.copytree(source_folder, manualinputfolder)
        
        check_database_valid(manualinputfolder)
        
        print("Starting triangulation")
        os.makedirs(os.path.join(distortedmodel, "0"), exist_ok=True)
        cmd = f"{colmap_path} point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + os.path.join(distortedmodel,"0" ) \
        + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001 --clear_points 1"
    else:
        cmd = colmap_path + " mapper \
        --database_path " + dbfile  + "\
        --image_path "  + inputimagefolder +"\
        --output_path "  + os.path.join(distortedmodel) + "\
        --Mapper.ba_global_function_tolerance=0.000001"
    
    exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"Failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    # sometimes it creates multiple reconstructions, we want to keep the largest one
    # no way this is necessary right?
    reconstructions = os.listdir(distortedmodel)
    if not manual and len(reconstructions) > 1:
        max_size = 0
        max_folder = None
        for r in reconstructions:
            size = sum(os.path.getsize(os.path.join(distortedmodel, r, f)) for f in os.listdir(os.path.join(distortedmodel, r)))
            if size > max_size:
                max_size = size
                max_folder = r
        source_folder = os.path.join(distortedmodel, max_folder)
        shutil.rmtree(os.path.join(distortedmodel, "0"))
        os.rename(source_folder, os.path.join(distortedmodel, "0"))
        
    # Undistort input images
    img_undist_cmd = f"{colmap_path}" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + os.path.join(distortedmodel, "0") + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    
    files = os.listdir(os.path.join(folder, "sparse"))
    os.makedirs(os.path.join(folder, "sparse", "0"), exist_ok=True)
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def getcolmapsingleimundistort(folder, offset, colmap_path="colmap"):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = f"{colmap_path} feature_extractor SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = f"{colmap_path} exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = f"{colmap_path} point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


 

    img_undist_cmd = f"{colmap_path}" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    shutil.rmtree(inputimagefolder)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    #Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
   



def getcolmapsingleimdistort(folder, offset, colmap_path="colmap"):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = f"{colmap_path} feature_extractor SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 
    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = f"{colmap_path} exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = f"{colmap_path}" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
        

def getcolmapsingletechni(folder, offset, colmap_path="colmap"):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = f"{colmap_path} feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = f"{colmap_path} exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = f"{colmap_path} point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = f"{colmap_path}" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  #
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)


    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
    
    return 
    
