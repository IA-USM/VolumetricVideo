import sys
import os

source = "D:/datasets/plenoptic/flame_steak/colmap_0"
model = "output/flame"
config = "configs/custom/main.json"
total_duration = 300
section_size = 50
sections = total_duration // section_size

for i in range(0, sections):
    cmd = f"python train_section.py -s {source} -m {model} --config {config} --section_idx {i}"
    print(cmd)
    os.system(cmd)