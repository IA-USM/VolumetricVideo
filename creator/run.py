import sys
import os

source = "/media/pablo/Nuevo_vol/capturas/cocina/output/colmap_1"
model = "output/huevo"
config = "configs/custom/main.json"
total_duration = 299
section_size = 60
sections = total_duration // section_size

for i in range(0, sections-1):
    cmd = f"python train_section.py -s {source} -m {model} --config {config} --section_idx {i}"
    print(cmd)
    os.system(cmd)