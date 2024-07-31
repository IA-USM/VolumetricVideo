import os

source = "D:/capturas/cocina/dataset/colmap_1"
model = "output/cocinamil"
config = "configs/custom/main.json"
total_duration = 297
section_size = 6
sections = total_duration // section_size

for i in range(1, sections-1):
    cmd = f"python train_section_sog.py -s {source} -m {model} --config {config} --section_idx {i}"
    print(cmd)
    os.system(cmd)