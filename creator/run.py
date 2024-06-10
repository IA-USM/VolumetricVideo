import os

source = "/home/pablo/single4d/sav/dataset/colmap_0"
model = "output/singleview"
config = "configs/custom/main.json"
total_duration = 12
section_size = 6
sections = total_duration // section_size

for i in range(0, sections-1):
    cmd = f"python train_section.py -s {source} -m {model} --config {config} --section_idx {i}"
    print(cmd)
    os.system(cmd)