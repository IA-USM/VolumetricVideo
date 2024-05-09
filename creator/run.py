import os

source = "D:/capturas/voley/dataset/colmap_1"
model = "output/volley_new"
config = "configs/custom/volley.json"
total_duration = 299
section_size = 10
sections = total_duration // section_size

for i in range(0, sections-1):
    cmd = f"python train_section.py -s {source} -m {model} --config {config} --section_idx {i}"
    print(cmd)
    os.system(cmd)