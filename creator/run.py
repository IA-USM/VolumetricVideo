import sys
import os

source = "D:/capturas/cocina/output/colmap_1"
model = "output/cocinawarp"
config = "configs/custom/main.json"
total_duration = 299
section_size = 10
sections = total_duration // section_size

for i in range(1, sections-1):
    cmd = f"python train_section.py -s {source} -m {model} --config {config} --section_idx {i}"
    print(cmd)
    os.system(cmd)