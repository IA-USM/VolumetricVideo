import sys
import os

source = "D:/capturas/ballet/"
model = "output/ballet"
config = "configs/custom/main.json"
total_duration = 60
section_size = 60
sections = total_duration // section_size

for i in range(0, sections):
    cmd = f"python train_section.py -s {source} -m {model} --config {config} --section_idx {i}"
    print(cmd)
    os.system(cmd)