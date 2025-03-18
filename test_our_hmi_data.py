import os

# Specify the path to your data directory here
input_dir = '/mnt/data/cifs/HMI/0407-Ecoli-PAA'
 
temp = []
def find_dat_files(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dat'):
                temp.append(os.path.join(subdir, file))

find_dat_files(input_dir)

import bacterial_hmi as bhmi

total_segment1 = 0
total_segment2 = 0
total_segment3 = 0
counter = 0
passing = 0

for t in temp:
    try:
        segment1, segment2, segment3 = bhmi.extract_hmi_data(t)
        total_segment1 += segment1
        total_segment2 += segment2
        total_segment3 += segment3
        counter += 1
    except Exception:
        passing += 1

print(f"Max of segment 1: {total_segment1/counter}, segment 2: {total_segment2/counter}, segment 3: {total_segment3/counter}")
print("passing", passing)
