import nibabel as nib
import os
import json

# data_dir = '/home/jz207/workspace/data/synapse_raw/imagesTr/'
data_dir = '/home/jz207/workspace/data/FLARE22_raw/images/'

file_list = os.listdir(data_dir)
file_list.sort()
affine_list = dict()
for file in file_list:
    nii_img = nib.load(os.path.join(data_dir, file))
    affine = nii_img.affine
    affine_list[file] = affine.tolist()

with open('affine.json', "w") as f:
    json.dump(affine_list, f, indent=4)
