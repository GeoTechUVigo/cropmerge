'''
Copyright (C) 2023 GeoTECH Group <geotech@uvigo.gal>
Copyright (C) 2023 Daniel Lamas Novoa <daniel.lamas.novoa@uvigo.gal>
This file is part of the program cropmerge.
The program is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or any later version.
The program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
more details.
You should have received a copy of the GNU General Public License along 
with the program in COPYING. If not, see <https://www.gnu.org/licenses/>.
'''


#import cropmerge
from src import cropmerge
import numpy as np
import time

# Parameters cropmerge
block_size = 10.0
n_points =  4096
overlap = 1.0
seed = 1

# Load point cloud with labels
import laspy # it is not in requirements.txt
las = laspy.read('data/BaileyTruss_000.las')
coordinates = las.xyz
sem_gt = las.classification
inst_gt = las.user_data

# one hot encoder
nb_classes = sem_gt.max()+1
targets = np.array([sem_gt]).reshape(-1)
sem_prob_gt = np.eye(nb_classes)[targets]

#coordinates = np.round(coordinates, 4) # rounding is recommended to avoid decimal point problems with points near boundaries
# crop
unique_index, indexes = cropmerge.crop(coordinates=coordinates, block_size=block_size, n_points=n_points, overlap=overlap, seed=seed)

# Downsampling data
coordinates_ds = coordinates[unique_index]
sem_prob_gt_ds = sem_prob_gt[unique_index]
inst_gt_ds = inst_gt[unique_index]

# Predicting labels for each crop (using ground truth)
indexes_obj = np.zeros((len(indexes)),dtype='object')
for i in range(len(indexes)):
    indexes_obj[i] = indexes[i]

sem_prob_blocks_obj = np.zeros((len(indexes)),dtype='object')
inst_blocks_obj = np.zeros((len(indexes)),dtype='object')
for i in range(len(indexes)):
    sem_prob_blocks_obj[i] = sem_prob_gt_ds[indexes_obj[i]]
    inst_blocks_obj[i] = inst_gt_ds[indexes_obj[i]]

sem_prob_blocks = np.zeros((len(indexes), n_points, nb_classes))
inst_blocks = np.zeros((len(indexes), n_points))   
for i in range(len(indexes)):
    sem_prob_blocks[i] = sem_prob_gt_ds[indexes[i]]
    inst_blocks[i] = inst_gt_ds[indexes[i]]


#merge
sem_prob, inst = cropmerge.merge(indexes, sem_blocks=sem_prob_blocks, inst_blocks=inst_blocks)
sem_prob_2, inst_2 = cropmerge.merge(indexes_obj, sem_blocks=sem_prob_blocks_obj, inst_blocks=inst_blocks_obj)

###################################################################################################################################
# Check predicted and merge results with the downsampled data

# check result semantic
sem = sem_prob.argmax(1)
sem_gt_ds = sem_gt[unique_index]
print(np.all(sem==sem_gt_ds))

# check result instance
idx_insts = np.unique(inst)
check_inst = np.zeros(idx_insts.shape, dtype='bool')

for i in range(len(check_inst)):
    idx_inst = idx_insts[i]

    pred_inst = inst==idx_inst
    gt_idx = np.unique(inst_gt_ds[pred_inst])

    gt_inst = inst_gt_ds==gt_idx

    check_inst[i] = np.all(pred_inst==gt_inst)

print(np.all(check_inst))

print(np.all(inst == inst_orig))

#save las
las_out = laspy.create(point_format=las.point_format)
las_out.xyz = coordinates_ds
las_out.classification = sem
las_out.user_data = inst_2
las_out.write('out_2.las')