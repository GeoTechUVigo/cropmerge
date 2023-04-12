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


from crop import crop
from merge import merge
import numpy as np

# Parameters cropmerge
cube_size = 10.0
n_points = 4096
overlap = 1.0
seed = 1

# Load point cloud with labels
import laspy # it is not in requirements.txt
las = laspy.read('BaileyTruss_000.las')
coordinates = las.xyz
sem_gt = las.classification
inst_gt = las.user_data

# crop
unique_index, indexes = crop(coordinates=coordinates, cube_size=cube_size, n_points=n_points, overlap=overlap, seed=seed)

# Downsampling data
coordinates_ds = coordinates[unique_index]
sem_gt_ds = sem_gt[unique_index]
inst_gt_ds = inst_gt[unique_index]

# Predicting labels (using ground truth)
coordinates_cubes = coordinates_ds[indexes]
sem_cubes = sem_gt_ds[indexes]
inst_cubes = inst_gt_ds[indexes]

# merge
sem, inst = merge(sem_cubes, inst_cubes, indexes)

###################################################################################################################################
# Check predicted and merge results with the downsampled data

# check result semantic
print(np.all(sem==sem_gt_ds))
# check result instance
idx_insts = np.unique(inst)
check_inst = np.zeros(idx_insts.shape, dtype='bool')

for i in range(len(check_inst)):
    idx_inst = idx_insts

    pred_inst = inst==idx_inst
    gt_idx = np.unique(inst_gt_ds[pred_inst])

    gt_inst = inst_gt_ds==gt_idx

    check_inst[i] = np.all(pred_inst==gt_inst)

print(np.all(check_inst))