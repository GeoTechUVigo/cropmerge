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


import cropmerge
#from src import cropmerge
import laspy # it is not in requirements.txt
import numpy as np
import time

REPEAT_EXPERIMENT = 10
# Parameters cropmerge
block_size = 20.0 #10 20
n_points =  None #1024 4086 16384 65536 None
overlap = 4.0 # 0.1 0.2
seed = 1

# Load point cloud
las = laspy.read('data/BaileyTruss_000.las')
coordinates = las.xyz # coordinates

# Ground truth
sem_gt = las.classification # semantic labels
inst_gt = las.user_data # instance labels
######################################################################################################################################
time_crop_sum = 0
time_merge_sum = 0
for repeat in range (REPEAT_EXPERIMENT):
    # CROP
    time_crop = time.time()
    unique_index, indexes = cropmerge.crop(coordinates=las.xyz, block_size=block_size, n_points=n_points, overlap=overlap, seed=seed)
    time_crop = time.time() - time_crop
    time_crop_sum += time_crop

    # Downsampling data
    las_ds = las[unique_index]
    sem_gt_ds = sem_gt[unique_index]
    inst_gt_ds = inst_gt[unique_index]

    ######################################################################################################################################
    # INFERENCE (Fake, we use the ground truth)

    # one hot encoder for semantic probabilities
    nb_classes = sem_gt_ds.max()+1
    targets = np.array([sem_gt_ds]).reshape(-1)
    sem_prob_gt_ds = np.eye(nb_classes)[targets]

    # Intitialising arrays for saving the predictions.If n_points/block is fixed, it can be type number, else it must bu dtype object
    sem_prob_blocks = np.zeros((len(indexes)),dtype='object')
    inst_blocks = np.zeros((len(indexes)),dtype='object')
    # predictions byblock using ground truth
    for i in range(len(indexes)):
        sem_prob_blocks[i] = sem_prob_gt_ds[indexes[i]]
        inst_blocks[i] = inst_gt_ds[indexes[i]]

    ########################################################################################################################################
    # MERGE
    time_merge = time.time()
    sem_prob, inst = cropmerge.merge(indexes, sem_blocks=sem_prob_blocks, inst_blocks=inst_blocks)
    time_merge = time.time() - time_merge
    time_merge_sum += time_merge

#####################################################################################################################################
time_crop = time_crop_sum /REPEAT_EXPERIMENT
time_merge = time_merge_sum /REPEAT_EXPERIMENT

# Print times
print('n points raw: ', len(las))
print('n instances raw: ', len(np.unique(las.user_data)))
print('n blocks: ', len(indexes))
print('n points/block: ', n_points)
print('overlap/block_size: ', overlap/block_size)
print('Merge crop: ', time_crop)
print('Merge merge: ', time_merge)

###################################################################################################################################
# Check predicted and merge results with the downsampled data ground truth
# check result semantic
sem = sem_prob.argmax(1)
sem_gt_ds = sem_gt[unique_index]
print('Check semantic labels: ', np.all(sem==sem_gt_ds))
# check result instance
idx_insts = np.unique(inst)
check_inst = np.zeros(idx_insts.shape, dtype='bool')
for i in range(len(check_inst)):
    idx_inst = idx_insts[i]

    pred_inst = inst==idx_inst
    gt_idx = np.unique(inst_gt_ds[pred_inst])

    gt_inst = inst_gt_ds==gt_idx

    check_inst[i] = np.all(pred_inst==gt_inst)
print('Check instance labels: ', np.all(check_inst))

###################################################################################################################################
#save downsampled point cloud with predictions
las_ds.add_extra_dim(laspy.point.format.ExtraBytesParams('classification_prob', 'float32'))

las_ds.classification = sem_prob.argmax(1)
las_ds.classification_prob = sem_prob.max(1)
las_ds.user_data = inst
las_ds.write('out.las')
