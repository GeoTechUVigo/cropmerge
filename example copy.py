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
import laspy # it is not in requirements.txt
import numpy as np
import MODEL # inferene model

#####################################################################
# Parameters cropmerge
block_size = 10.0
n_points =  None #4096
overlap = 1.0
seed = 1

# Load point cloud
las = laspy.read('data/BaileyTruss_000.las')

#####################################################################
# CROP
unique_index, indexes = cropmerge.crop(coordinates=las.xyz, block_size=block_size, n_points=n_points, overlap=overlap, seed=seed)

# Downsampling data
las_ds = las[unique_index]

######################################################################################################################################
# INFERENCE

# Intitialising arrays for saving the predictions.If n_points/block is fixed, it can be type number, else it must bu dtype object
sem_prob_blocks = np.zeros((len(indexes), MODEL.num_classes),dtype='object')
inst_blocks = np.zeros((len(indexes)),dtype='object')
for i in range(len(indexes)):
    sem_prob_blocks[i], inst_blocks[i] = MODEL(las_ds[indexes[i]])

########################################################################################################################################
# MERGE
sem_prob, inst = cropmerge.merge(indexes, sem_blocks=sem_prob_blocks, inst_blocks=inst_blocks)

###################################################################################################################################
#SAVE downsampled point cloud with predictions
las_ds.add_extra_dim(laspy.point.format.ExtraBytesParams('instances', 'uint16'))
las_ds.add_extra_dim(laspy.point.format.ExtraBytesParams('classification_prob', 'float32'))

las_ds.classification = sem_prob.argmax(1)
las_ds.classification_prob = sem_prob.max(1)
las_ds.instances = inst
las_ds.write('out.las')