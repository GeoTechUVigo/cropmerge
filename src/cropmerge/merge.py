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


import numpy as np
from ismember import ismember


def merge(indexes, sem_blocks=None, inst_blocks=None, inst_type=np.int64):
    """
    Function to merge the semantic and the instance segmentation information of the blocks of the same point cloud.
    Instance j of block J and instance i of block I are merge if and only if the maximum IoU of instane j with all
    the instances of block I is with the instance i, and vice versa. 
    Semantic probabilities are calculated by averaging.

    :param indexes: numpy array with the index of each point in the point cloud that belong to each block (nº blocks, nº points in each block).   
    :param sem_blocks: numpy array with the semantic segmentation probabilities of the blocks (nº blocks, nº points in each block, nº classes). [Default: None]
    :param inst_subes: numpy array with the instance segmentation labels of the blocks (nº blocks, nº points in each block). [Default: None]
    : param int_type: numpy.dtype for the output inst. [Default: numpy.int64]
    :returns:
        - sem: semantic segmentation probabilities.
        - inst: instance segmentation labels.
    """

    # Initialise arrays
    num_points = np.concatenate(indexes).max()+1
    if not sem_blocks is None:
        sem = np.zeros((num_points, sem_blocks[0].shape[-1]), dtype=sem_blocks[0].dtype)
        count_sem = np.zeros((num_points,1), dtype=inst_type)
    else:
        sem = None

    if not inst_blocks is None:
        inst = np.zeros(num_points, dtype=np.int64)
    else:
        inst = None

    # Number of the last label. 0 is avoid because inst is intialised with 0s
    new_inst_number = 1
    
    # Going through all the blocks
    for i in range(len(indexes)):

        # Semantic labels
        if not sem_blocks is None:
            sem[indexes[i]] = sem[indexes[i]] + sem_blocks[i]
            count_sem[indexes[i]] = count_sem[indexes[i]] + 1
    
        # Instance labels
        if inst_blocks is None: continue
        # Instance labels of block i
        labels_i = np.unique(inst_blocks[i])

        # Array to write the label in the other block with more points in common
        labels_i_merge = np.zeros((len(labels_i)),dtype='object')

        # Compare overlapping between this block an the already analysed blocks
        for j in range(i):

            if not np.any(ismember(indexes[i],indexes[j])[0]):
                continue

            # Going through all the instances in block i and compare it with block j
            for k in range(len(labels_i)):
                # Indexes of this instance in overlap area
                indexes_k = indexes[i][inst_blocks[i] == labels_i[k]]
                # label of this indexes in block j and number of points for each one. The label in inst, since these points have already been analysed
                instance, counts = np.unique(inst[indexes[j][ismember(indexes[j], indexes_k)[0]]], return_counts=True)

                if np.any(instance):
                    # instance in block j with more common points with instance k in block i
                    instance_j = instance[counts == counts.max()][0]

                    # Check if vice versa is the same (compare the instance in block j with block i)
                    indexes_common = np.where(inst == instance_j)[0]
                    instance, counts = np.unique(inst_blocks[i][ismember(indexes[i], indexes_common)[0]], return_counts=True)

                    instance_i = instance[counts == counts.max()][0]
                    # If so, merge instances
                    if instance_i == labels_i[k]:
                        labels_i_merge[k] = np.append(labels_i_merge[k], instance_j)
            
        # Write label
        for j in range(len(labels_i)):
            
            # Indexes of this instance
            this_indexes = indexes[i][inst_blocks[i] == labels_i[j]]

            # If there are indexes to merge
            if np.any(labels_i_merge[j] != 0):
                # remove 0
                labels_i_merge[j] = labels_i_merge[j][1:]
                number = labels_i_merge[j].min()

                inst[ismember(inst,labels_i_merge[j])[0]] = number
            else:
                number = new_inst_number
                new_inst_number +=1

            # Write the label of this indexes
            inst[this_indexes] = number

    # Change indexes numbers from 0 to number of instances
    _, inst = np.unique(inst, return_inverse=True)

    # Normalise semantic probabilities
    sem = sem / count_sem

    return sem, inst