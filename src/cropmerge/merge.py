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


def merge(indexes, sem_cubes=None, inst_cubes=None):
    """
    Function to merge the semantic and the instance segmentation information of the cubes of the same point cloud.
    Instance j of cube J and instance i of cube I are merge if and only if the maximum IoU of instane j with all
    the instances of cube I is with the instance i, and vice versa. 
    Semantic probabilities are calculated by averaging.

    :param indexes: numpy array with the index of each point in the point cloud that belong to each cube (nº cubes, nº points in each cube).   
    :param sem_cubes: numpy array with the semantic segmentation probabilities of the cubes (nº cubes, nº points in each cube, nº classes). [Default: None]
    :param inst_subes: numpy array with the instance segmentation labels of the cubes (nº cubes, nº points in each cube). [Default: None]
    :returns:
        - sem: semantic segmentation probabilities.
        - inst: instance segmentation labels.
    """

    # Initialise arrays
    if not sem_cubes is None:
        sem = np.zeros((indexes.max()+1, sem_cubes.shape[2]), dtype=sem_cubes.dtype)
        count_sem = np.zeros((indexes.max()+1,1), dtype=np.int_)
    else:
        sem = None

    if not inst_cubes is None:
        inst = np.zeros(indexes.max()+1, dtype=inst_cubes.dtype)
    else:
        inst = None

    # Number of the last label. 0 is avoid because inst is intialised with 0s
    number = 1
    
    # Going through all the cubes
    for i in range(len(indexes)):

        # Semantic labels
        if not sem_cubes is None:
            sem[indexes[i]] = sem[indexes[i]] + sem_cubes[i]
            count_sem[indexes[i]] = count_sem[indexes[i]] + 1
    
        # Instance labels
        if inst_cubes is None: continue
        # Instance labels of cube i
        labels_i = np.unique(inst_cubes[i])

        # Array to write the label in the other cube with more points in common
        labels_i_merge = np.zeros(len(labels_i),dtype='object')

        # Compare overlapping between this cube an the already analysed cubes
        for j in range(i):

            if not np.any(ismember(indexes[i],indexes[j])[0]):
                continue

            # Going through all the instances in cube i and compare it with cube j
            for k in range(len(labels_i)):
                # Indexes of this instance in overlap area
                indexes_k = indexes[i, inst_cubes[i] == labels_i[k]]
                # label of this indexes in cube j and number of points for each one. The label in inst, since these points have already been analysed
                instance, counts = np.unique(inst[indexes[j, ismember(indexes[j], indexes_k)[0]]], return_counts=True)
                
                if np.any(instance):
                    # instance in cube j with more common points with instance k in cube i
                    instance_j = instance[counts == counts.max()][0]

                    # Check if vice versa is the same (compare the instance in cube j with cube i)
                    indexes_common = np.where(inst == instance_j)[0]
                    instance, counts = np.unique(inst_cubes[i, ismember(indexes[i], indexes_common)[0]], return_counts=True)
                    instance_i = instance[counts == counts.max()][0]
                    # If so, merge instances
                    if instance_i == labels_i[k]:
                        labels_i_merge[k] = np.append(labels_i_merge[k], instance_j)

            
        # Write label
        for j in range(len(labels_i)):
            
            # Indexes of this instance
            this_indexes = indexes[i, inst_cubes[i] == labels_i[j]]
            # Write the label of this indexes
            inst[this_indexes] = number
            # If there are indexes to merge
            if np.any(labels_i_merge[j] != 0):
                # remove 0
                labels_i_merge[j] = labels_i_merge[j][1:]
                inst[ismember(inst,labels_i_merge[j])[0]] = number

            # Update label number
            number += 1

    # Change indexes numbers from 0 to number of instances
    _, inst = np.unique(inst, return_inverse=True)

    # Normalise semantic probabilities
    sem = sem / count_sem

    return sem, inst