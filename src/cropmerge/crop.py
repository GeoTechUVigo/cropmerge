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
import numbers


def in_block(coordinates, min_max=None,centre=None, block_size=None, n_points=None, idx_keep=None, seed=np.random.randint(0,100)):
    """
    Function that returns an array with a len equal to self.npoints with the index corresponding to points in
    a block of self.block_size long centred in centre. The block extends in Z dimension.

    :param coordinates: numpy array with the coorindates.
    :param min_max: minimum and maximum dimnsions of the block. If it is not specified, centre and block_size are used. [Default: None]
    :param centre: coordinates of the centre of the block. [Default: None]
    :param block_size: list with the sizes of the block. If it is a number the 3 dimensions are equal. [Default: None]
    :param n_points: number of points selected. If None it returns all. [Default: None]
    :param idx_keep: list of indexes of coordinates that must be always selected. [Default: None]
    :param seed: seed for random processes. [Default: np.random.randint(0,100)]
    :returns:
        - idx_in_block: array with npoints indexes corresponding to points in the block (replace if there are less than n_points).
    """

    np_RandomState = np.random.RandomState(seed)

    if min_max is not None:
        min_coords = min_max[0]
        max_coords = min_max[1]
    else:
        if centre is None or block_size is None:
            raise ValueError('If mim_max is not specified, centre and block_size must be specified.')
        else:
            if isinstance(block_size, numbers.Number): 
                block_size = [block_size, block_size, block_size]

                # Get boundaries of the block. 
                min_coords = centre-[block_size[0]/2, block_size[1]/2, block_size[2]/2]
                max_coords = centre+[block_size[0]/2, block_size[1]/2, block_size[2]/2]

    # Take points that are inside of the block.
    idx_in_block = np.sum((coordinates>=min_coords)*(coordinates<=max_coords),axis=1)==coordinates.shape[1]
    idx_in_block = np.where(idx_in_block)[0]

    # From those points, pick self.npoints randomly.
    if n_points is None: 
        return idx_in_block

    # keep indexes
    if not idx_keep is None:
        keep_bool = np.isin(idx_in_block, idx_keep)
        keep = idx_in_block[keep_bool]
        idx_in_block = idx_in_block[~keep_bool]
        n_points -= len(keep)

    if len(idx_in_block) == 0:
        return idx_in_block
    
    elif len(idx_in_block) >= n_points:
        choice = np_RandomState.choice(len(idx_in_block), n_points, replace=False)
    else:
        choice = np_RandomState.choice(len(idx_in_block), n_points, replace=True)

    idx_in_block = idx_in_block[choice]

    # keep indexes
    if not idx_keep is None:
        idx_in_block = np.concatenate((idx_in_block, keep))

    return idx_in_block


def crop(coordinates, block_size, n_points=None, overlap=0, repeat_points: bool=True, idx_keep=None, seed=np.random.randint(0,100)):
    '''
    Function to calculated the indexes of the blocks through the coordinates. The point cloud is divied in blocks.
    The blocks has a minimum overlap equal to self.overlap between them. The point cloud is divided first in X and then in Y dimension.
    The first block and the last are located in the extrems. The inner blocks are positioned so that they all have the same overlapping.
    The points in the overlaped volumes are selected in all the blocks that share these volumes.
    It returns the array unique_index with the indexes in the input coordinates of those points that are in any block.
    It also return an array number of blocks x number of points with the indexes referred to the unique_index array with the 

    :param coordinates: numpy array with the coordinates xyz.
    :param block_size: list with the sizes of the block. If it is a number the 3 dimensions are equal.    :param n_points: int number of points in each block. If None it returns the indexes of all the points in each block. [Default: None]
    :param overlap: double minimum overlap between blocks. [Default: 0]
    :param repeat_points: if True, the selected points in the overlapping zones are the same between blocks. [Default: True]
    :param idx_keep: list of indexes of coordinates that must be always selected in their block. Only used if n_points=None or overlap=0 or repeat_points=False. [Default: None]
    :param seed: int seed for random processes. [Default: np.random.randint(0,100)]
    :returns:
        - unique_index: array with the indexes of the point present in any block referred to the input coordinates array.
        - indexes: array num_blocks x n_points with the indexes of the selected points in the each block referred to the unique_indexes array.
    '''

    np_RandomState = np.random.RandomState(seed)

    # Calculate the number of blocks and their centre
    # n * c = L + S_u (n-1) -> n = ceil((L - S_u_min) / (c - S_u_min)) ; S_u = (n * C - L) / (n - 1)
    
    # Calculate the centre coordinates of each block.
    if isinstance(block_size, numbers.Number): block_size = [block_size, block_size, block_size]
    if isinstance(overlap, numbers.Number): overlap = [overlap, overlap, overlap]

    # split in X
    max_loc = coordinates[:,0].max()
    min_loc = coordinates[:,0].min()
    l = max_loc - min_loc # length in X
    n = np.ceil((l - overlap[0]) / (block_size[0] - overlap[0])) # number of blocks
    overlap_rec = (n * block_size[0] - l) / (n - 1) if n > 1 else 0 # recalculate overlap
    #centres_x = min_loc + block_size[0]/2 + np.arange(n) * (block_size[0] - overlap_rec) # X locations of the centres
    mins_x = min_loc + np.arange(n) * (block_size[0] - overlap_rec) # min X locations of blocks
    maxs_x = mins_x + block_size[0] # max X locations of blocks
    mins_x[0], maxs_x[-1] = min_loc, max_loc # avoid round errors

    # split in Y
    #centres_xy = np.array([]).reshape(-1,3)
    mins_xy = np.array([]).reshape(-1,3)
    maxs_xy = np.array([]).reshape(-1,3)
    for min_x,max_x in zip(mins_x, maxs_x):
        # Points between this X positions
        this_coordinates = np.logical_and(coordinates[:,0] >= min_x, coordinates[:,0] <= max_x)
        
        # Calculate The number of blocks and their centres.
        max_loc = coordinates[this_coordinates,1].max()
        min_loc = coordinates[this_coordinates,1].min()
        l = max_loc - min_loc
        n = np.ceil((l - overlap[1]) / (block_size[1] - overlap[1])).astype('int')
        overlap_rec = (n * block_size[1] - l) / (n - 1) if n > 1 else 0

        #centres_y = np.zeros((n,3))
        #centres_y[:,0] = centre_x # X centre is the same for all
        #centres_y[:,1] = min_loc + block_size[1]/2 + np.arange(n) * (block_size[1] - overlap_rec) # Y centres

        min_y, max_y = np.zeros((n,3)), np.zeros((n,3))
        min_y[:,0], max_y[:,0] = min_x, max_x # X is the same for all
        min_y[:,1] = min_loc + np.arange(n) * (block_size[1] - overlap_rec) # min Y locations of blocks
        max_y[:,1] = min_y[:,1] + block_size[1] # max Y locations of blocks
        min_y[0,1],max_y[-1,1] = min_loc, max_loc # avoid round errors

        # Append these centres with the others
        #centres_xy = np.append(centres_xy, centres_y, axis=0)
        mins_xy = np.append(mins_xy, min_y, axis=0)
        maxs_xy = np.append(maxs_xy, max_y, axis=0)

    # split in Z
    #centres = np.array([]).reshape(-1,3)
    mins_xyz = np.array([]).reshape(-1,3)
    maxs_xyz = np.array([]).reshape(-1,3)
    for min_xy, max_xy in zip(mins_xy, maxs_xy):
        # Points between this X and Y positions
        this_coordinates = np.logical_and(np.logical_and(coordinates[:,0] >= min_xy[0], coordinates[:,0] <= max_xy[0]),
                                          np.logical_and(coordinates[:,1] >= min_xy[1], coordinates[:,1] <= max_xy[1]))
        
        # Calculate The number of blocks and their centres.
        max_loc = coordinates[this_coordinates,2].max()
        min_loc = coordinates[this_coordinates,2].min()
        l = max_loc - min_loc
        n = np.ceil((l - overlap[2]) / (block_size[2] - overlap[2])).astype('int')
        overlap_rec = (n * block_size[2] - l) / (n - 1) if n > 1 else 0

        #centres_z = np.zeros((n,3))
        #centres_z[:,0] = centre_xy[0] # X centre is the same for all
        #centres_z[:,1] = centre_xy[1] # Y centre is the same for all
        #centres_z[:,2] = min_loc + block_size[2]/2 + np.arange(n) * (block_size[2] - overlap_rec) # Z centres

        min_z, max_z = np.zeros((n,3)), np.zeros((n,3))
        min_z[:,:], max_z[:,:] = min_xy, max_xy # XY is the same for all
        min_z[:,2] = min_loc + np.arange(n) * (block_size[2] - overlap_rec) # min Y locations of blocks
        max_z[:,2] = min_z[:,2] + block_size[2] # max Y locations of blocks
        min_z[0,2],max_z[-1,2] = min_loc, max_loc # avoid round errors

        # Append these centres with the others
        #centres = np.append(centres, centres_z, axis=0)
        mins_xyz = np.append(mins_xyz, min_z, axis=0)
        maxs_xyz = np.append(maxs_xyz, max_z, axis=0)

    n_blocks = len(mins_xyz)
    mins_maxs_block = np.concatenate((mins_xyz.reshape(-1,1,3), maxs_xyz.reshape(-1,1,3)), axis=1)
    #######################################################################################################################3
    # Calcualte the indexes of n_points of each block.
    if n_points is None or overlap == [0,0,0] or repeat_points == False:
        if n_points is None:
            indexes = np.zeros((n_blocks), dtype='object')
        else:
            indexes = np.zeros((n_blocks, n_points), dtype='int')

        no_empty= np.zeros((n_blocks), dtype='bool')
        for i in range(n_blocks):

            # indexes in this block
            idx_in_block = in_block(coordinates, min_max=mins_maxs_block[i], n_points=n_points, idx_keep=idx_keep, seed=seed)

            if ~np.any(idx_in_block):
                continue
            else:
                no_empty[i] = True
            indexes[i] = idx_in_block

        indexes = indexes[no_empty]

        if n_points is None:
            unique_index = np.arange(0,len(coordinates))

        else:
            # Calculated the indexes in the input coordinates that are in any block.
            unique_index, indexes_inv = np.unique(indexes, return_inverse=True)
            # Expressing the indexes of the points in each block as indexes in the vector unique_index.
            indexes = indexes_inv.reshape(indexes.shape)

        return unique_index, indexes

    # If the number of points is specified and there are overlap and repeat_points=True 
    #indexes of the points of each block.
    indexes = np.zeros((n_blocks, n_points), dtype='int')
    no_empty= np.zeros((n_blocks), dtype='bool')

    # Select the points of each block by taking the same points in the overlapping areas
    this_block = np.zeros(len(coordinates), dtype='bool')
    selected_points = np.zeros(len(coordinates), dtype='bool')
    analysed_points = np.zeros(len(coordinates), dtype='bool')

    for i in range(n_blocks):

        # indexes in this block
        this_block[:] = False
        idx_in_block = in_block(coordinates, min_max=mins_maxs_block[i], n_points=None, seed=seed)
        # If there are no point continue
        if ~np.any(idx_in_block):
            continue
        else:
            no_empty[i] = True

        this_block[idx_in_block] = True

        # Points in this block that have been selected in another block
        overlap_points = np.where(np.logical_and(this_block, selected_points))[0]

        # Points in the block that are not in any block already analysed (selected in another block or not selected)
        idx_in_block_no_overlap = np.where(np.logical_and(this_block, ~analysed_points))[0]

        # Select randomly points from overlap areas and not overlap areas separately.
        # The percentage of points selected in each area is relative to the total distribution of points.
        npoint_no_overlap = np.round(len(idx_in_block_no_overlap)/len(idx_in_block) * n_points).astype(int)
        npoint_overlap = n_points - npoint_no_overlap

        # Overlap area
        # If there are not points
        if npoint_overlap<=0:
            idx_in_block_overlap = np.array([], dtype='int')
        # If there are more points than necesary
        elif len(overlap_points) >= npoint_overlap:
            choice = np_RandomState.choice(len(overlap_points), npoint_overlap, replace=False)
            idx_in_block_overlap = overlap_points[choice]
        # If it is need to select more points
        else:
            # Points in the block that are in any block already analysed (selected in another block or not selected)
            idx_in_block_overlap = np.where(np.logical_and(this_block, analysed_points))[0]
            # Remove points already selected (points in overlap_points)
            idx_in_block_overlap_remaining = idx_in_block_overlap[~(ismember(idx_in_block_overlap, overlap_points)[0])]
            # Random selection of the remaining points
            npoint_overlap_remaining = npoint_overlap - len(overlap_points)
            if len(idx_in_block_overlap_remaining)>= npoint_overlap_remaining:
                choice = np_RandomState.choice(len(idx_in_block_overlap_remaining), npoint_overlap_remaining, replace=False)
                idx_in_block_overlap = np.concatenate((idx_in_block_overlap_remaining[choice], overlap_points))
            else:
                choice = np_RandomState.choice(len(idx_in_block_overlap), npoint_overlap, replace=True)
                idx_in_block_overlap = idx_in_block_overlap[choice]

        # Non overlap area
        # If there are not points
        if npoint_no_overlap<=0:
            idx_in_block_no_overlap = np.array([], dtype='int')
        # If there are more points than necesary
        elif len(idx_in_block_no_overlap) >= npoint_no_overlap:
            choice = np_RandomState.choice(len(idx_in_block_no_overlap), npoint_no_overlap, replace=False)
            idx_in_block_no_overlap = idx_in_block_no_overlap[choice]
        else:
            choice = np_RandomState.choice(len(idx_in_block_no_overlap), npoint_no_overlap, replace=True)
            idx_in_block_no_overlap = idx_in_block_no_overlap[choice]

        # Indexes selected
        idx_block_selected = np.concatenate((idx_in_block_no_overlap, idx_in_block_overlap))

        # Upload selected_points
        selected_points[idx_block_selected] = True

        # upload analysed points
        analysed_points[idx_in_block] = True

        # Upload the seleteced points in the variable indexes
        indexes[i,:] = idx_block_selected

    # Remove empty blocks
    indexes = indexes[no_empty]

    # Calculated the indexes in the input coordinates that are in any block.
    unique_index, indexes_inv = np.unique(indexes, return_inverse=True)
    # Expressing the indexes of the points in each block as indexes in the vector unique_index.
    indexes = indexes_inv.reshape(indexes.shape)

    return unique_index, indexes