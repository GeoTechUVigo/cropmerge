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


def in_cube(coordinates, centre, cube_size, n_points=np.isnan, seed=np.random.randint(0,100)):
    """
    Function that returns an array with a len equal to self.npoints with the index corresponding to points in
    a cube of self.cube_size long centred in centre. The cube extends in Z dimension.

    :param coordinates: numpy array with the coorindates xyz.
    :param centre: xyz coordinates of the centre of the cube.
    :param n_points: number of points selected. If np.nan it returns all. [Default: np.nan]
    :param seed: seed for random processes. [Default: np.random.randint(0,100)]
    :returns:
        - idx_in_cube: array with npoints indexes corresponding to points in the cube (replace if there are less than n_points).
    """

    np_RandomState = np.random.RandomState(seed)
    
    # Get boundaries of the cube. 
    min_coords = centre-[cube_size/2, cube_size/2, cube_size/2]
    max_coords = centre+[cube_size/2, cube_size/2, cube_size/2]

    # To make sure that it is not leaving data on the Z axis.
    # Modify those values to the limit of the cloud
    min_coords[2] = np.min(coordinates,axis=0)[2]
    max_coords[2] = np.max(coordinates,axis=0)[2]

    # Take points that are inside of the cube.
    idx_in_cube = np.sum((coordinates>=min_coords)*(coordinates<=max_coords),axis=1)==3
    idx_in_cube = np.where(idx_in_cube)[0]

    # From those points, pick self.npoints randomly.
    if np.isnan(n_points): 
        return idx_in_cube

    elif len(coordinates) >= n_points:
        choice = np_RandomState.choice(len(idx_in_cube), n_points, replace=False)
    else:
        choice = np_RandomState.choice(len(idx_in_cube), n_points, replace=True)

    idx_in_cube = idx_in_cube[choice]

    return idx_in_cube


def crop(coordinates, cube_size, n_points=np.nan, overlap=0, return_coords=True, seed=np.random.randint(0,100)):
    '''
    Function to calculated the indexes of the cubes through the coordinates. The point cloud is divied in cubes.
    The cubes has a minimum overlap equal to self.overlap between them. The point cloud is divided first in X and then in Y dimension.
    The first cube and the last are located in the extrems. The inner cubes are positioned so that they all have the same overlapping.
    The points in the overlaped volumes are selected in all the cubes that share these volumes.
    It returns the array unique_index with the indexes in the input coordinates of those points that are in any cube.
    It also return an array number of cubes x number of points with the indexes referred to the unique_index array with the 

    :param coordinates: numpy array with the coordinates xyz.
    :param cube_size: int size of the cube.
    :param n_points: int number of points in each cube. If np.nan it returns the indexes of all the points in each cube. [Default: np.nan]
    :param overlap: double minimum overlap between cubes. [Default: 0]
    :param seed: int seed for random processes. [Default: np.random.randint(0,100)]
    :returns:
        - unique_index: array with the indexes of the point present in any cube referred to the input coordinates array.
        - indexes: array num_cubes x n_points with the indexes of the selected points in the each cube referred to the unique_indexes array.
    '''

    np_RandomState = np.random.RandomState(seed)

    # Calculate the number of cubes and their centre
    # n * c = L + S_u (n-1) -> n = ceil((L - S_u_min) / (c - S_u_min)) ; S_u = (n * C - L) / (n - 1)
    centres = np.array([]).reshape(-1,3)
    
    # Calculate the centre coordinates of each cube.

    # split in X
    max_loc = coordinates[:,0].max()
    min_loc = coordinates[:,0].min()
    l = max_loc - min_loc # length in X
    n = np.ceil((l - overlap) / (cube_size - overlap)) # number of cubes
    overlap = (n * cube_size - l) / (n - 1) if n > 1 else 0 # recalculate overlap
    centres_x = min_loc + cube_size/2 + np.arange(n) * (cube_size - overlap) # X locations of the centres

    # split in Y
    for centre_x in centres_x:
        # Points between this X positions
        this_coordinates = np.logical_and(coordinates[:,0] >= (centre_x - cube_size/2), coordinates[:,0] <= (centre_x + cube_size/2))
        
        # Calculate The number of cubes and their centres.
        max_loc = coordinates[this_coordinates,1].max()
        min_loc = coordinates[this_coordinates,1].min()
        l = max_loc - min_loc
        n = np.ceil((l - overlap) / (cube_size - overlap)).astype('int')
        overlap = (n * cube_size - l) / (n - 1) if n > 1 else 0

        centres_y = np.zeros((n,3))
        centres_y[:,0] = centre_x # X centre is the same for all. Z does not care because the cube is expanded in Z.
        centres_y[:,1] = min_loc + cube_size/2 + np.arange(n) * (cube_size - overlap) # Y centres

        # Append these centres with the others
        centres = np.append(centres, centres_y, axis=0)
        
    #######################################################################################################################3
    # Calcualte the indexes of n_points of each cube.
    # If the number of points is not specified, all points are returned.
    if np.isnan(n_points):
        indexes = np.zeros((len(centres)), dtype='object')
        no_empty= np.zeros((len(centres)), dtype='bool')
        for i in range(len(centres)):
            # indexes in this cube
            
            idx_in_cube = in_cube(coordinates, centres[i], cube_size, n_points=np.nan)

            if ~np.any(idx_in_cube):
                continue
            else:
                no_empty[i] = True
            indexes[i] = idx_in_cube

        indexes = indexes[no_empty]
        unique_index = np.arange(0,len(coordinates))

        return unique_index, indexes

    # If the number of points is specified.

    #indexes of the points of each cube.
    indexes = np.zeros((len(centres), n_points), dtype='int')
    no_empty= np.zeros((len(centres)), dtype='bool')

    # Select the points of each cube by taking the same points in the overlapping areas
    this_cube = np.zeros(len(coordinates), dtype='bool')
    selected_points = np.zeros(len(coordinates), dtype='bool')
    analysed_points = np.zeros(len(coordinates), dtype='bool')
    for i in range(len(centres)):

        # indexes in this cube
        this_cube[:] = False
        idx_in_cube = in_cube(coordinates, centres[i], cube_size, n_points=np.nan)
        # If there are no point continue
        if ~np.any(idx_in_cube):
            continue
        else:
            no_empty[i] = True

        this_cube[idx_in_cube] = True

        # Points in this cube that have been selected in another cube
        overlap_points = np.where(np.logical_and(this_cube, selected_points))[0]

        # Points in the cube that are not in any cube already analysed (selected in another cube or not selected)
        idx_in_cube_no_overlap = np.where(np.logical_and(this_cube, ~analysed_points))[0]

        # Select randomly points from overlap areas and not overlap areas separately.
        # The percentage of points selected in each area is relative to the total distribution of points.
        npoint_no_overlap = np.round(len(idx_in_cube_no_overlap)/len(idx_in_cube) * n_points).astype(int)
        npoint_overlap = n_points - npoint_no_overlap

        # Overlap area
        # If there are not points
        if npoint_overlap<=0:
            idx_in_cube_overlap = np.array([], dtype='int')
        # If there are more points than necesary
        elif len(overlap_points) >= npoint_overlap:
            choice = np_RandomState.choice(len(overlap_points), npoint_overlap, replace=False)
            idx_in_cube_overlap = overlap_points[choice]
        # If it is need to select more points
        else:
            # Points in the cube that are in any cube already analysed (selected in another cube or not selected)
            idx_in_cube_overlap = np.where(np.logical_and(this_cube, analysed_points))[0]
            # Remove points already selected (points in overlap_points)
            idx_in_cube_overlap_remaining = idx_in_cube_overlap[~(ismember(idx_in_cube_overlap, overlap_points)[0])]
            # Random selection of the remaining points
            npoint_overlap_remaining = npoint_overlap - len(overlap_points)
            if len(idx_in_cube_overlap_remaining)>= npoint_overlap_remaining:
                choice = np_RandomState.choice(len(idx_in_cube_overlap_remaining), npoint_overlap_remaining, replace=False)
                idx_in_cube_overlap = np.concatenate((idx_in_cube_overlap_remaining[choice], overlap_points))
            else:
                choice = np_RandomState.choice(len(idx_in_cube_overlap), npoint_overlap, replace=True)
                idx_in_cube_overlap = idx_in_cube_overlap[choice]

        # Non overlap area
        # If there are not points
        if npoint_no_overlap<=0:
            idx_in_cube_no_overlap = np.array([], dtype='int')
        # If there are more points than necesary
        elif len(idx_in_cube_no_overlap) >= npoint_no_overlap:
            choice = np_RandomState.choice(len(idx_in_cube_no_overlap), npoint_no_overlap, replace=False)
            idx_in_cube_no_overlap = idx_in_cube_no_overlap[choice]
        else:
            choice = np_RandomState.choice(len(idx_in_cube_no_overlap), npoint_no_overlap, replace=True)
            idx_in_cube_no_overlap = idx_in_cube_no_overlap[choice]

        # Indexes selected
        idx_cube_selected = np.concatenate((idx_in_cube_no_overlap, idx_in_cube_overlap))

        # Upload selected_points
        selected_points[idx_cube_selected] = True

        # upload analysed points
        analysed_points[idx_in_cube] = True

        # Upload the seleteced points in the variable indexes
        indexes[i,:] = idx_cube_selected

    # Remove empty cubes
    indexes = indexes[no_empty]

    # Calculated the indexes in the input coordinates that are in any cube.
    unique_index, indexes_inv = np.unique(indexes, return_inverse=True)
    # Expressing the indexes of the points in each cube as indexes in the vector unique_index.
    indexes = indexes_inv.reshape(indexes.shape)

    return unique_index, indexes


