# cropmerge
Library for cropping point clouds in blocks and merging them by remapping semantic and instance labels.

Created by [Daniel Lamas Novoa](https://orcid.org/0000-0001-7275-183X), [Mario Soilán Rodríguez](https://orcid.org/0000-0001-6545-2225), and [Belén Riveiro Rodríguez](https://orcid.org/0000-0002-1497-4370) from [GeoTech Group](https://geotech.webs.uvigo.es/en/), [CINTECX](http://cintecx.uvigo.es/gl/), [UVigo](https://www.uvigo.gal/).

## Overview
The purpose of this library is to facilitate the cropping and merging of point clouds. These processes allow the use of segmentation models to segment blocks of point clouds of constant dimensions and/or number of points. The cropping method repeats points between blocks, which are used for the merging process to remap the instance and semantic labels.

### crop
The cropping method divides the point cloud into blocks of a given size. By specifying a minimum overlap, the blocks are placed to maximise the overlap volume with the minimum number of blocks. A constant number of points can be specified. The method returns the indices of the selected points in each block. The method places emphasis on selecting points that repeat between blocks, while maintaining the density distribution of the original point cloud.

### merge
The merge method remaps the instance and semantic labels of each block using the indices calculated by the crop method. It does not apply any geometric operations. The final semantic probabilities are calculated as the average of the probabilities calculated in each block. The instance labels are remaped by merging the instances that share the most number of points between each pair of blocks with an overlapping volume.


## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{lamas_cropmerge_2024,
	address = {Vigo, Spain},
	title = {CROPMERGE: a method for cropping and merging point clouds to facilitate semantic and instance segmentation.},
	url = {https://3dgeoinfoeg-ice.webs.uvigo.es/proceedings},
	abstract = {3D point cloud data is a commonly used data form. Numerous works have been conducted on semantic and instance segmentation of this data format. However, preparing a point cloud to meet the requirements of a segmentation model can be complicated. A typical requirement is to specify the maximum size or a fix number of points. Here, we introduce a methodology for cropping point clouds and merging them after the segmentation process. The method aims to maximise the points that are repeated between crops to use their labels to merge the instances and compute the average semantic probabilities.},
	author = {Lamas, Daniel and Soilán, Mario and Riveiro, Belén},
	booktitle = {31st International Workshop on Intelligent Computing in Engineering},
	month = jul,
	year = {2024},
	pages = {637--646},
}
```

## Licence
cropmerge

Copyright (C) 2023 GeoTECH Group <geotech@uvigo.gal>

Copyright (C) 2023 Daniel Lamas Novoa <daniel.lamas.novoa@uvigo.gal>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program in ![COPYING](https://github.com/GeoTechUVigo/cropmerge/blob/main/COPYING). If not, see <https://www.gnu.org/licenses/>.

## Installation
To install cropmerge (available in pip):
```
python3 -m pip install cropmerge
```
