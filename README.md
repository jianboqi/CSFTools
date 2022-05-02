# CSFTools
Tools to processing LiDAR point cloud based on CSF.

CSFTools provides a set of Python based tools, including:

 - csfground.py: to filter a point cloud 
 - csfdem.py: a simple gridding and interpolation algorithm to generate a DEM/DSM/CHM
 - csfnormalize.py: normalize point cloud
 - csfclassify.py: use a scalar field to classify the point cloud into 2 classes
 - csflai.py: compute leaf area index (LAI) from airborne discrete-return LiDAR data
 - csfcrown.py: segment tree crowns from CHM

More details, please refer to User Manual.

## Installation

This preprocessing tool requires a few python libraries, to make it easier to install, we recommend to use anaconda (python 3.6+), which has already been integrated with a few scientific computing libraries.
Other libs:

 - laspy: supporting reading and writing of las file. https://github.com/laspy/laspy
	run: 
	
		pip install laspy
		
	or download the source and run:
	
		python setup.py build
		python setup.py install

 - GDAL
    
        conda install gdal
    
 - joblib: supporting parallel computing for python
    
        pip install joblib
 
 - mahotas: computer vision library, supporting watershed transform, etc.
        
        conda config --add channels conda-forge
        conda install mahotas

 - CSF: ground filtering library, go to: https://github.com/jianboqi/CSF, and download all the source code:
		Under the folder python, run:
		
        python setup.py build
        python setup.py install
