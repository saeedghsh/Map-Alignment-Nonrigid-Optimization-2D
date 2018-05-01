Nonrigid Optimization of Multimodal 2D Map Alignment
====================================================

Details of the method are presented in the following publication.
- Saeed Gholami Shahbandi, Martin Magnusson, Karl Iagnemma. *Nonlinear Optimization of Multimodal Two-Dimensional Map Alignment With Application to Prior Knowledge Transfer*, in IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2040-2047, July 2018. doi: 10.1109/LRA.2018.2806439. [URL](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8292790&isnumber=8302435)

<p align="center">
	<img src="https://github.com/saeedghsh/Map-Alignment-Nonrigid-Optimization-2D/blob/master/docs/opt_motion.gif" width="500">
</p>

Dependencies and Download
-------------------------
Download, install the package and its dependencies.
Most dependencies are listed in ```requirements.txt```, except for [OpenCV](http://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html) which needs to be installed with its python wrapper.
```shell
# Download
git clone https://github.com/saeedghsh/Map-Alignment-Nonrigid-Optimization-2D.git
cd Map-Alignment-Nonrigid-Optimization-2D

# Install dependencies
pip install -r requirements.txt

# Install the package [optional]
python setup.py install
```

NOTE: This repository is dedicated to the implementation of the *nonrigid optimization* of the alignment.
This work relies on the [*2D Map Alignment With Region Decomposition*](https://github.com/saeedghsh/Map-Alignment-2D) for the initial guess of the alignment.
That package and its dependencies have to be installed too.

Usage Example
-------------
Run this:
```shel
python demo.py --img_src 'map_sample/F5_04.png' --img_dst 'map_sample/F5_layout.png' --hyp_sel_metric 'fitness' -visualize -save_to_file -multiprocessing
```
And you should see this:
![example](https://github.com/saeedghsh/Map-Alignment-Nonrigid-Optimization-2D/blob/master/docs/map_sample_F5_04__map_sample_F5_layout.png)


Data Set
--------
A collection layout maps and sensor maps of four different environments are available at [this repository](https://github.com/saeedghsh/Halmstad-Robot-Maps).

<!-- Region Segmentation Transfer -->
<!-- ---------------------------- -->


Laundry List
------------
- [ ] Clean up `optimize_alignment.optimize_alignment.py`.
  For instance, put all the methods for region segmentation and region segmentation transfer into a separate module.
  And complete the documentation of all methods
- [ ] Add more demos, like for region segmentation and knowledge transfer.
- [ ] Include the 3D plots? hmmm, not sure.
  
License
-------
Distributed with a GNU GENERAL PUBLIC LICENSE; see [LICENSE](https://github.com/saeedghsh/Map-Alignment-Nonrigid-Optimization-2D/blob/master/LICENSE).
```
Copyright (C) Saeed Gholami Shahbandi
```
