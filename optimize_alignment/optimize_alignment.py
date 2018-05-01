'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi (saeed.gh.sh@gmail.com)

This file is part of Arrangement Library.
The of Arrangement Library is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>
'''
from __future__ import print_function

import itertools
import time

import cv2
import numpy as np

import scipy.spatial
import skimage.transform

import polyIO
from map_alignment import map_alignment as mapali

'''
All images (np.array 2D), are indexed by [row,col] (i.e. y,x)
All points are presented as (x,y)
'''

################################################################################
################################################################################
################################################################################
def get_gradient(image, kernel_size=3) :
    '''
    '''
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=kernel_size)
    grad_y = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=kernel_size)
    
    ### Combine the two gradients
    # # only the magnitude
    # grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

    # oriented 
    # grad = np.stack( [grad_x, grad_y], axis=2) # dx, dy stacked
    grad = grad_x + 1j*grad_y # with complex value

    return grad

################################################################################
def get_fitness_gradient(image, fitness_sigma=50, grd_k_size=3, normalize=True):
    '''
    provided with an OGM as input image, this method computes the following:
    1) binary image (input image) | counting unexplored as open
    2) distance map (binary image)
    3) fitness map (distance map) | gaussina(distance)
    4) gradient (fitness map)

    it only returns the last two maps
    
    Parameters
    ----------
    
    fitness_sigma (default: 50)
    defines how far the gradient (motion field) should extend from target
    points (occupied) also, the higher the sigma, the "milder" the gradient
    would be should "fitness_sigma" and "grid_res" be correlated? should
    they be equal?
    
    grd_k_size (default: 3)
    the kernel size of sobel filter for gradient image

    normalize (default: True)
    normalizes the outputs to 1
    '''
    [thr1, thr2] = 100, 255 # counting unexplored as open to get occupied points
    
    ########## fitness and motion field construction
    ret, bin_img = cv2.threshold(image, thr1,thr2 , cv2.THRESH_BINARY) # cv2.THRESH_BINARY_INV)
    dis_map = cv2.distanceTransform(bin_img.astype(np.uint8), cv2.DIST_L2,  maskSize=cv2.DIST_MASK_PRECISE)
    fitness_map = gaussian( dis_map, s=fitness_sigma )
    gradient_map = get_gradient(fitness_map, kernel_size=grd_k_size)
    
    if normalize:
        fitness_map /= fitness_map.max()
        gradient_map /= np.absolute(gradient_map).max()
        
    return fitness_map, gradient_map


################################################################################
def gaussian(x, s=1., mu=0):
    ''' a simple normal distribution '''
    return np.exp(- (x-mu)**2 /(2.0 * s**2)) /(s*np.sqrt(2*np.pi))



################################################################################
################################################################################
################################################################################
def get_refined_edge(image,dilate_iterations=1):
    '''
    bitwise and of a dilated countour and occupied cells to assure:
    1) sampled points are on occupied cells
    2) avoide oversampling the clump-like occupied regions
    '''
    occu_thr1, occu_thr2 = 100, 255 # counting unexplored as open to only get occupied pixels
    edge_thr1, edge_thr2 = 200, 255 # counting unexplored as occupied to get one side edge
    edge_apt_size = 3

    _, occu_img = cv2.threshold(image, occu_thr1, occu_thr2 , cv2.THRESH_BINARY_INV)

    _, bin_img = cv2.threshold(image, edge_thr1,edge_thr2 , cv2.THRESH_BINARY) # cv2.THRESH_BINARY_INV)
    edge_img = cv2.Canny(bin_img, edge_thr1,edge_thr2, edge_apt_size) # thr1, thr2, apt_size
    edge_img = cv2.dilate(edge_img, np.ones((3,3),np.uint8), iterations = dilate_iterations)

    refined_edge = cv2.bitwise_and(occu_img, edge_img)

    return refined_edge

################################################################################
def convert_map_to_points(image,
                          point_source=['edge','occupancy'][0],
                          edge_refine_dilate_itr=2):
    '''
    input is a bitmap image, representing an occupancy map
    (occupied=0, unexplored=127, open=255)

    This method returns a 2d array of size Mx2
    each row in the array is the (x,y - col,row) coordinate of the pixels that 
    belong to either i) the edges of the open space, or ii) the occupied cells
    '''
    
    if point_source=='occupancy':
        pts = np.roll( np.array(np.nonzero(image==np.min(image))).T, 1, axis=1 )
    elif point_source=='edge':
        refined_edge = get_refined_edge(image, dilate_iterations=edge_refine_dilate_itr)
        pts = np.roll( np.array(np.nonzero(refined_edge==np.max(refined_edge))).T, 1, axis=1 )
    return pts

################################################################################
def get_contour_path(image,
                     contour_dilate_ksz=3,
                     contour_dilate_itr=20):
    ''''''

    ### convert to binary (unexplored == occupied)
    ret, bin_img = cv2.threshold(image, 200, 255 , cv2.THRESH_BINARY)

    ### dilating to expand the contour
    dilate_k = np.ones((contour_dilate_ksz,contour_dilate_ksz),np.uint8)
    bin_img = cv2.dilate(bin_img, dilate_k, iterations=contour_dilate_itr)

    ### find contours
    im2, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_pts = [c[:,0,:] for c in contours]

    ### sort contours according to thier sizes
    areas = [cv2.contourArea(c) for c in contours_pts] # [c.shape[0] for c in contours_pts]
    contours_pts = [c for (a,c) in sorted(zip(areas,contours_pts),reverse=True)]
    
    ### selecting the biggest contour
    # max_idx = [c.shape[0] for c in contours].index(max([c.shape[0] for c in contours]))
    # contour_pts = contours[max_idx]

    # creating one path per contour
    contours_path = [mapali._create_mpath(c) for c in contours_pts]

    return contours_pts, contours_path

################################################################################
def get_corner_sample(image,
                      edge_refine_dilate_itr= 5,
                      maxCorners=500, qualityLevel=0.01, minDistance=25):
    '''
    for the description of the following parameters see cv2.goodFeaturesToTrack 
    maxCorners, qualityLevel, minDistance

    Note on the qualityLevel
    ------------------------
    for my sensor maps, and with minDistance=25
    qualityLevel below 0.1 returns almost all corners, and the only criteria is 
    left to minDistance. In this configuration, the minDistance acts as the grid
    resolution from get_point_sample() method
    '''

    # with dilate_iterations=1 some border lines are missing.
    # hoping that cv2.goodFeaturesToTrack would take care of spacing I set
    # dilate_iterations=3   
    refined_edge = get_refined_edge(image, dilate_iterations=edge_refine_dilate_itr)
    corner_points = cv2.goodFeaturesToTrack(refined_edge, maxCorners, qualityLevel, minDistance)
    corner_points = corner_points.astype(np.int)[:,0,:]
    
    return corner_points

################################################################################
def uniform_sampling_with_grid(data_points, grid_points, grid_res=None):
    '''
    obsolete
    used in get_point_sample() which is obsolete thanks to get_corner_sample()

    How it works
    ------------
    idx is index to data_points
    dist.argmin(axis=0) gives the index to the closest data_point to each grid_points
    thus dist.argmin(axis=0) has a len equal to grid_points.shape[0] and dist.shape[1]
    np.nonzero(dist.min(axis=0) < grid_res)[0] is the index to grid_points whos
    distance to its closest data_point is less than a threshold.
    Finally "idx" is the index to the closest data_point of those grid_points whose
    closest data_point is less than a threshold
    
    Note
    ----
    the "uniform-sampling" is not that uniform! needs pruning :D
    '''
    if grid_res is None: grid_res = grid_points[1,0] - grid_points[0,0]
    dist = scipy.spatial.distance.cdist(data_points, grid_points)
    idx = dist.argmin(axis=0)[ np.nonzero(dist.min(axis=0) < grid_res)[0] ]
    return idx

################################################################################
def get_point_sample(image, data_points=None, grid_res=50, point_type=['edge','occupancy'][0]):
    '''
    OBSOLETE (use get_corner_sample instead)

    provided with an OGM as input image, this method returns an array of points'
    coordinates, corresponding to edges/occupied pixels of the input image.

    furthermore, it superimposes a sparse grid on top of the image, and performs
    some sort of uniform sampling of aforementioned edge/occupied points. This 
    sampling is provided in form of indices to aforementioned points
    '''

    ########## extract the coordinate of occupied/or/edge points 
    if data_points is None and point_type in ['edge','occupancy']:
        data_points = convert_map_to_points(image, point_source=point_type, edge_refine_dilate_itr=2)
    else:
        raise(StandardError(' "data points" are not provided and "point type" is unknown '))

        # if point_type == 'edge':
        #     [thr1, thr2] = 200, 255 # counting unexplored as occupied to get one side edge
        #     apt_size = 3
        #     ret, bin_img = cv2.threshold(image, thr1,thr2 , cv2.THRESH_BINARY) # cv2.THRESH_BINARY_INV)
        #     edge_img = cv2.Canny(bin_img, thr1, thr2, apt_size)
        #     data_points = np.roll( np.array(np.nonzero(edge_img==np.max(edge_img))).T, 1, axis=1 )
        # elif point_type == 'occupancy':
        #     [thr1, thr2] = 100, 255 # counting unexplored as open to only get occupied pixels
        #     ret, bin_img = cv2.threshold(image, thr1,thr2 , cv2.THRESH_BINARY_INV)
        #     data_points = np.roll( np.array(np.nonzero(bin_img==np.max(bin_img))).T, 1, axis=1 )

    ########## grid construction
    # (image.shape[1] % grid_res) // 2 -> add a margin at the begining to center the grid over image
    x = np.arange((image.shape[1] % grid_res) // 2, image.shape[1], grid_res)
    y = np.arange((image.shape[0] % grid_res) // 2, image.shape[0], grid_res)
    xv, yv = np.meshgrid(x, y)
    
    # these are pixel coordinates, unlikely to go above 65535! and definitly not negative!
    grid_points = np.stack( ( xv.reshape(xv.shape[0]*xv.shape[1]),
                              yv.reshape(yv.shape[0]*yv.shape[1]) ),
                            axis=1).astype(np.uint16).astype(np.int16)
    
    ########## sampling
    data_points = data_points.astype(np.int16)
    sampling_idx = uniform_sampling_with_grid(data_points, grid_points)#, grid_res=grid_res)

    return data_points[sampling_idx,:]


################################################################################
def estimate_fitness(X, fitness_map):
    '''
    This method returns the fitness value of a set of points from a fitness map

    Input
    -----
    fitness_map: np.array 2D, essentially a gray-scale image
    X: np.array 2D, a list of 2D points 

    Output
    ------
    fitness: np.array 1D
    the fitness value of points in X according to fitness_map
    fitness of points out of fitness_map boundary is zero
    '''
    # masking points that are out of the image boundary
    valid_mask = np.logical_and( np.logical_and(0<X[:,0], X[:,0]<fitness_map.shape[1]), # x in bound
                                 np.logical_and(0<X[:,1], X[:,1]<fitness_map.shape[1]) )# y in bound
    valid_idx = np.nonzero(valid_mask)[0]
    
    fitness = np.zeros(X.shape[0])
    fitness[valid_idx] = fitness_map[X[valid_idx,1].astype(int), X[valid_idx,0].astype(int)]
    return fitness

################################################################################
def estimate_motion(X, gradient_map, X_corr_stacked):
    '''
    given a set of coordinates "X", and a gradient map
    '''
    # # the gist of this method is the following two lines, howver, the points
    # # out of image boundary are turning to other side and take ivalide values.
    # # we have to assign zero to motion of these points
    # dX = gradient_map[X.astype(np.int)[:,1], X.astype(np.int)[:,0]]
    # dX = np.stack((dX.real, dX.imag), axis=1)

    # masking points that are out of the image boundary
    valid_mask = np.logical_and( np.logical_and(0<X[:,0], X[:,0]<gradient_map.shape[1]), # x in bound
                                 np.logical_and(0<X[:,1], X[:,1]<gradient_map.shape[0]) )# y in bound
    valid_idx = np.nonzero(valid_mask)[0]
    X_valid = X[valid_idx]

    # estimating the motion of idividual points
    dX = np.array(np.zeros(X.shape[0]), dtype=complex)
    dX[valid_idx] = gradient_map[X_valid.astype(np.int)[:,1], X_valid.astype(np.int)[:,0]]
    dX = np.stack((dX.real, dX.imag), axis=1)
    
    ########## weighted average
    # apply local coherency constratint average motions
    dX_stacked = np.stack([dX for _ in range(dX.shape[0])], axis=1)
    dX_constrained = (dX_stacked * X_corr_stacked).mean(axis=0) # hadamard-schur product
    
    return dX_constrained, dX

################################################################################
def data_point_correlation(X, correlation_sigma=400, normalize=True):
    '''
    Given a set of point in Euclidean space, the method calculates a matrix of
    pairswise distances between all points, and applying a Gaussian to the
    distance matrix, it returns a matrix that give the correlation of points
    according to their distance

    It is used for applying a local constraint over the motion of point that are
    obtained from a motion field (gradient image).

    Paramters
    ---------

    correlation_sigma (default: 400)
    defines the scope of locallity

    normalize (default: True)
    if True, the correlation matrix is normalized to (0,1), so the correlation 
    of a point with itself is 1. This is helpful so that it can be directly used
    as the weight in averaging point motions.

    on why of the normalization of X_correlation:
    yes, because we use gauss instead of the expression from the paper
    '''

    X_pdist = scipy.spatial.distance.pdist( X, metric='euclidean' )
    X_pdist = scipy.spatial.distance.squareform( X_pdist )
    X_correlation = gaussian( X_pdist, s=correlation_sigma )

    if normalize:
        X_correlation /= X_correlation.max() # normalizing correlation to (0,1)

    return X_correlation



################################################################################
def double_fitness(src_img, dst_img,
                   tform_align, tform_opt,
                   fitness_sigma=5,
                   src_contour_dilate_itr=20):
    '''
    src image is the sensor map
    dst image is the layout map

    src_fit is the fitness of src_points to dst_fit_map
    des_fit is the fitness of dst_points to src_fit_map

    

    moving src contour to dst:
    This doesn't work becasue tform_opt can't correctly transform all contour points

    moving dst contour to src    
    This has the same problem as the other way, but those destination points
    out of the convext of tform_opt tesselation's convex hull will map to (0,0)
    but that is ok, because we don't care about them any way. However we still 
    need the contour of src to filter out the other dest points that are inside 
    the convexhull of the tesselation and yet irrelavant because they are out of
    src contour
    '''
    ### Source map
    src_fit_map, src_grd_map = get_fitness_gradient(src_img, fitness_sigma=fitness_sigma, grd_k_size=3, normalize=True)
    X_src_original = convert_map_to_points(src_img, point_source='edge', edge_refine_dilate_itr=2)
    X_src_in_dst_frame = tform_opt(tform_align(X_src_original))

    ### Destination map
    dst_fit_map, dst_grd_map = get_fitness_gradient(dst_img, fitness_sigma=fitness_sigma, grd_k_size=3, normalize=True)
    X_dst_original = convert_map_to_points(dst_img, point_source='occupancy')
    X_dst_in_src_frame = tform_align.inverse(tform_opt.inverse(X_dst_original))


    '''
    there is the problem of transforming X_dst_original to src_frame. Those
    points out of the convex-hulll of `tform_opt._tesselation` are not 
    transformed predictibaly, so it's safest to exlude them all.
    But the problem is unless I transform them, I can't check their containment
    in the contours of source image...
    So, don't read much into this!
    [un]fortunately the success detection (see note below) didn't work, so I 
    won't be using the [mis]fitness of X_dst anyway. I will be using fitness of
    X_src for map fusion

    this was for back when I wanted to use this for a "global fitness" towards
    success detection. For that, I wanted to calculate the average of "misfit"
    of dst_point only if they are inside the contour of src_map to account for
    partiality of the src_map. which didn't work...
    '''
    ### creating a path from the contour of src
    contours_pts, contours_path = get_contour_path(src_img, contour_dilate_ksz=3, contour_dilate_itr=20)

    ### subjecting dst point to valid region (contour) of transformed src
    contained = np.stack([path.contains_points(X_dst_in_src_frame) for path in contours_path], axis=1)
    contained = np.any(contained, axis=1)
    contained_idx = np.nonzero( contained )[0]
    not_contained_idx = np.nonzero( np.logical_not( contained ) )[0]
    # those points of dst that are out the contours of src_img, are transformed
    # to a point  (e.g. [-1,-1] ) which is out of the boundary of "src_fit_map"
    X_dst_in_src_frame[not_contained_idx,:] = [-1,-1] 

    #################### computing the fitness of src and dst points
    src_fit = estimate_fitness(X=X_src_in_dst_frame, fitness_map=dst_fit_map)
    dst_fit = estimate_fitness(X=X_dst_in_src_frame, fitness_map=src_fit_map)

    X_src = {'original':X_src_original, 'in_dst_frame':X_src_in_dst_frame}
    X_dst = {'original':X_dst_original, 'in_src_frame':X_dst_in_src_frame, 'in_src_contours_idx':contained_idx}

    return src_fit, dst_fit, X_src, X_dst

################################################################################
def motion_decoherency(X_aligned, X_optimized, X_correlation):
    '''
    OBSOLETE
    This does not consider the underlying transformation! It only works if the 
    underlying tform is a Translation. For instance, if the underlying tform is
    rotation, all the motion might be coherent, yet the Euclidean distance of
    motions is non-zero
    '''
    dX_pdist = scipy.spatial.distance.pdist( (X_aligned - X_optimized), metric='euclidean' )
    dX_pdist = scipy.spatial.distance.squareform( dX_pdist )
    return (dX_pdist * X_correlation).std() # hadamard-schur product


################################################################################
def get_tform_per_point(tform):
    '''
    This method takes a `PiecewiseAffineTransform` object and returns a list of 
    `AffineTransform` objects. The output list is a subset of `tform.affines`.
    While `tform.affines` is a list of `AffineTransform` objects per each
    simplex in `tform._tesselation`, the output list is `AffineTransform`
    objects per each point in `tform._tesselation.points`.

    This is an approximation, since each point in `tform._tesselation.points`
    belongs to multiple simplices and therefor could be transformed by either
    one of them. the one `AffineTransform` assigned to a point is the one among
    those, which has paramters (sx,sy,tx,ty,rot) closest to all others

    input
    -----
    tform: skimage.transform._geometric.PiecewiseAffineTransform object

    output
    ------
    tforms: list of skimage.transform._geometric.AffineTransform objects

    Note
    ----
    tfrom is supposed to be the transformation yielded from optimization, hence
    np.allclose(tform._tesselation.points, X_optimized) == True
    '''
    M = tform._tesselation.points.shape[0]

    params = np.stack([ np.array( [aff.translation[0], aff.translation[1],
                                   aff.scale[0], aff.scale[1],
                                   aff.rotation] )
                        for aff in tform.affines ], axis=0)

    tf_per_point = []
    for p_idx in range(M):
        # smpl_idx is the index of the simplices that point[p_idx,:] belongs to
        smpl_idx, _ = np.nonzero(tform._tesselation.simplices == p_idx)

        # for each simplex, there is an affine transform: [tform_opt.affines[s_idx] for s_idx in smpl_idx]
        # since averaging tforms is stupid, instead, we pick the tform that is closest to all others in the same list
        # closeness is the Eculidean distance of tform matrices over their paramters (sx,sy,tx,ty,rot)
        dist_mat = scipy.spatial.distance.pdist(params[smpl_idx], 'euclidean')
        dist_mat = scipy.spatial.distance.squareform(dist_mat)
        tf_per_point.append( np.array(tform.affines)[smpl_idx][ np.argmin(dist_mat.sum(axis=0)) ] )

    return tf_per_point


################################################################################
def get_motion_decoherency(tform, X_aligned, X_optimized, X_correlation):
    '''
    tform is a piece-wsie affine transform
    '''
    tf_per_point = get_tform_per_point(tform)

    M = X_aligned.shape[0] # M is the number of points underlying the tesselation, ie X.shape[0]
    actual_motion = X_optimized - X_aligned
    actual_motion = np.stack( [actual_motion for _ in range(M)], axis=0)

    motion_with_others_model = np.array([ tf_per_point[idx]._apply_mat(X_aligned, tf_per_point[idx]._inv_matrix) - X_aligned
                                          for idx in range(M) ])

    motion_discrepency = actual_motion - motion_with_others_model
    motion_discrepency = np.sqrt((motion_discrepency**2).sum(axis=2))
    # assert np.allclose( np.trace(motion_discrepency), 0)

    motion_decoherency = motion_discrepency * X_correlation # hadamard-schur product
    return np.median(motion_decoherency)
    return motion_decoherency.std()
    return motion_decoherency.mean(axis=1)
    return motion_decoherency.mean()




################################################################################
def get_correlation_sigma(itreration,
                          max_iteration=10000,
                          x_step = 1000,
                          rule=['magnify', 'alleviate'][0],
                          mapping=['linear'][0]
                      ):
    '''
    x is the normalized value of iteration
    y is the sigma value corresponding to x, according to the specified mapping

    extent of sigma (y)
    -------------------
    y_min <- size of locallity scope (defailt: 400)
    y_max <- max(image.shape)

    iteration to x conversion
    -------------------------
    a linear map from (0, max_iteration) to (x_min, x_max) or (x_max, x_min)
    x_min = 1
    x_max = max_iteration//x_step

    Linear mapping
    --------------
    y = a*x + b
    y_min = a* x_min + b
    y_max = a* x_max + b
    
    a = np.float(y_max - y_min)/ (x_max-x_min)
    b = y_min - a*x_min
    y = a*x + b
    '''

    x_min, x_max = 1, max_iteration//x_step
    y_min, y_max = 400., 1500.


    # normalize the value of iteration to (x_min, x_max)
    if rule=='magnify':
        x = itreration//x_step + 1
    elif rule=='alleviate':
        x = max_iteration//x_step - itreration//x_step

    if mapping == 'linear':
        a = np.float(y_max - y_min)/ (x_max-x_min)
        b = y_min - a*x_min
        y = a*x + b
    elif mapping == 'xxx':
        pass

    return y # -> correlation sigma


################################################################################
def optimize_alignment(X0, X_correlation, gradient_map, fitness_map, config, verbose=True):
    '''
    Note on opt_rate
    ----------------
    with grd_img normalized to 1, optimization rate should be 1 (=10**0)
    with normalized grd_map and opt_rate=10**3, the points will slide out

    TODO [termination critera]
    ----
    1_ Should I threshold the growth instead of the value of fitness/max_motion
    2_ max_motion is not much helpful for success detection, is it a relevant optimization criterion?
    '''

    sigma_update_steps = [0, 1000][0] # if 0, won't update
    
    ########################################
    ###################### optimization loop
    ########################################
    X_corr_stacked = np.stack([X_correlation,X_correlation], axis=2)

    itr = 0
    X = X0.copy()


    save_for_animate = False
    if save_for_animate: X_history = np.atleast_3d( X0.copy() )

    tic = time.time()
    while True:

        # update sigma and X_correlation
        if sigma_update_steps and (itr%sigma_update_steps ==0):
            correlation_sigma = get_correlation_sigma(itreration=itr,
                                                      max_iteration=config['max_itr'],
                                                      x_step=sigma_update_steps,
                                                      rule=['magnify', 'alleviate'][0],
                                                      mapping=['linear'][0])
            X_correlation = data_point_correlation(X=X0, correlation_sigma=correlation_sigma, normalize=True)
            X_corr_stacked = np.stack([X_correlation,X_correlation], axis=2)

        # estimate motion
        dX_constrained, dX = estimate_motion(X, gradient_map, X_corr_stacked)
        
        # estimate motion
        motion = dX_constrained * config['opt_rate']
        # motion = dX * config['opt_rate']

        max_motion = np.sqrt(motion[:,0]**2 + motion[:,1]**2).max()
        
        # update point locations
        X += motion

        if save_for_animate: X_history = np.append(X_history, np.atleast_3d(X), axis=2)

        # estimate fitness
        fitness = estimate_fitness(X, fitness_map).mean()
        
        itr += 1
        maxed = itr >= config['max_itr']
        no_motion = max_motion < config['tol_mot']
        well_fit = fitness > config['tol_fit']
        if (maxed) or (no_motion) or (well_fit): break

    if save_for_animate: np.save('X_history.npy', X_history)
        
    ########################################
    ######### reporting optimization results
    ########################################
    log = {}
    if maxed:
        log['break_cause'] = 'maximum iteration reached'
    elif no_motion:
        log['break_cause'] = 'motion step became too small'
    elif well_fit:
        log['break_cause'] = 'fitness is high enough'
    else:
        log['break_cause'] = 'warning warning warning'

    if verbose:
        dX_constrained, dX = estimate_motion(X0, gradient_map, X_corr_stacked)
        max_mot_bef = np.sqrt(dX[:,0]**2 + dX[:,1]**2).max() * config['opt_rate']
        fitness_bef = estimate_fitness(X0, fitness_map).mean()

        # befor - after
        log['iteration'] = (0, itr)
        log['motion'] = (max_mot_bef, max_motion)
        log['fitness'] = (fitness_bef, fitness)

        termination_message = '''
        optimization loop terminated in {:.5f} seconds, stopped because "{:s}"
        \t measure:\t -before  \t -after
        \t iteration:\t {:d} \t \t {:d}
        \t maximum motion: {:.5f} \t {:.5f}
        \t fitness:\t {:.5f} \t {:.5f}
        '''.format(time.time()-tic,
                   log['break_cause'],
                   log['iteration'][0], log['iteration'][1],
                   log['motion'][0], log['motion'][1],
                   log['fitness'][0], log['fitness'][1])
        print ( termination_message )
    
    X_optimized = X.copy()
    return X_optimized, log


################################################################################
###################################################### region segmentation stuff
################################################################################
def find_intersection(A,B):
    '''
    intersection between two arrays
    which is a pattern matching of 2-sequence entry between input arrays
    a pair of sequential points corresponds to a line
    the objective is to find if there is a line contained by both input arrays
    

    A,B are two numpy 2d.arrays 
    A = [ (p1x,p1y), (p2x,p2y), ...] , B = ...
    
    This method searches for matching PAIRS OF POINTS
    The output is a list of [(p(i)_x, p(i)_y), (p(i+1)_x,p(i+1)_y)] that are 
    present in both A and B 

    note that the input to this method is the vertices array of a mpath
    this means first point is repeated at the end and end-to-first is accounted
    fo. Otherwise:
    a = np.concatenate([ A, np.atleast_2d(A[0,:]) ])
    b = np.concatenate([ B, np.atleast_2d(B[0,:]) ])
    '''

    # put smaller array in b to minimize the for-loop iteration
    a,b = (A,B) if A.shape[0] > B.shape[0] else (B,A)

    ### for-loop
    # a_loc=[]
    # for a_idx in range(a.shape[0]-1):
    #     for b_idx in range(b.shape[0]-1):
    #         if np.allclose(a[a_idx:a_idx+2, :], b[b_idx:b_idx+2, :]):
    #             a_loc.append( a_idx )
    #             break

    # ### a and b stacked to 3D, and a for-loop for b_template
    # a_loc = []
    # a_stacked = np.stack( [a, np.roll(a,-1,axis=0)], axis=1)
    # for b_idx in range(b.shape[0]-1):
    #     b_template_stacked = np.stack( [b[b_idx:b_idx+2, :] for _ in range(a.shape[0])], axis=0 )
    #     diff = np.abs(a_stacked-b_template_stacked).sum(axis=-1).sum(axis=-1)
    #     a_loc += np.nonzero(diff < np.spacing(10**2) )[0].tolist()

    ### both b and a stacked to 4D
    # slightly better than 3D since we still have a for-loop here
    # it would have been nicer if b_stacked_4D was better
    a_stacked = np.stack( [a, np.roll(a,-1,axis=0)], axis=1)
    # axis=0 is the iteration over b_template, axis=1 is the iteration over a_template
    a_stacked_4d = np.stack( [a_stacked
                              for b_idx in range(b.shape[0]-1)],
                             axis=0)
    b_stacked_4d = np.stack( [ np.stack( [b[b_idx:b_idx+2, :] for _ in range(a.shape[0])], axis=0 )
                              for b_idx in range(b.shape[0]-1)],
                             axis=0)
    diff = np.abs(a_stacked_4d-b_stacked_4d).sum(axis=-1).sum(axis=-1)
    mask = np.where(diff<np.spacing(10**2), True, False )
    a_loc = np.nonzero( np.any(mask,axis=0) )[0]

    intersection = [a[a_idx:a_idx+2, :] for a_idx in a_loc]
    return intersection


################################################################################
def detect_border_lines(pathes):
    '''

    since the direction of pathes is always CCW:
    one of the pathes must be reversed for a successful template match
    '''
    border_lines = []    
    for path_a, path_b in itertools.combinations(pathes, 2):
        intersection = find_intersection(A=path_a.vertices, B= np.flip(path_b.vertices,axis=0) )
        border_lines += intersection
    return border_lines


################################################################################
def rasterize_region_segments(pathes, image_shape, ogm=None):
    '''
    pathes: a list of "matplotlib.path"es
    each path represents a segmented region of the open space

    image_shape: a tuple of two
    the shape of image which the pathes belong to

    ogm: np.2darray (bitmap image )
    the occupancy grid map for "regseg transfer"
    If "ogm" is None, it is assumed the open space of the map corresponds to the
    union of the pathes. This is the case where the objective is to just 
    rasterize the region segmentaitions from pathes.
    If "ogm" is provided, it is assumed the open space of the map does not 
    correspond to the union of the pathes. This is the case where the objective
    is to transfere the region segmentation from one map to another
    IMPORTNT NOTE: regardless of whether "pathes" and "ogm' are from the same 
    original map or not, they must be in the same frame of references. I.e. the 
    "ogm" must have been transfered to the frame of reference of the map which
    yielded the set of "pathes"


    rasterized np.2darray
    has the same size as "image_shape"
    its dtype is uint
    labels zero is for unlabeled areas (occupied and unexplored) and the 
    segmented regions have labels from 1 to len(pathes)
    '''
    rasterized = np.zeros(image_shape, dtype=np.uint16)
    xx, yy = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    xy =  np.array( [xx.flatten(), yy.flatten()] ).T
    for idx, path in enumerate(pathes):
        contained_idx = np.nonzero( path.contains_points(xy) )[0]
        # label zero is reserved for unlabelel regions
        rasterized[ xy[contained_idx,1], xy[contained_idx,0]] = idx+1


    # if ogm is provided, then the open space of the subject map does not
    # corresponds to the union of the pathes, so some pixels should not have
    # been labeled. Here, we set the label of those labels to zero.
    if ogm is not None:
        occ_thr = 200
        # close_idx is the coordinates (x,y) of occupied and unexplored pixels
        close_idx = np.roll( np.array( np.nonzero( ogm < occ_thr ) ).T, 1, axis=1 )
        rasterized[close_idx[:,1], close_idx[:,0]] = 0

    return rasterized

################################################################################
def detect_defect_in_label_image(label_image, occupancy_image,
                                 region_min_size=2000):
    '''
    This method is specifically designed for detecting two types of errors in 
    labels images where the region segmentation is from another map.
    For further explanation see the regseg transfer in:
        rasterize_region_segments()

    error type 1: small regions

    error type 2: src/dst mis-alignment
    there could be pixels belonging to openspaces of the occupancy map that are
    assigned label 0. This happens due to mis-alignment of src-dst and some 
    unlabeled regions of dst map overlap some open spaces of the src map and 
    prevent the openspaces of src map to receive any label other than 0.


    assumption:
    label==0 -> unlabeled regions (occupied or unxplored)

    output 
    2darrany 
    boolian with the same size as the input images
    those whose labels are assumed to be correct have False flag,
    those whose labels are detected as defective are flagged True
    '''

    ########## detection of error type 1
    # counting the population of all present labels
    labels_count = np.array([ (lbl, np.count_nonzero(label_image==lbl))
                              for lbl in np.unique(label_image) ])
    # lbl, cnt = labels_count[:,0], labels_count[:,1]

    # masking those regions smaller than a threshold as false label
    small_region_labels = labels_count[ np.argwhere( labels_count[:,1]<region_min_size ), 0][:,0]
    
    # pixels belonging to small regions are flagged (set to True)
    defective_mask = np.full(label_image.shape, fill_value=False)
    for srl in small_region_labels: defective_mask = np.logical_or( label_image == srl, defective_mask)
    # defective_mask = np.any( anp.stack([label_image == srl for srl in small_region_labels], axis=2), axis=2)

    ########## detection of error type 1    
    # those pixels that are assinged label==0 due to mis-alignment of src/dst
    label_missing_mask = np.logical_and( occupancy_image>200, label_image==0)
    defective_mask = np.logical_or( label_missing_mask, defective_mask)

    return defective_mask


################################################################################
def propagate_labels(defective_mask, defective_label_image, sequenced_labels=True):
    '''
    this method 

    '''
    ####################################
    ###### distance to closest, cv2.dist
    ####################################
    # setting pixels from pixels with defective labels to zero
    # so they don't mess up the search for nearest_layer
    zeroed_label_image = np.where(defective_mask, 0, defective_label_image )

    # for each label in src_img_labeled, creat a mask image and compute it's distance map
    # stack all distance maps in axis=2 (indices in axis=2 are labels in src_img_labeled)
    # for each un-labeled pixel, fined argmin stacked array along axis=2, which is closest
    unique_labels = np.unique(zeroed_label_image)
    unique_labels = np.delete(arr=unique_labels, obj=np.argwhere(unique_labels==0) )
    stacked = np.stack([ cv2.distanceTransform( np.where(zeroed_label_image==l, 0, 255).astype(np.uint8),
                                                cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
                             for l in unique_labels], axis=2)
    nearest_layer = np.argmin( stacked, axis=2 )

    defective_pixels = np.roll( np.array( np.nonzero(defective_mask) ).T, 1,axis=1)
    defective_pixels_label = unique_labels[ nearest_layer[defective_pixels[:,1], defective_pixels[:,0]] ]

    propagated_label_image = defective_label_image.copy()
    propagated_label_image[defective_pixels[:,1], defective_pixels[:,0]] = defective_pixels_label


    if sequenced_labels:
        tmp = np.zeros(propagated_label_image.shape)
        for new_label, label in enumerate( np.unique(propagated_label_image) ):
            tmp = np.where(propagated_label_image==label, new_label, tmp)
        propagated_label_image = tmp

    return propagated_label_image

################################################################################
def get_labeled_skiz_n_transition_points(img, labeled_img, max_dis=1):
    '''
    extract the skiz of the input image (img) and convert then to point 
    coordinates. then find their corresponding labels from the second input 
    image (labeled_img)

    inputs
    ------
    img: numpy.2darray
    occupancy grid map

    labeled_img: numpy.2darray
    region segmentation labels for each pixels in the "img"

    parameters
    ----------
    max_dis (int: default 1)
    for two skiz points with different lablels to be associated, their
    distance should be below "max_dis"

    outputs
    -------
    skiz_pts: numpy.2darray 
    coordinate of points on the skiz 

    skiz_pts_labels: numpy.1darray
    region labels of points in skiz_pts, read from labeled_img

    '''

    ##### extract skiz in bitmap form
    thr = 200 #unexplored = occupied
    _, img_bin = cv2.threshold(img.astype(np.uint8), thr, 255 , cv2.THRESH_BINARY)
    
    img_skiz = mapali._skiz_bitmap(img_bin, invert=False, return_distance=False)

    ##### convert skiz bitmap to coordinates and fetch their labels
    skiz_pts = np.roll( np.array( np.nonzero( img_skiz > img_skiz.max()/2. ) ).T, 1,axis=1)
    skiz_pts_labels = labeled_img[skiz_pts[:,1], skiz_pts[:,0]]

    ##### finding transition points
    transition_points = {}
    for lbl_0,lbl_1 in itertools.combinations(np.unique(skiz_pts_labels), 2):
        # pts_lbl_0 and pts_lbl_1 are skiz points with labels lbl_0, lbl_1
        pts_lbl_0 = skiz_pts[ np.nonzero(skiz_pts_labels==lbl_0)[0], :]
        pts_lbl_1 = skiz_pts[ np.nonzero(skiz_pts_labels==lbl_1)[0], :]
        dists = scipy.spatial.distance.cdist(pts_lbl_0, pts_lbl_1)
    
        brd_pts_idx_0, brd_pts_idx_1 = np.nonzero( dists <= max_dis )
        if brd_pts_idx_0.shape[0] > 0 :
            # pts_lbl_0[brd_pts_idx_0,:] -> skiz points with (label==lbl_0) that are near skiz points with (label==lbl_1)
            # pts_lbl_1[brd_pts_idx_1,:] -> skiz points with (label==lbl_1) that are near skiz points with (label==lbl_0)
            # for every pair of points, transition is their average
            transition_points[(lbl_0,lbl_1)] = np.stack( (pts_lbl_0[brd_pts_idx_0,:],
                                                          pts_lbl_1[brd_pts_idx_1,:]),
                                                         axis=2 ).mean(axis=2)

    return skiz_pts, skiz_pts_labels, transition_points


################################################################################
def merge_transition_points(transition_points, max_dist=20):
    '''
    TODO: to cluster of points above each other will fail this menthod :(

    input
    -----
    transition_points (dict)
    keys are the tuple of two neighbouring regions
    values are list (np.2darray Nx2) of skiz points on the border


    parameters
    ----------
    max_dist (int: default 20)
    points closer than this value will be merged

    output
    ------
    mt_pts_dic (dict)
    keys are the tuple of two neighbouring regions
    values are single points (np.1darray 2,) as transition points

    mt_pts_arr (np.2darray Nx2)
    all points in value fields of mt_pts_dic, compiled into a numpy array
 

    description
    -----------
    clustering all border points close to each other, while keeping an eye for
    regions with two transition points. Clustering of the points is done by
    thresholding their distance after sorting them. It is assumed that the
    threshold between intra-cluster and inter-cluster distance can be safely 
    set to 20 for instance.
    '''

    if np.any([len(pts)==0 for pts in transition_points.values()]):
        msg = '''
        \t Hey chief, you screwed up!
        \t Some fields are empty, Don't send me dictionaries with empty fields!
        \t \t -truly yours,
        \t \t -merge_transition_points()
        '''
        raise (StandardError(msg))

    mt_pts_dic = {}  # merged transition points
    for key,pts in transition_points.iteritems():
        # take all the points between two regseg key[0], key[1]

        # sort the points, first accoding to x and then y
        # NOTE: in "np.lexsort" last key set in the tuple is the primary
        pts_sort_idx = np.lexsort( (pts[:,1], pts[:,0]) )
        pts_sorted = pts[pts_sort_idx, :]
        
        # find the distance between consecutive poitns in pts_sorted
        # and find consecutive points with a distance more than max_dist
        pt2pt_distance = np.sqrt((np.diff( pts_sorted, axis=0 )**2).sum(axis=1))
        split_idx = np.nonzero( pt2pt_distance > max_dist )[0]
        
        # represent the array of all transition points with a list of single points
        mt_pts_dic[key] = [cluster.mean(axis=0) for cluster in np.split( pts_sorted, split_idx+1)]
    
    mt_pts_arr = np.array([ pt for pts in mt_pts_dic.values() for pt in pts ])
    # print('pts_arr: \t', np.any(np.isnan(mt_pts_arr)))

        
        
    return mt_pts_dic, mt_pts_arr

################################################################################
def segmentation_with_transition_points(img, transition_points,
                                        region_min_size=2000,
                                        sequenced_labels=True):
    '''
    for a more comprehensive inline comments see:
    segmentation_with_transition_points__single_iteration()
    which is obsolete now, but kept it for comments   

    inputs
    ------
    img: np.2darray (h x w)
    original occupancy maps 

    transition_points: np.2darray (N x 2)

    parameters
    ----------
    region_min_size: (int - default 2000)
    regions smaller than this size will be flagged as defective and will recieve
    a neighboring label

    sequenced_labels (boolean default: True)
    If True, makes sure the output label_image has a full range

    output
    ------
    label_image: np.2darray (h x w)

    '''

    ########################################
    ########################################
    ########################################
    def iterate(img, marker, region_min_size):
        '''
        provided with an image and markers,
        detects defective or missing labels
        and fix it with watershed
        '''
        dummy_label = marker.max()+1 # false label for unexplored area
        
        # detect defectives and add them to zeros, for watershed to add marker
        defective_labels_mask = detect_defect_in_label_image(marker, img, region_min_size)
        marker = np.where(defective_labels_mask, 0, marker)
        
        # "watershed" won't leave any pixel unlabeled. Inevitably one region will grow to
        _, img_bin = cv2.threshold(img.astype(np.uint8), thr, 255 , cv2.THRESH_BINARY)
        watershed_marker = np.where(img_bin==0, dummy_label, marker )
        
        # "watershed" is stupid! it wants a 3-channel image!
        watershed_image = np.stack( [ img_bin.astype(np.uint8) for _ in range(3) ], axis=2)
        watershed_marker = cv2.watershed(watershed_image, watershed_marker)
        
        # {0}: unlabeled regions
        # {1,...}: labels of different regions
        marker = np.where( np.logical_or(watershed_marker==-1, watershed_marker==dummy_label), 0, watershed_marker)
        return marker
    ########################################        
    ########################################
    ########################################


    ########## padding binary image with circles at transition points
    thr = 200 # for binarization: unexplored = occupied

    # get distance map for the radius of the circles 
    _, img_bin = cv2.threshold(img.astype(np.uint8), thr, 255 , cv2.THRESH_BINARY)
    img_dis = cv2.distanceTransform(img_bin, cv2.DIST_L2,  maskSize=cv2.DIST_MASK_PRECISE)

    # eroding src iamge
    img_padded = img.copy()
    img_padded = cv2.erode(img_padded, np.ones((3,3),np.uint8), iterations = 5)

    # sorting all circles (cent,rad) 
    center_radius = { (int(pt[0]), int(pt[1])): int( img_dis[ int(pt[1]), int(pt[0])] )
                      for pt in transition_points if int( img_dis[ int(pt[1]), int(pt[0])] ) > 1 }
    sorted_keys = sorted(center_radius.keys(), key=lambda k: center_radius[k], reverse=True)

    ########################################
    ############################ First round
    ########################################
    # first round of creating marker: in presence of erosion and all circles
    # markers are created here, and updated in the following iterations

    # padding with all circles 
    img_circled = img_padded.copy()        
    for (x,y), r in center_radius.iteritems():
        cv2.circle(img_circled, center=(x,y), radius=r, color=0, thickness=-1)

    # watershed: in presence of erosion and all circles
    _, img_circled_bin = cv2.threshold(img_circled.astype(np.uint8), thr, 255 , cv2.THRESH_BINARY)
    _, markers = cv2.connectedComponents( img_circled_bin )

    ########################################
    ############# circle removing iterations
    ########################################
    # at each iteration the biggest circle is removed and its place
    # is subjected to "watershed"
    while len(center_radius) > 0:
        # pop and dump the bigest circle
        center_radius.pop( sorted_keys.pop(0) )

        # padding with circles
        img_circled = img_padded.copy()        
        for (x,y), r in center_radius.iteritems():
            cv2.circle(img_circled, center=(x,y), radius=r, color=0, thickness=-1)

        markers = iterate(img_circled, markers, region_min_size)

    ########################################
    ########## last round to recover erosion
    ########################################
    markers = iterate(img, markers, region_min_size)
        
    ########################################
    ######### storing results in the out put
    ########################################
    label_image = markers.copy()

    ########## correting the orders of labels
    if sequenced_labels:
        tmp = np.zeros(label_image.shape)
        for new_label, label in enumerate( np.unique(label_image) ):
            tmp = np.where(label_image==label, new_label, tmp)
        label_image = tmp
    
    return label_image


################################################################################
def segmentation_with_transition_points__single_iteration(img, transition_points,
                                                          region_min_size=2000,
                                                          sequenced_labels=True):
    '''
    OBSOLETE - just kept it for the inline comments

    inputs
    ------
    img: np.2darray (h x w)
    original occupancy maps 

    transition_points: np.2darray (N x 2)

    parameters
    ----------
    region_min_size: (int - default 2000)
    regions smaller than this size will be flagged as defective and will recieve
    a neighboring label

    sequenced_labels (boolean default: True)
    If True, makes sure the output label_image has a full range

    output
    ------
    label_image: np.2darray (h x w)

    '''
    ########## padding binary image with circles at transition points
    thr = 200 # for binarization: unexplored = occupied

    # get distance map for the radius of the circles 
    _, img_bin = cv2.threshold(img.astype(np.uint8), thr, 255 , cv2.THRESH_BINARY)
    img_dis = cv2.distanceTransform(img_bin, cv2.DIST_L2,  maskSize=cv2.DIST_MASK_PRECISE)

    # padding with circles
    # "img_padded" is the eroded and circle-padded version of "img"
    img_padded = img.copy()
    img_padded = cv2.erode(img_padded, np.ones((3,3),np.uint8), iterations = 5)

    for pt in transition_points:
        x,y = int(pt[0]), int(pt[1])
        r = int( img_dis[ y, x] ) # int( 1.1 *img_dis[ y, x] )
        # Thickness of the circle outline, if positive.
        # Negative thickness means that a filled circle is to be drawn.
        cv2.circle(img_padded, center=(x,y), radius=r, color=0, thickness=-1)

    ########## segment regions with connected components
    # img_padded_bin is the eroded and circle-padded version
    _, img_padded_bin = cv2.threshold(img_padded.astype(np.uint8), thr, 255 , cv2.THRESH_BINARY)

    # con_com_marker is the markers of the connected components
    # 0 is reserved for the unlabeled, but, there are too many regions including small corners
    # labels are complete, ie. it can't be [0,2,3..] where 1 is missing
    _, con_com_marker = cv2.connectedComponents( img_padded_bin )


    ########## detecting defective lablels
    # defective_labels_mask flags those pixels of open-space in "src_img" that also:
    # 1) were the circle paddings,
    # 2) were regions (conncected components) smaller than "region_min_size", and
    # 3) edges pixels next to occupied pixels that missed label due to erosion
    defective_labels_mask = detect_defect_in_label_image(con_com_marker, img, region_min_size=region_min_size)


    ########## fixing defected labels
    ''' approach #1: (not in use)
    propagate labels to "unlabeld" and "open-cells" from closest labeled pixels
    There is a problem due to mis-alignment, some corners of one regions might
    take the label of neighboring region. Then, in those case, segmenting a
    label-region whose label spreaded to neighborig regions (room-region) with
    np.where() will result in more than one poly for label. While one belongs to
    the current label, the smallers one don't. '''
    # # 0 is reserved for the unlabeled, but, there are too many regions including small corners
    # # labels are complete, ie. it can't be [0,2,3..] where 1 is missing
    # label_image = propagate_labels(defective_labels_mask, con_com_marker, sequenced_labels=True)

    ''' approach #2:
    propagate labels to "unlabeld" and "open-cells" from closest labeled pixels with
    watershed which won't cross occupancy barriers. this fixes the problem of corners
    inheriting labels from neighboring room.
    
    The problem is where a big circle is reaching a doorway, it allows the label of 
    the room to extend to hall/corridor area '''
    # src_mrk is the a copy of connected components labels, expect defective labels 
    # are set to zero, ie unlabeled, to be "watershed"
    # note that small regions are removed here, so labels are NOT complete, ie. it
    # CAN be [0,2,3..] where 1 is missing
    con_com_marker = np.where(defective_labels_mask, 0, con_com_marker)
    
    # "watershed" won't leave any pixel unlabeled. Inevitably one region will grow to
    # cover the unexplored regions. So dedicate a new label to occupied/unexploted
    # areas (src_bin==0) then reject the label from "marker" later 
    # in watershed_marker here -> {0}:unlabeled, {1,...,fl-1}:connectected component labels, fl:unexplored area
    # note that labels are NOT complete, ie. it CAN be [0,2,3..] where 1 is missing
    dummy_label = con_com_marker.max()+1 # false label for unexplored area
    watershed_marker = np.where(img_bin==0, dummy_label , con_com_marker )
    
    # "watershed" is stupid! it wants a 3-channel image!
    watershed_image = np.stack( [ img_bin.astype(np.uint8) for _ in range(3) ], axis=2)
    # watershed_marker here (labels are NOT complete, CAN be [0,2,3..] where 1 is missing):
    # {-1}: unlabeled which referes to the fronterier of the water shed (to be ignored)
    # {0}: there is no labled ==0! there were "whatershed"
    # {1,...,dl-1}: connectected component labels, after watershed
    # dl: unexplored area (no be ignored)
    watershed_marker = cv2.watershed(watershed_image, watershed_marker)

    # {0}: unlabeled regions
    # {1,...}: labels of different regions
    label_image = np.where( np.logical_or(watershed_marker==-1, watershed_marker==dummy_label), 0, watershed_marker)

    ########## correting the orders of labels
    if sequenced_labels:
        tmp = np.zeros(label_image.shape)
        for new_label, label in enumerate( np.unique(label_image) ):
            tmp = np.where(label_image==label, new_label, tmp)
        label_image = tmp
    
    return label_image

################################################################################
def label_map_with_pathes(img, pathes):
    '''
    label 0 is reserved for unsegmented regions

    '''
    lbl_img = np.zeros(img.shape)
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]) )
    xy = np.stack( (xx.flatten(), yy.flatten()), axis=1 )

    for p_idx, path in enumerate(pathes):
        contained_idx = np.nonzero( path.contains_points(xy) )[0]
        #labels zero is for unsegmented regions
        lbl_img[ xy[contained_idx,1], xy[contained_idx,0] ] = p_idx +1

    return lbl_img
    
################################################################################
def extract_pathes_from_label_image(label_image, dilate_itr=0):
    '''
    assuming label 0 is reserved for unsegmented regions

    paramters
    ---------
    dilate_itr (int: default 0)
    for 3D point cloud segmentation, it is desired that the path to be slightly
    bigger than the countour of the open-space, so that it contains the
    surrounding walls/stuff. [I guess 10 should be good enough]


    NOTE
    ----
    If "lbl_reg" returns more than one contour, then it will be split
    into two regions according to pathes. It can be remedied by:

    tmp = lbl_reg
    while np.unique( cv2.connectedComponents(tmp) ).shape[0] > 2:
        tmp = cv2.erode(tmp, np.ones((3,3),np.uint8), iterations = 1)

    This will dilate the lbl_reg until all sub-regions are merged
    '''
    unique_labels = np.unique(label_image)
    unique_labels.sort()

    pathes = []
    for lbl in unique_labels[1:]:
        lbl_reg = np.where(label_image==lbl, 255, 0).astype(np.uint8)
        lbl_reg = cv2.dilate(lbl_reg, np.ones((3,3),np.uint8), iterations=dilate_itr )
        polys = polyIO._extract_contours( lbl_reg, only_save_one_out=False )['out']
        pathes += [ mapali._create_mpath(poly) for poly in polys]

    return pathes

################################################################################
def warp_twice(image, warp1, warp2, out_shape=None):
    '''
    if a tform is estimated from src to dst,
    1) to warp image from src to dst: warp(image, tform.inverse)
    2) to transform points from src to dst: tform(points)

    this method expects the warp1 and warp2 to be inversed alread before passing

    Usage
    -----
    src image to dst frame
    warp_twice(src_img, warp1=tform_align.inverse, warp2=tform_opt.inverse)

    dst image to src frame
    warp_twice(dst_img, warp1=tform_opt, warp2=tform_align)


    note
    ----
    for transforming points, warps should be inversed (not the order)

    src pts to dst frame
    tform_opt( tform_align( pts ) )

    dst pts to src frame
    tform_align.inverse( tform_opt.inverse( pts ) )
    '''
    out_shape = image.shape if out_shape is None else out_shape

    warped1 = skimage.transform.warp(image, warp1, output_shape=(out_shape),
                                     preserve_range=True, mode='constant', cval=127)
    warped2 = skimage.transform.warp(warped1, warp2, output_shape=(out_shape),
                                     preserve_range=True, mode='constant', cval=127)

    return warped2

################################################################################
def transfer_region_segmentation(src_image, dst_pathes, config, dst_image_shape=None):
    ''''''

    ########## setting default values for paramters
    if dst_image_shape is None: dst_image_shape = src_image.shape
    if 'region_min_size' not in config.keys(): config['region_min_size'] = 2000
    if 'max_dist' not in config.keys(): config['max_dist'] = 20

    ########## warp src image to dst frame to be labeled
    src_img_in_dst_frame = warp_twice(src_image,
                                      warp1=config['tform_align'].inverse,
                                      warp2=config['tform_opt'].inverse,
                                      out_shape=dst_image_shape)

    ########## segmentation of src image [ in dst frame ]
    # pixel labeling with pathes
    src_img_labeled_in_dst_frame = rasterize_region_segments( pathes=dst_pathes,
                                                              image_shape=dst_image_shape,
                                                              ogm=src_img_in_dst_frame)
    # detecting defects in src regseg
    src_def_mask_in_dst_frame = detect_defect_in_label_image( label_image=src_img_labeled_in_dst_frame,
                                                              occupancy_image=src_img_in_dst_frame,
                                                              region_min_size=config['region_min_size'])
    # correcting defects in src regseg
    src_img_labeled_in_dst_frame = propagate_labels( defective_mask=src_def_mask_in_dst_frame,
                                                     defective_label_image=src_img_labeled_in_dst_frame,
                                                     sequenced_labels=True)

    ########## skiz extraction and labeling [ in dst frame ]
    # extract, skiz points, their corresponding labels, and transition points
    skiz_pts, skiz_lbl, trn_pts = get_labeled_skiz_n_transition_points(src_img_in_dst_frame,
                                                                       src_img_labeled_in_dst_frame)
    src_transition_points_in_dst_frame = trn_pts
    # src_skiz_pts_in_dst_frame = skiz_pts
    # src_skiz_pts_labels = skiz_lbl

    ########## transform transition points to src frame    
    src_transition_points = { key: config['tform_align'].inverse( config['tform_opt'].inverse( pts ) )
                              for key,pts in src_transition_points_in_dst_frame.iteritems() }
    
    src_mt_pts_dict, src_mt_pts = merge_transition_points(src_transition_points, max_dist=config['max_dist'])

    ########## generating region segmentation from transition points [in src frame]
    src_labeled = segmentation_with_transition_points(src_image, src_mt_pts, region_min_size=config['region_min_size'])
    
    return src_labeled


# ################################################################################
# ################################################################################
# ################################################################################
# def find_peaks_naive(x, win_siz, peak=['max','min'][0], max_peak_n=None):
#     '''
#     this is a very simple peak detection method
#     1) it assumes the signal is periodic
#     2) the only criterial for a point to be peak is to be bigger/smaller than all 
#     its neighbors

#     Additional criterion: maximum number of peaks
#     '''
    
#     nieghbors = np.stack([np.roll(x, r) for r in range(-win_siz,win_siz+1)], axis=0)

#     if peak =='max':
#         nieghbors_max = nieghbors.max(axis=0)
#         idx = np.nonzero(x >= nieghbors_max)[0]
#     elif peak =='min':
#         nieghbors_min = nieghbors.min(axis=0)
#         idx = np.nonzero(x <= nieghbors_min)[0]

#     if max_peak_n is not None and idx.shape[0] > max_peak_n:
#         srt_idx = np.argsort(x[idx])
#         if peak =='max': srt_idx = np.flip(srt_idx, axis=0)
#         idx = idx[ srt_idx[:max_peak_n] ]    

#     return idx


# ################################################################################
# #################################### point to line segment distance - 3 versions
# ################################################################################
# ###############################  useful for the construction of distance map and
# ############################### fitness map from a distribution of line segments
# ################################################################################
# def point2segment_distance_1p1s(segment, p):
#     '''
#     1p1s: 1-point 1-segment version
    
#     '''

#     p2p_distance = lambda p1, p2: np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

#     p1 = segment[0] # (x,y)
#     p2 = segment[1]
    
#     seg_len = p2p_distance(p1, p2)
#     d = np.dot( p-p1 , p2-p1) /np.abs( seg_len )**2
#     px = p1 + ((p2-p1) * d)
    
#     xMin, xMax = min([p1[0], p2[0]]), max([p1[0], p2[0]])
#     yMin, yMax = min([p1[1], p2[1]]), max([p1[1], p2[1]])
#     eps = np.spacing(10)
#     if not(xMin-eps <= px[0] <= xMax+eps) or not(yMin-eps <= px[1] <= yMax+eps):
#         d1 = p2p_distance(p1,px)
#         d2 = p2p_distance(p2,px)
#         px = p1 if d1<d2 else p2
        
#     return p2p_distance( px, p) # point to segment distance


# ################################################################################
# def point2segment_distance_mp1s(segment, pts):
#     '''
#     mp1s: multi-points 1-segment version
    
#     Input 
#     -----
#     segment: np.array 2d
#     segment[0,:]=p1, segment[1,:]=p2

#     pts: np.array 2d
#     first index, is index to points, second index is index to x,y
    
#     output
#     ------
#     distances: np.array - 1d
    
#     How it works
#     ------------
#     projection of a point "p" on line (not segment) defined by "p1, p2"  is
#     computed by
#     "px = p1 + p1_2_prj_dis*v2"
#     where,
#     v() -> vector
#     v1, v2 = v(p1, point), v(p1,p2)
#     p1_2_prj_dis := distance of p1 to projection of point on v2
#     p1_2_prj_dis = dot(v1, v2) / len(v2)^2
    
#     to find the distance of the point to segment, we have to check if the
#     projected point is insided the interval of the segment, otherwise return 
#     the distance to the one ending point of the segment that is closer to the
#     peojected point.
    
#     Vectorization speed-up
#     ----------------------
#     this method takes ~0.37% as much time as 1p1s version for 10000 points, 
#     however, operating on single points, it takes almost twice as much time as
#     the single version.
#     '''
    
#     p1, p2 = seg[0], seg[1]
    
#     # proj_on_lin is an array of all the points projected on the LINE of the segment
#     V1, V2 = (pts - p1).astype(np.float), (p2 - p1).astype(np.float)
#     p1_2_prj_dis = np.dot( V1 , V2 ) / ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
#     proj_on_lin = segment[0] + V2 * np.atleast_2d(p1_2_prj_dis).T
    
#     # check if points projected on line are inside the segment's inteval
#     xMin, xMax = min([p1[0], p2[0]]), max([p1[0], p2[0]])
#     yMin, yMax = min([p1[1], p2[1]]), max([p1[1], p2[1]])
#     eps = np.spacing(10)
    
#     p_in_seg_x = np.logical_and ( xMin-eps <= proj_on_lin[:,0], proj_on_lin[:,0] <= xMax+eps )
#     p_in_seg_y = np.logical_and ( yMin-eps <= proj_on_lin[:,1], proj_on_lin[:,1] <= yMax+eps )
#     p_in_seg = np.logical_and (p_in_seg_x, p_in_seg_y) # shape -> (pts.shape[0],)
#     p_in_seg = np.atleast_2d( p_in_seg ).T # shape -> (pts.shape[0], 1)
    
#     # check the distance between peojected points and p1/p2 of the segement,
#     # and pick the p1 or p2, which is closer to the point itseld,
#     # ie. to take the ending point of segment as the projection point
#     p1_tile = np.tile(p1,(pts.shape[0],1)) # shape -> (pts.shape[0], 2)
#     p2_tile = np.tile(p2,(pts.shape[0],1)) # shape -> (pts.shape[0], 2)
#     proj_dis_to_p1 = np.sqrt((p1_tile[:,0]-proj_on_lin[:,0])**2 + (p1_tile[:,1]-proj_on_lin[:,1])**2) # shape: (pts.shape[0],)
#     proj_dis_to_p2 = np.sqrt((p2_tile[:,0]-proj_on_lin[:,0])**2 + (p2_tile[:,1]-proj_on_lin[:,1])**2) # shape: (pts.shape[0],)
#     cond = np.atleast_2d( proj_dis_to_p1<proj_dis_to_p2 ).T  # shape: (pts.shape[0],1)
#     alternative = np.where(cond, p1_tile, p2_tile ) # shape: (pts.shape[0],2)
    
#     # if a projected point is inside the segment, return that as result,
#     # otherwise return closest ending point from "alternative"
#     px = np.where(p_in_seg, proj_on_lin, alternative)  # shape: (pts.shape[0],2)
#     pts2seg_dis = np.sqrt((pts[:,0]-px[:,0])**2 + (pts[:,1]-px[:,1])**2)
    
#     return pts2seg_dis
 
# ################################################################################
# def point2segment_distance_mpms(segments, pts):
#     '''
#     '''
#     # for all stacked arrays: ax0-pts[idx] / ax1-seg[idx] / ax2-xy[idx]
#     pts_stacked = np.stack( [pts for _ in range(segments.shape[0])] ,axis=1)
#     p1s_stacked = np.stack( [segments[:,0,:] for _ in range(pts.shape[0])] ,axis=0)
#     p2s_stacked = np.stack( [segments[:,1,:] for _ in range(pts.shape[0])] ,axis=0) 
    
#     V1 = (pts_stacked - p1s_stacked).astype(np.float)
#     V2 = (p2s_stacked - p1s_stacked).astype(np.float)
    
#     V1_dot_V2 = V1[:,:,0]*V2[:,:,0] + V1[:,:,1]*V2[:,:,1]
#     del V1
#     seg_len_sqr = (segments[:,0,0]-segments[:,1,0])**2 + (segments[:,0,1]-segments[:,1,1])**2
#     seg_len_sqr_stacked = np.stack( [seg_len_sqr for _ in range(pts.shape[0])] ,axis=0)
#     p1_2_prj_dis = V1_dot_V2 / seg_len_sqr_stacked
#     proj_on_lin = p1s_stacked + V2 * np.stack( [p1_2_prj_dis, p1_2_prj_dis], axis=2)
#     del V2, p1_2_prj_dis, seg_len_sqr, seg_len_sqr_stacked

#     # could just find the min-max of segments and then stack with pts.shape[0]
#     xMin = np.stack([p1s_stacked[:,:,0], p2s_stacked[:,:,0]], axis=2).min(axis=2) - np.spacing(100)
#     xMax = np.stack([p1s_stacked[:,:,0], p2s_stacked[:,:,0]], axis=2).max(axis=2) + np.spacing(100)
#     yMin = np.stack([p1s_stacked[:,:,1], p2s_stacked[:,:,1]], axis=2).min(axis=2) - np.spacing(100)
#     yMax = np.stack([p1s_stacked[:,:,1], p2s_stacked[:,:,1]], axis=2).max(axis=2) + np.spacing(100)
    
#     p_in_seg_x = np.logical_and ( xMin <= proj_on_lin[:,:,0], proj_on_lin[:,:,0] <= xMax )
#     p_in_seg_y = np.logical_and ( yMin <= proj_on_lin[:,:,1], proj_on_lin[:,:,1] <= yMax )
#     p_in_seg = np.logical_and (p_in_seg_x, p_in_seg_y) # shape -> (pts.shape[0],)
#     p_in_seg = np.stack( [p_in_seg, p_in_seg], axis=2 )
#     del xMin, xMax, yMin, yMax, p_in_seg_x, p_in_seg_y

#     # check the distance between peojected points and p1/p2 of the segement,
#     # and pick the p1 or p2, which is closer to the point itseld,
#     # ie. to take the ending point of segment as the projection point
#     proj_dis_to_p1 = np.sqrt((p1s_stacked[:,:,0]-proj_on_lin[:,:,0])**2 + (p1s_stacked[:,:,1]-proj_on_lin[:,:,1])**2)
#     proj_dis_to_p2 = np.sqrt((p2s_stacked[:,:,0]-proj_on_lin[:,:,0])**2 + (p2s_stacked[:,:,1]-proj_on_lin[:,:,1])**2)
#     p1_is_closer = proj_dis_to_p1<proj_dis_to_p2
#     p1_is_closer = np.stack( [p1_is_closer, p1_is_closer], axis=2 )
#     alternative = np.where(p1_is_closer, p1s_stacked, p2s_stacked ) # shape: (pts.shape[0],2)
#     del p1_is_closer, proj_dis_to_p1, proj_dis_to_p2, p1s_stacked, p2s_stacked

#     # if a projected point is inside the segment, return that as result,
#     # otherwise return closest ending point from "alternative"
#     px = np.where(p_in_seg, proj_on_lin, alternative)  # shape: (pts.shape[0],2)
#     del p_in_seg, proj_on_lin, alternative
    
#     pts2seg_dis = np.sqrt((pts_stacked[:,:,0]-px[:,:,0])**2 + (pts_stacked[:,:,1]-px[:,:,1])**2)

#     distance = pts2seg_dis.min(axis=1) # tried with .sum(axis=1), it goes crazy!
    
#     return distance



