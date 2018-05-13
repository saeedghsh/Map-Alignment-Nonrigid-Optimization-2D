'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi

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

import sys
new_paths = [
    u'../arrangement/',
    u'../Map-Alignment-2D',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import time
import numpy as np
import skimage.transform

# map alignment package
import map_alignment.map_alignment as mapali
import map_alignment.mapali_plotting as maplt

# nonrigid optimization of alignment package
import optimize_alignment.optimize_alignment as optali
import optimize_alignment.plotting as optplt

################################################################################
###################################################################### functions
################################################################################
def _extract_target_file_name(img_src, img_dst, method=None):
    '''
    This method takes the names of the input files, and construct a name for the
    output files based on the input files.
    '''
    spl_src = img_src.split('/')
    spl_dst = img_dst.split('/')
    if len(spl_src)>1 and len(spl_dst)>1:
        # include the current directories name in the target file's name
        tmp = spl_src[-2]+'_'+spl_src[-1][:-4] + '__' + spl_dst[-2]+'_'+spl_dst[-1][:-4]
    else:
        # only include the input files' name in the target file's name
        tmp = spl_src[-1][:-4] + '__' + spl_dst[-1][:-4]

    return tmp if method is None else method+'_'+ tmp


################################################################################
################################################################################
################################################################################
if __name__ == '__main__':
    '''
    Parameters and Options:
    -----------------------
    --img_src 'the-address-name-to-sensor-map'
    --img_dst 'the-address-name-to-layout-map'

    --hyp_sel_metric 'fitness' # use fitness quality measure for map alignemnt (only sensor-layout)
    --hyp_sel_metric 'matchscore' # use arrangement matchscore for map alignemnt (sensor-layout and sensor-sensor)

    -visualize
    -save_to_file (also visualizes)
    -multiprocessing

    Examples:
    ---------
    python demo.py --img_src 'map_sample/F5_04.png' --img_dst 'map_sample/F5_layout.png' --hyp_sel_metric 'fitness' -visualize -multiprocessing

    python demo.py --img_src 'map_sample/F5_04.png' --img_dst 'map_sample/F5_layout.png' --hyp_sel_metric 'matchscore' -visualize -save_to_file -multiprocessing
    '''

    ################################################################################
    ######################################################## INITIALIZATION from CLI
    ################################################################################
    args = sys.argv

    ###### fetching options from input arguments
    # options are marked with single dash
    options = []
    for arg in args[1:]:
        if len(arg)>1 and arg[0] == '-' and arg[1] != '-':
            options += [arg[1:]]

    ###### fetching parameters from input arguments
    # parameters are marked with double dash,
    # the value of a parameter is the next argument
    listiterator = args[1:].__iter__()
    while 1:
        try:
            item = next( listiterator )
            if item[:2] == '--':
                exec(item[2:] + ' = next( listiterator )')
        except:
            break

    ##### setting defaults values for parameter
    if 'hyp_sel_metric' not in locals():
        hyp_sel_metric = ['fitness', 'matchscore'][1]
    visualize = True if 'visualize' in options else False
    save_to_file = True if 'save_to_file' in options else False
    multiprocessing = True if 'multiprocessing' in options else False

    if save_to_file: save_to_file = _extract_target_file_name(img_src, img_dst, method=None)

    ################################################################################
    #################################################### FIRST STAGE - MAP ALIGNMENT
    ################################################################################

    ########################################
    ############### Alignment Configurations
    ########################################

    ########## lock and load
    lnl_config = {'binary_threshold_1': 200, # with numpy - for SKIZ and distance
                  'binary_threshold_2': [100, 255], # with cv2 - for trait detection
                  'traits_from_file': False, # else provide yaml file name
                  'trait_detection_source': 'binary_inverted',
                  'edge_detection_config': [50, 150, 3], # thr1, thr2, apt_size
                  'peak_detect_sinogram_config': [15, 15, 0.15], # [refWin, minDist, minVal]
                  'orthogonal_orientations': True} # for dominant orientation detection

    ########## arrangement (and pruning)
    arr_config = {'multi_processing':4, 'end_point':False, 'timing':False,
                  'prune_dis_neighborhood': 2,
                  'prune_dis_threshold': .075,#075, # home:0.15 - office:0.075
                  'occupancy_threshold': 200} # cell below this is considered occupied

    ########## pick the winning tform_align
    sel_config = {'multiprocessing': multiprocessing,
                  'too_many_tforms': 3000,
                  'dbscan_eps': 0.051,
                  'dbscan_min_samples': 2}

    ########## Tform_Align generation
    hyp_config = { 'scale_mismatch_ratio_threshold': .3, # .5,
                   'scale_bounds': [.5, 2], #[.1, 10]
                   'face_occupancy_threshold': .5}

    ########################################
    ########################## Map Alignment
    ########################################

    ########## image loading, SKIZ, distance transform and trait detection
    src_results, src_lnl_t = mapali._lock_n_load(img_src, lnl_config)
    dst_results, dst_lnl_t = mapali._lock_n_load(img_dst, lnl_config)

    ########## arrangement and pruning
    src_results['arrangement'], src_arr_t = mapali._construct_arrangement(src_results, arr_config)
    dst_results['arrangement'], dst_arr_t = mapali._construct_arrangement(dst_results, arr_config)

    ########## Tform_Align generation
    tforms, hyp_gen_t, tforms_total, tforms_after_reject = mapali._generate_hypothese(src_results['arrangement'],
                                                                                      src_results['image'].shape,
                                                                                      dst_results['arrangement'],
                                                                                      dst_results['image'].shape,
                                                                                      hyp_config)

    if tforms.shape[0] == 0:
        raise( Exception('map alignment failed, nothing to optimize...') )
        # print ('no tform survived ... setting to identity... ')
        # tform_align = skimage.transform.AffineTransform()
        # n_cluster, sel_win_t = 0, 0


    ########## pick the winning tform_align
    if hyp_sel_metric == 'matchscore': # with arrangement match score
        tform_align, n_cluster, sel_win_t = mapali._select_winning_hypothesis(src_results['arrangement'],
                                                                              dst_results['arrangement'],
                                                                              tforms, sel_config)

    elif hyp_sel_metric == 'fitness': # with fitness match score
        X_original = optali.get_corner_sample(src_results['image'], maxCorners=500, qualityLevel=0.01, minDistance=25)
        ### construction of MOTION FIELD (of the destination map)
        fitness_sigma = 50
        fit_map, grd_map = optali.get_fitness_gradient(dst_results['image'],
                                                       fitness_sigma=fitness_sigma,
                                                       grd_k_size=3,
                                                       normalize=True)
        ### select winner
        n_points = X_original.shape[0]
        tic = time.time()
        fn = [ optali.estimate_fitness(tf._apply_mat(X_original, tf.params), fit_map).mean() for tf in tforms ]

        import operator
        index, value = max(enumerate(fn), key=operator.itemgetter(1))
        tform_align = tforms[index]
        n_cluster, sel_win_t = 0, time.time()-tic

    else:
        raise( Exception('unknown value for hyp_sel_metric') )

    arr_match_score = mapali._arrangement_match_score(src_results['arrangement'], dst_results['arrangement'], tform_align)

    ########################################
    ###################### reporting results
    ########################################
    mapali_details = {
        'src_lnl_t': src_lnl_t,
        'dst_lnl_t': dst_lnl_t,
        'src_arr_t': src_arr_t,
        'dst_arr_t': dst_arr_t,
        'hyp_gen_t': hyp_gen_t,
        'sel_win_t': sel_win_t,
        'tforms_total': tforms_total,
        'tforms_after_reject': tforms_after_reject,
        'n_cluster': n_cluster
    }

    ########## print the elapsed time
    time_key = ['src_lnl_t', 'dst_lnl_t', 'src_arr_t', 'dst_arr_t', 'hyp_gen_t']
    print ('total alignment time: {:.5f}'.format( np.array([mapali_details[key] for key in time_key]).sum() ) )

    ################################################################################
    ####################################### SECOND STAGE - OPTIMIZATION OF ALIGNMENT
    ################################################################################
    opt_config = {
        # dst
        'fitness_sigma': 50,
        'gradient_ksize': 3,
        'correlation_sigma': 800,

        # src - good feature to track
        'edge_refine_dilate_itr': 5, #3
        'max_corner': 500,
        'quality_level': .01,
        'min_distance': 25,

        # optimization - loop
        'opt_rate': 10**1,          # optimization rate
        'max_itr': 10000,           # maximum number of iterations
        'tol_fit': .9999, #.99        # break if (fitness > tol_fit)
    }
    opt_config['tol_mot'] = 0.001 # * opt_config['opt_rate'] # break if (max_motion < tol_mot)

    ########################################
    ########## POINT SAMPLING occupied cells (of the source image)
    ########################################
    opt_tic = time.time()
    X_original = optali.get_corner_sample( src_results['image'],
                                           edge_refine_dilate_itr=opt_config['edge_refine_dilate_itr'],
                                           maxCorners=opt_config['max_corner'],
                                           qualityLevel=opt_config['quality_level'],
                                           minDistance=opt_config['min_distance'])
    X_aligned = tform_align._apply_mat( X_original, tform_align.params )

    ########################################
    ########### construction of MOTION FIELD (of the destination map)
    ########################################
    fit_map, grd_map = optali.get_fitness_gradient( dst_results['image'],
                                                    fitness_sigma=opt_config['fitness_sigma'],
                                                    grd_k_size=opt_config['gradient_ksize'],
                                                    normalize=True)

    ########################################
    ################# data point correlation (for averaging motion)
    ########################################
    X_correlation = optali.data_point_correlation(X_aligned, correlation_sigma=opt_config['correlation_sigma'], normalize=True)

    ########################################
    ########################### Optimization
    ########################################
    X_optimized, optimization_log = optali.optimize_alignment( X0=X_aligned, X_correlation=X_correlation,
                                                               gradient_map=grd_map, fitness_map=fit_map,
                                                               config=opt_config,
                                                               verbose=True)
    print ('total optimization time: {:.5f}'.format( time.time() - opt_tic ) )

    tform_opt = skimage.transform.PiecewiseAffineTransform()
    tform_opt.estimate(X_aligned, X_optimized)

    '''
    Note on how to generate X_aligned and X_optimized from tforms:
    print ( np.allclose(X_aligned , tform_align(X_original))  )
    print ( np.allclose(X_aligned , tform_align._apply_mat(X_original, tform_align.params))  )
    print ( np.allclose(X_optimized , tform_opt(X_aligned)) )
    '''

    if visualize or save_to_file:
        ########################################
        ########### warpings of the source image
        ########################################
        # warp source image according to alignment tform_align
        src_img_aligned = skimage.transform.warp(src_results['image'], tform_align.inverse, #_inv_matrix,
                                                 output_shape=(dst_results['image'].shape),
                                                 preserve_range=True, mode='constant', cval=127)

        # warp aligned source image according to optimization
        src_img_optimized = skimage.transform.warp(src_img_aligned, tform_opt.inverse,
                                                   output_shape=(dst_results['image'].shape),
                                                   preserve_range=True, mode='constant', cval=127)


        ########## save/plotting alignment, motion of points and optimized alignment
        optplt.plot_alignment_motion_optimized(dst_results['image'],
                                               src_img_aligned, src_img_optimized, grd_map,
                                               X_aligned, X_optimized, save_to_file)

    if save_to_file:
        ########## saving results in a numpy file
        np.save(save_to_file+'.npy',
                {'arr_match_score': arr_match_score,
                 'X_original': X_original,
                 'tform_align': tform_align,
                 'tform_opt': tform_opt,
                 'mapali_details': mapali_details,
                 'optimization_log': optimization_log})
