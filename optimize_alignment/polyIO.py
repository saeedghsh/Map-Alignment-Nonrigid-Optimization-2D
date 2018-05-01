from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

from map_alignment import map_alignment as mapali

################################################################################
def load_poly_file(file_name):
    '''
    the poly format this method is concerned with is the one used by:
    http://masc.cs.gmu.edu/wiki/Dude2D

    input [.poly format]
    --------------------
    int(#polygones) - first line of the file
    section1
    section2
    ...
    sectionn

    for each section of the .poly file:
    int(#vertices),  str(poly_type{'in', 'out'}) - first line of the section
    float(v[0].x), float(v[0].y)
    ...
    1 2 3 .... int(#vertices) - last line of the section

    output
    ------
    poly_dic: a dictionary with two keys: ['in', 'out']
    poly_dic['key']: the field to each key is a list of polygons
    each polygon is an ordered array of vertices [numpy.ndarray of shape (nx2)]
    '''
    f_lines = [line for line in open(file_name, 'r')]

    number_of_poly = f_lines.pop(0)
    poly_dic = {'in':[], 'out':[]}
    while len(f_lines)>0:
        line = f_lines.pop(0)
        spl = [s.replace('\n', '') for s in line.split(' ')]
        spl = [s for s in spl if len(s)>0]

        if ('in' in spl) or ('out' in spl):
            n_vert, poly_type = int(spl[0]), spl[1]
            poly_dic[poly_type].append( np.array( [ [float(s.replace('\n', '')) for s in f_lines.pop(0).split(' ')]
                                                    for idx in range(n_vert) ] ))
            f_lines.pop(0) # list of vertices, range(1,n_vert)
        
    return poly_dic

################################################################################
def save_poly_file(file_name, poly_dic):
    '''
    the poly format this method is concerned with is the one used by:
    http://masc.cs.gmu.edu/wiki/Dude2D

    input
    -----
    poly_dic: a dictionary with two keys: ['in', 'out']
    poly_dic['key']: the field to each key is a list of polygons
    each polygon is an ordered array of vertices [numpy.ndarray of shape (nx2)]
        

    output [.poly format]
    --------------------
    int(#polygones) - first line of the file
    section1
    section2
    ...
    sectionn

    for each section of the .poly file:
    int(#vertices),  str(poly_type{'in', 'out'}) - first line of the section
    float(v[0].x), float(v[0].y)
    ...
    1 2 3 .... int(#vertices) - last line of the section

    '''
    # open file to write
    f = open(file_name, 'w')

    # set the first line of the file
    number_of_poly = len(poly_dic['in']) + len(poly_dic['out'])
    f.write(str(number_of_poly)+'\n')

    # set the sections
    for poly_type in ['out', 'in']:
        for poly in poly_dic[poly_type]:
            n_vert = poly.shape[0]
            
            # set first line of the section
            f.write( ' '.join( [str(n_vert), poly_type] )+'\n' )

            # set middle lines of the section
            for pts in poly:
                f.write( ' '.join( [str(pts[0]), str(pts[1])] ) + '\n' )

            # set last line of the section
            f.write( ' '.join( [str(idx+1) for idx in range(n_vert)] )+'\n' )

    f.close()

################################################################################
def _convert_to_poly_dict(contours, only_save_one_out=False):
    '''
    '''
    # detecting 'in' and 'out' polys
    poly_dic = {'in':[], 'out':[]}
    for idx1, cnt1 in enumerate(contours):
        poly_type = 'out'
        for idx2, cnt2 in enumerate(contours):
            if cv2.pointPolygonTest(cnt2,(cnt1[0,0],cnt1[0,1]),measureDist=False) > 0 :
                poly_type = 'in'
                break
        poly_dic[poly_type].append( cnt1 )

    # Correct orientations (out: CW / positive area - in: CCW / negative area)
    for idx, poly in enumerate(poly_dic['in']):
        if cv2.contourArea(poly, oriented=True) > 0 :
            poly_dic['in'][idx] = np.flipud(poly)

    for idx, poly in enumerate(poly_dic['out']):
        if cv2.contourArea(poly, oriented=True) < 0 :
            poly_dic['out'][idx] = np.flipud(poly)
            
    # save only one 'out'
    if only_save_one_out:
        biggest_cnt_idx = np.argmax([c.shape[0] for c in poly_dic['out']])
        poly_dic['out'] = [ poly_dic['out'][biggest_cnt_idx] ]
        
        # remove "in" polygons that are not inside any "out" polygon
        # this might happen because we migh have rejected an "out" polygon
        for idx in range(len(poly_dic['in'])-1,-1,-1):
            x, y = poly_dic['in'][idx][0,:]
            # (inside or outside or on the contour: +1, -1, 0 respectively)
            if cv2.pointPolygonTest(poly_dic['out'][0],(x,y), measureDist=False) < 0:
                poly_dic['in'].pop(idx)

    # # check 'in' and 'out' polys have correct orientations
    # for poly in poly_dic['out']: assert cv2.contourArea(poly, oriented=True) > 0 
    # for poly in poly_dic['in']: assert cv2.contourArea(poly, oriented=True) < 0 

    return poly_dic

################################################################################
def _reject_small_contours(contours, min_vert, min_area):
    '''
    '''
    for idx in range(len(contours)-1, -1, -1):
        small = False if min_area is None else np.abs(cv2.contourArea(contours[idx])) <= float(min_area)
        short = False if min_vert is None else contours[idx].shape[0] <= float(min_vert)
        if small or short: contours.pop(idx)

    return contours


################################################################################
def _extract_contours(image,
                      only_save_one_out=False,
                      min_vert=None, min_area=None):
    '''
    '''

    # convert to binary
    thr = 200 # unexplored = occupied
    _, img_bin = cv2.threshold(image.astype(np.uint8), thr, 255 , cv2.THRESH_BINARY)

    ##### detect contours
    im2, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = [c[:,0,:] for c in contours]

    ##### reject small 
    if (min_vert is not None) or (min_area is not None):
        contours = _reject_small_contours(contours, min_vert, min_area)

    ##### convert contours from list of numpy arrays to dict with 'in' and 'out'
    poly_dic = _convert_to_poly_dict(contours, only_save_one_out)

    return poly_dic


################################################################################
################################################################################
################################################################################
def load_corresponding_polys_and_scale(image_name):
    ''''''

    ##### loading all poly files corresponding to the input image
    image_name_split = image_name.split('/')
    dude_results_path = '/'.join(image_name_split[:-1])
    dude_results_path += '/dude_result_'+image_name_split[-1].split('.')[0]+'/'
    dude_results_list = [fn for fn in os.listdir(dude_results_path) if fn[-4:]=='poly']
    pathes = [ mapali._create_mpath( load_poly_file(dude_results_path+poly_name)['out'][0] ) # assuming a single poly
               for poly_name in dude_results_list ]
    
    ##### scaling the polys
    # original size
    main_poly = load_poly_file( image_name+'.poly' )['out'][0] # assuming a single poly in the list
    dX = main_poly[:,0].max() - main_poly[:,0].min()
    dY = main_poly[:,1].max() - main_poly[:,1].min()
    # current size
    verts = np.concatenate([p.vertices for p in pathes],axis=0)
    dx = verts[:,0].max() - verts[:,0].min()
    dy = verts[:,1].max() - verts[:,1].min()
    
    scale = dX/dx # = dY/dy
    for p in pathes: p.vertices *= scale


    return pathes






################################################################################
################################################################## Visualization
################################################################################
def _visualize_save(image, poly_dic):
    '''
    '''
    fig, axes = plt.subplots(1,1, figsize=(20,12))
    axes.imshow(image, cmap='gray', interpolation='nearest', origin='lower')
    axes.set_title('# contours: {:d}'.format( sum([len(poly_dic[k]) for k in poly_dic.keys()] )))

    # plot poly_dic['out']
    for idx, contour in enumerate(poly_dic['out']):
        axes.text( contour[0,0], contour[0,1], 
                   'cnt-out{:d} / {:d}xVertices'.format(idx, contour.shape[0]),
                   fontdict={'color':'b',  'size': 8})    
        axes.plot(contour[:,0], contour[:,1], 'b.-')
    
    # plot poly_dic['in']
    for idx, contour in enumerate(poly_dic['in']):
        axes.text( contour[0,0], contour[0,1], 
                   'cnt-in{:d} / {:d}xVertices'.format(idx, contour.shape[0]),
                   fontdict={'color':'r',  'size': 8})    
        axes.plot(contour[:,0], contour[:,1], 'r.-')


    plt.tight_layout()
    plt.show()
