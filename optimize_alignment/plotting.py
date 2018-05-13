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
import numpy as np
import cv2

# import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import optimize_alignment as optali

################################################################################
def plot_point_sampling(image, X, marker_='r.'):
    ''''''
    fig, axes = plt.subplots(1,1, figsize=(10,10))
    axes.imshow(image, cmap='gray', alpha=1., interpolation='nearest', origin='lower')
    axes.plot(X[:,0], X[:,1], marker_)
    axes.axis('off')
    plt.tight_layout()
    plt.show()

################################################################################
def plot_gradient_quiver(image, gradient, X=None, skp=20, save_to_file=False):
    '''
    skp: sparsing the quiver visualization
    '''
    fig, axes = plt.subplots(1,1, figsize=(10,10))
    axes.imshow(image, cmap='gray', alpha=1., interpolation='nearest', origin='lower')

    if X is not None:
        axes.plot(X[:,0], X[:,1], 'r.')

    X_q, Y_q = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    U_q, V_q = gradient.real, gradient.imag
    M_q = np.absolute(gradient) # == np.hypot(U, V)

    axes.quiver(X_q[::skp, ::skp], Y_q[::skp, ::skp],
                U_q[::skp, ::skp], V_q[::skp, ::skp],
                M_q[::skp, ::skp],
                pivot='mid',
                scale= 50 * M_q.max()
            )


    if save_to_file:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        axes.axis('off')
        fig.savefig(save_to_file)#, bbox_inches='tight', pad_inches=0)

    plt.tight_layout()
    plt.show()


################################################################################
def plot_dx_constrained(gradient, X, dX, dX_constrained):
    '''
    '''
    fig, axes = plt.subplots(1,2, figsize=(20,12), sharex=True, sharey=True)

    axes[0].set_title('without local constraint')
    axes[0].imshow(np.absolute(gradient), cmap='gray', alpha=1., interpolation='nearest', origin='lower')
    axes[0].plot(X[:,0], X[:,1], 'r,')
    X_q, Y_q = X[:,0], X[:,1]
    U_q, V_q = dX[:,0], dX[:,1]
    M_q = np.hypot(U_q, V_q)
    axes[0].quiver(X_q, Y_q, U_q, V_q, M_q, pivot='tail', scale= 50 * M_q.max() )

    axes[1].set_title('without local constraint')
    axes[1].imshow(np.absolute(gradient), cmap='gray', alpha=1., interpolation='nearest', origin='lower')
    axes[1].plot(X[:,0], X[:,1], 'r,')
    X_q, Y_q = X[:,0], X[:,1]
    U_q, V_q = dX_constrained[:,0], dX_constrained[:,1]
    M_q = np.hypot(U_q, V_q)
    axes[1].quiver(X_q, Y_q, U_q, V_q, M_q, pivot='tail', scale= 50 * M_q.max() )

    plt.tight_layout()
    plt.show()



################################################################################
def plot_point_before_after_optimization(gradient, X0,X1, axes=None, img_alpha=0.5, set_title=True):
    ''''''
    internal_plot = True if axes is None else False

    if internal_plot:
        fig, axes = plt.subplots(1,1, figsize=(20,12))

    if set_title: axes.set_title(' points before and after motion, over abs(gradient) ')
    axes.imshow(np.absolute(gradient), cmap='gray', alpha=img_alpha, interpolation='nearest', origin='lower')

    axes.plot(X0[:,0], X0[:,1], 'r.')
    axes.plot(X1[:,0], X1[:,1], 'g.')
    for (x1,y1),(x2,y2) in zip(X0, X1): axes.plot([x1,x2], [y1,y2],'r-')

    if internal_plot:
        plt.tight_layout()
        plt.show()
    else:
        return axes


################################################################################
def plot_aligned_optimized(dst_image,
                           src_image_aligned, src_image_optimized,
                           save_to_file):
    '''
    OBSOLETE: use plot_alignment_motion_optimized() instead
    '''
    fig, axes = plt.subplots(1,2, figsize=(20,12))

    axes[0].set_title(' alignment result ')
    axes[0].imshow(dst_image, origin='lower', cmap='gray', alpha=.5, clip_on=True)
    axes[0].imshow(src_image_aligned, origin='lower', cmap='gray', alpha=.5, clip_on=True)

    axes[1].set_title(' optimization result ')
    axes[1].imshow(dst_image, origin='lower', cmap='gray', alpha=.5, clip_on=True)
    axes[1].imshow(src_image_optimized, origin='lower', cmap='gray', alpha=.5, clip_on=True)

    if save_to_file:
        plt.savefig(save_to_file+'.png', bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    else:
        plt.tight_layout()
        plt.show()

################################################################################
def plot_alignment_motion_optimized(dst_image, src_img_aligned, src_img_optimized,
                                    gradient_map, X_aligned, X_optimized,
                                    save_to_file):
    ''''''
    fig, axes = plt.subplots(1,3, figsize=(20,12))

    axes[0].set_title(' alignment result ')
    axes[0].imshow(dst_image, origin='lower', cmap='gray', alpha=.5, clip_on=True)
    axes[0].imshow(src_img_aligned, origin='lower', cmap='gray', alpha=.5, clip_on=True)

    axes[1].set_title(' point motion over gradient field ')
    axes[1] = plot_point_before_after_optimization(gradient_map, X_aligned, X_optimized, axes=axes[1], img_alpha=0.5)

    axes[2].set_title(' optimization result ')
    axes[2].imshow(dst_image, origin='lower', cmap='gray', alpha=.5, clip_on=True)
    axes[2].imshow(src_img_optimized, origin='lower', cmap='gray', alpha=.5, clip_on=True)


    if save_to_file:
        plt.savefig(save_to_file+'.png', bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    else:
        plt.tight_layout()
        plt.show()


################################################################################
def plot_triangualtion_of_PiecewiseAffineTransform(dst_image,
                                                   pwa_tform,
                                                   X_aligned, X_optimized ):

    fig, axes = plt.subplots(1,1, figsize=(12,12))

    axes.imshow(dst_image, origin='lower', cmap='gray', alpha=.5, clip_on=True)

    axes.triplot(pwa_tform._tesselation.points[:,0],
                 pwa_tform._tesselation.points[:,1],
                 pwa_tform._tesselation.simplices.copy(), 'b,-')

    axes = plot_point_before_after_optimization(dst_image, X_aligned, X_optimized, axes=axes, img_alpha=0.0)

    plt.tight_layout()
    plt.show()

################################################################################
def plot_tesselation_motion_heatmap(dst_image, src_img_aligned,
                                    X_aligned, X_optimized,
                                    tform_opt,
                                    plot_points=False):
    ''''''

    # motion of each triangle is the average of the motion of its vertices
    tri_motion = np.array([ np.sqrt( ((X_aligned[idx,:]-X_optimized[idx,:])**2).sum(axis=1) ).sum()
                              for idx in tform_opt._tesselation.simplices ]) / 3


    fig, axes = plt.subplots(1,2, figsize=(20,12))

    axes[0].set_title('histogram of the motion of the vertices of triangles')
    axes[0].hist(tri_motion,bins=20)

    # normalizing the motions to (0, .5) to be used for opacity of patches
    tri_motion = (tri_motion - tri_motion.min()) / ( 2* (tri_motion.max()-tri_motion.min()) )

    axes[1].set_title('based on the motion of the vertices of each triangle')
    axes[1].imshow(dst_image, origin='lower', cmap='gray', alpha=.5)
    axes[1].imshow(src_img_aligned, origin='lower', cmap='gray', alpha=.5)
    codes = (1, 2, 2, 79)
    for s_idx, idx in enumerate(tform_opt._tesselation.simplices):
        pts = X_aligned[idx,:]
        verts = (pts[0,:], pts[1,:], pts[2,:], pts[0,:])
        axes[1].add_patch( mpatches.PathPatch( mpath.Path(verts, codes),
                                            facecolor='r', edgecolor='none',#'r',
                                            alpha=tri_motion[s_idx]) )

    if plot_points:
        axes[1] = plot_point_before_after_optimization(dst_image, X_aligned, X_optimized, axes=axes[1], img_alpha=0.0)

    plt.tight_layout()
    plt.show()

    # tri_fitness = np.array([ fit_map[X_aligned[idx,1].astype(int), X_aligned[idx,0].astype(int)].sum() / 3.
    #                          for idx in tform_opt._tesselation.simplices ])
    # tri_fitness = (tri_fitness - tri_fitness.min()) / (tri_fitness.max()-tri_fitness.min())
    # axes[0].set_title('based on the fitness of the vertices of each triangle')
    # axes[0].imshow(dst_results['image'], origin='lower', cmap='gray', alpha=.5, clip_on=True)
    # axes[0].imshow(src_img_aligned, origin='lower', cmap='gray', alpha=.5, clip_on=True)
    # codes = (1, 2, 2, 79)
    # for s_idx, idx in enumerate(tform_opt._tesselation.simplices):
    #     pts = X_aligned[idx,:]
    #     verts = (pts[0,:], pts[1,:], pts[2,:], pts[0,:])
    #     axes[0].add_patch( mpatches.PathPatch( mpath.Path(verts, codes),
    #                                            facecolor='r', edgecolor='r',
    #                                            alpha=( 1-tri_fitness[s_idx])/1 ) )

################################################################################
def plot_double_fitness(src_img, dst_img, tform_align, tform_opt, fitness_sigma=5):
    ''''''

    src_fit_map,_ = optali.get_fitness_gradient(src_img,
                                                fitness_sigma=fitness_sigma,
                                                grd_k_size=3, normalize=True)

    dst_fit_map,_ = optali.get_fitness_gradient(dst_img,
                                                fitness_sigma=fitness_sigma,
                                                grd_k_size=3, normalize=True)


    src_fit, dst_fit, X_src, X_dst = optali.double_fitness(src_img, dst_img, tform_align, tform_opt,
                                                           fitness_sigma=fitness_sigma,
                                                           src_contour_dilate_itr=20)

    fig, axes = plt.subplots(1,2, figsize=(20,10))

    # axes[0].imshow(src_img, origin='lower', cmap='gray', alpha=.8)
    axes[0].imshow(src_fit_map, origin='lower', cmap='gray', alpha=.8)
    axes[0].plot(X_dst['in_src_frame'][:,0], X_dst['in_src_frame'][:,1], 'r,', alpha=1)
    rgba_colors = np.zeros((X_dst['in_src_frame'].shape[0],4))
    rgba_colors[:, 1] = 1.0
    rgba_colors[:, 3] = dst_fit
    axes[0].scatter(X_dst['in_src_frame'][:,0], X_dst['in_src_frame'][:,1], marker=',', color=rgba_colors)

    # axes[1].imshow(dst_img, origin='lower', cmap='gray', alpha=.8)
    axes[1].imshow(dst_fit_map, origin='lower', cmap='gray', alpha=.8)
    axes[1].plot(X_src['in_dst_frame'][:,0], X_src['in_dst_frame'][:,1], 'r,', alpha=1)
    rgba_colors = np.zeros((X_src['in_dst_frame'].shape[0],4))
    rgba_colors[:, 1] = 1.0
    rgba_colors[:, 3] = src_fit
    axes[1].scatter(X_src['in_dst_frame'][:,0], X_src['in_dst_frame'][:,1], marker=',', color=rgba_colors)

    plt.tight_layout()
    plt.show()

################################################################################
def plot_motion_decoherency(dst_img, tform_opt, X_aligned, X_optimized, X_correlation):
    ''''''

    md = optali.get_motion_decoherency(tform_opt, X_aligned, X_optimized, X_correlation)

    fig, axes = plt.subplots(1,1, figsize=(10,10))

    axes.imshow(dst_img, origin='lower', cmap='gray', alpha=.8)

    axes.plot(X_aligned[:,0], X_aligned[:,1], 'b,', alpha=1)
    for p1,p2 in zip(X_aligned, X_optimized):
        axes.plot([p1[0], p2[0]], [p1[1], p2[1]] , 'b-', alpha=1)
    print (md.shape)

    rgba_colors = np.zeros((X_aligned.shape[0],4))
    rgba_colors[:, 0] = 1.0
    rgba_colors[:, 3] = md/md.max()
    axes.scatter(X_optimized[:,0], X_optimized[:,1], marker=',', color=rgba_colors)

    plt.tight_layout()
    plt.show()


################################################################################
def plot_pathes(axes, pathes):
    ''''''

    # ### plot patches with collection tool
    # patches = [mpatches.PathPatch(p) for p in pathes]
    # collection = mcollections.PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.7)
    # colors = np.linspace(0, 1, len(pathes))
    # collection.set_array(np.array(colors))
    # axes.add_collection(collection)

    ### plot patches directly
    hsv_col = np.ones( (len(pathes)+1,3) )
    hsv_col[:,0] = np.linspace(0, 1, len(pathes)+1)
    rgb_col = matplotlib.colors.hsv_to_rgb(hsv_col)
    for p,c in zip(pathes, rgb_col):
        axes.add_patch( mpatches.PathPatch(p, facecolor=c, edgecolor='none', alpha=0.5) )

    return axes
