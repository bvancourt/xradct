import numpy as np
import matplotlib.pyplot as plt
import importlib
from copy import deepcopy

import voxelhelp

importlib.reload(voxelhelp)

def curvature_loss(vector_field):
    inner_interpolated = (vector_field[2:,1:-1,1:-1,:] +
                          vector_field[1:-1,2:,1:-1,:] +
                          vector_field[1:-1,1:-1,2:,:] +
                          vector_field[:-2,1:-1,1:-1,:] +
                          vector_field[1:-1,:-2,1:-1,:] +
                          vector_field[1:-1,1:-1,:-2,:])/6

    differences = inner_interpolated - vector_field[1:-1,1:-1,1:-1,:]
    return np.sum(differences*differences)

def offset_to_interpolated(vector_field):
    inner_interpolated = (vector_field[2:,1:-1,1:-1,:] +
                          vector_field[1:-1,2:,1:-1,:] +
                          vector_field[1:-1,1:-1,2:,:] +
                          vector_field[:-2,1:-1,1:-1,:] +
                          vector_field[1:-1,:-2,1:-1,:] +
                          vector_field[1:-1,1:-1,:-2,:])/6

    differences = inner_interpolated - vector_field[1:-1,1:-1,1:-1,:]
    return np.pad(differences, ((1,1),(1,1),(1,1),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))

def curvature_loss_with_gradient(vector_field):
    a_sixth = np.array(1/6).astype(vector_field.dtype)
    inner_interpolated = (vector_field[2:,1:-1,1:-1,:] +
                          vector_field[1:-1,2:,1:-1,:] +
                          vector_field[1:-1,1:-1,2:,:] +
                          vector_field[:-2,1:-1,1:-1,:] +
                          vector_field[1:-1,:-2,1:-1,:] +
                          vector_field[1:-1,1:-1,:-2,:])*a_sixth

    differences = inner_interpolated - vector_field[1:-1,1:-1,1:-1,:]
    L = np.sum(differences*differences)

    direct_grad = np.pad(2*differences, ((1,1),(1,1),(1,1),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
    indirect_grad = np.zeros(direct_grad.shape, dtype=vector_field.dtype)
    indirect_grad[1:,:,:] += direct_grad[:-1,:,:]
    indirect_grad[:-1,:,:] += direct_grad[1:,:,:]
    indirect_grad[:,1:,:] += direct_grad[:,:-1,:]
    indirect_grad[:,:-1,:] += direct_grad[:,1:,:]
    indirect_grad[:,:,1:] += direct_grad[:,:,:-1]
    indirect_grad[:,:,:-1] += direct_grad[:,:,1:]

    return L, direct_grad-a_sixth*indirect_grad
    #direct_grad = np.pad(2*differences, ((1,1),(1,1),(1,1),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
    #indirect_grad = (np.roll(direct_grad, 1, axis=0) + 
    #                 np.roll(direct_grad, -1, axis=0) + 
    #                 np.roll(direct_grad, 1, axis=1) + 
    #                 np.roll(direct_grad, -1, axis=1) + 
    #                 np.roll(direct_grad, 1, axis=2) + 
    #                 np.roll(direct_grad, -1, axis=2))/6
##
    #return L, direct_grad - indirect_grad

def unstack_vector_to_xyz(vector_field):
    return vector_field[:,:,:,0], vector_field[:,:,:,1], vector_field[:,:,:,2]

def stack_scalar_fields(x, y, z):
    return np.stack([x, y, z], axis=3)

def index_grid_field(vol_shape):
    y, x, z = np.meshgrid(np.arange(vol.shape[1]), np.arange(vol.shape[0]), np.arange(vol.shape[2]))
    return np.stack([x, y, z], axis=3)

def normalize(f):
    mag = np.sqrt(f[:,:,:,0]*f[:,:,:,0] + f[:,:,:,1]*f[:,:,:,1] + f[:,:,:,2]*f[:,:,:,2])
    f[:,:,:,0] /= mag
    f[:,:,:,1] /= mag
    f[:,:,:,2] /= mag
    return f

def cell_volume_from_diffs(di, dj, dk):
    return np.sum(np.cross(di, dj, axis=3)*dk, axis=3) # not exact; not sure it would be possible to make it exact.

def normalize_diffs_for_cell_volume_one(di, dj, dk):
    initial_cell_volumes = cell_volume_from_diffs(di, dj, dk)
    return di/np.cbrt(initial_cell_volumes)[:,:,:,np.newaxis], dj/np.cbrt(initial_cell_volumes)[:,:,:,np.newaxis], dk/np.cbrt(initial_cell_volumes)[:,:,:,np.newaxis]

def grow_sampling_grid_from_diff_fields(index_grid, di, dj, dk, start_position, i_range, j_range, k_range, grow_order='kij', steps_per_voxel=2):
    sampling_grid = np.zeros([i_range[1]-i_range[0], j_range[1]-j_range[0], k_range[1]-k_range[0], 3])
    sampling_grid[-i_range[0], -j_range[0], -k_range[0], :] = start_position

    if grow_order=='kij':
        sampling_grid = grow_sampling_grid_from_diff_fields(index_grid, dj, dk, di, start_position, i_range, j_range, k_range, grow_order='jki')
        return sampling_grid.transpose([2,0,1,3])
    elif grow_order=='ijk':
        sampling_grid = grow_sampling_grid_from_diff_fields(index_grid, dk, di, dj, start_position, i_range, j_range, k_range, grow_order='jki')
        return sampling_grid.transpose([1,2,0,3])
    elif grow_order=='jki': # obviously it was a mistake to write it for this order, but here we are.
        first_row_upsamp_factor = 4
        first_surf_upsamp_factor = 2

        for j_offset in range(1, j_range[1]): # steps in +j direction
            next_point = sampling_grid[-i_range[0], -j_range[0]+j_offset-1, -k_range[0],:]
            for n in range(steps_per_voxel*first_row_upsamp_factor):
                next_point += np.squeeze(resample_vector_field(dj, next_point[np.newaxis,np.newaxis,np.newaxis,:])/steps_per_voxel/first_row_upsamp_factor)
            sampling_grid[-i_range[0], -j_range[0]+j_offset, -k_range[0], :] = next_point

        for j_offset in range(1, -j_range[0]): # steps in -j direction
            next_point = sampling_grid[-i_range[0], -j_range[0]-j_offset+1, -k_range[0],:]
            for n in range(steps_per_voxel*first_row_upsamp_factor):
                next_point -= np.squeeze(resample_vector_field(dj, next_point[np.newaxis,np.newaxis,np.newaxis,:])/steps_per_voxel/first_row_upsamp_factor)
            sampling_grid[-i_range[0], -j_range[0]-j_offset, -k_range[0], :] = next_point

        # grow center curve into center surface
        for k_offset in range(1, k_range[1]): # steps in +k direction
            next_row = sampling_grid[-i_range[0], :, -k_range[0]+k_offset-1,:]
            for n in range(steps_per_voxel*first_surf_upsamp_factor):
                next_row += np.squeeze(resample_vector_field(dk, next_row[np.newaxis,:,np.newaxis,:]))/steps_per_voxel/first_surf_upsamp_factor
            sampling_grid[-i_range[0], :, -k_range[0]+k_offset, :] = next_row

        for k_offset in range(1, -k_range[0]): # steps in -k direction
            next_row = sampling_grid[-i_range[0], :, -k_range[0]-k_offset+1,:]
            for n in range(steps_per_voxel*first_surf_upsamp_factor):
                next_row -= np.squeeze(resample_vector_field(dk, next_row[np.newaxis,:,np.newaxis,:]))/steps_per_voxel/first_surf_upsamp_factor
            sampling_grid[-i_range[0], :, -k_range[0]-k_offset, :] = next_row

        # grow center surface to full 3D grid
        for i_offset in range(1, i_range[1]): # steps in +k direction
            sampling_grid[-i_range[0]+i_offset, :, :, :] = np.squeeze(
                sampling_grid[-i_range[0]+i_offset-1, :, :, :][:,:,np.newaxis,:]
                + resample_vector_field(di, sampling_grid[-i_range[0]+i_offset-1, :, : ,:][:,:,np.newaxis,:])
            )
        for i_offset in range(1, -i_range[0]): # steps in -k direction
            sampling_grid[-i_range[0]-i_offset, :, :,:] = np.squeeze(
                sampling_grid[-i_range[0]-i_offset+1, :, :,:][:,:,np.newaxis,:]
                - resample_vector_field(di, sampling_grid[-i_range[0]-i_offset+1, :, :,:][:,:,np.newaxis,:])
            )

    return sampling_grid

def resample_vector_field(f, sampling_grid, default_val=(0,0,0), mode='linear'):
    return stack_scalar_fields(
        voxelhelp.resamp_volume(f[:,:,:,0], sampling_grid[:,:,:,0], sampling_grid[:,:,:,1], sampling_grid[:,:,:,2], mode=mode, default_val=default_val[0]),
        voxelhelp.resamp_volume(f[:,:,:,1], sampling_grid[:,:,:,0], sampling_grid[:,:,:,1], sampling_grid[:,:,:,2], mode=mode, default_val=default_val[1]),
        voxelhelp.resamp_volume(f[:,:,:,2], sampling_grid[:,:,:,0], sampling_grid[:,:,:,1], sampling_grid[:,:,:,2], mode=mode, default_val=default_val[2])
    )

def grad_cell_pressure(sampling_grid, mode='approx'):
    # not finished; do not use.
    if mode=='approx': # volumes of parallepiped
        # Not recommended; preliminary results suggest that this is not a very good approximation,
        # although for not-too-curved coordinate systems if might be good enough and preferable to
        # the other method below, becuase it would be faster and use much less memory.
        di = np.zeros(sampling_grid.shape, dtype=np.float32)
        dj = np.zeros(sampling_grid.shape, dtype=np.float32)
        dk = np.zeros(sampling_grid.shape, dtype=np.float32)

        di[1:-1,:,:,:] = (sampling_grid[2:,:,:,:]-sampling_grid[:-2,:,:,:])/2
        di[0,:,:,:] = sampling_grid[1,:,:,:]-sampling_grid[0,:,:,:]
        di[-1,:,:,:] = sampling_grid[-1,:,:,:]-sampling_grid[-2,:,:,:]

        dj[:,1:-1,:,:] = (sampling_grid[:,2:,:,:]-sampling_grid[:,:-2,:,:])/2
        dj[:,0,:,:] = sampling_grid[:,1,:,:]-sampling_grid[:,0,:,:]
        dj[:,-1,:,:] = sampling_grid[:,-1,:,:]-sampling_grid[:,-2,:,:]

        dk[:,:,1:-1,:] = (sampling_grid[:,:,2:,:]-sampling_grid[:,:,:-2,:])/2
        dk[:,:,0,:] = sampling_grid[:,:,1,:]-sampling_grid[:,:,0,:]
        dk[:,:,-1,:] = sampling_grid[:,:,-1,:]-sampling_grid[:,:,-2,:]
        
        base = np.cross(di, dj, axis=3)

        return np.sum(base*dk, axis=3)

def cell_volume(sampling_grid, mode='tetrahedra'):
    if mode=='approx': # volumes of parallepiped
        # Not recommended; preliminary results suggest that this is not a very good approximation,
        # although for not-too-curved coordinate systems if might be good enough and preferable to
        # the other method below, becuase it would be faster and use much less memory.
        di = np.zeros(sampling_grid.shape)
        dj = np.zeros(sampling_grid.shape)
        dk = np.zeros(sampling_grid.shape)

        di[1:-1,:,:,:] = (sampling_grid[2:,:,:,:]-sampling_grid[:-2,:,:,:])/2
        di[0,:,:,:] = sampling_grid[1,:,:,:]-sampling_grid[0,:,:,:]
        di[-1,:,:,:] = sampling_grid[-1,:,:,:]-sampling_grid[-2,:,:,:]

        dj[:,1:-1,:,:] = (sampling_grid[:,2:,:,:]-sampling_grid[:,:-2,:,:])/2
        dj[:,0,:,:] = sampling_grid[:,1,:,:]-sampling_grid[:,0,:,:]
        dj[:,-1,:,:] = sampling_grid[:,-1,:,:]-sampling_grid[:,-2,:,:]

        dk[:,:,1:-1,:] = (sampling_grid[:,:,2:,:]-sampling_grid[:,:,:-2,:])/2
        dk[:,:,0,:] = sampling_grid[:,:,1,:]-sampling_grid[:,:,0,:]
        dk[:,:,-1,:] = sampling_grid[:,:,-1,:]-sampling_grid[:,:,-2,:]
        
        base = np.cross(di, dj, axis=3)

        return np.sum(base*dk, axis=3)
    
    if mode=='tetrahedra':
        # This should exactly fill all space with 24 tetrehedra per voxel. Imagine block of tetrahedra
        # (forming approximately a cube) centered at each voxel. A blender image showing the
        # arrangement of tetrahedra used will hopefully be in documentation somewhere to explain how  
        # these are going to be used, but all 24 tetrahedra have a corner at the center of the block,
        # two corners at corners of the block, and one corner in the middle of a face. The volume
        # of each tetrahedron will be the dot product of one edge with the cross product of two
        # other edges. Only 12 cross products per block are needed because two tetrahedra can
        # share the two edges, the sign just has to be flipped for one of them to avoid getting
        # a negative volume... I will try to get the signs right, but also take the absolute
        # value of each volume at the end just in case.

        # The arrays half_di, half_dj, & half_dk, are the vectors from the center of the 
        # block to the center of a face. These vectors are only explicitly stored for faces
        # pointing in the positive i, j, and k directions becuase the other three vectors can 
        # be obtained by taking the vector to the opposite face of the previous block and flipping
        # the sign. This is exactly equivalent, becuase thecenters of the faces are defined to be
        # eactly half-way between the to centers. 
        
        half_di = np.zeros([sampling_grid.shape[0]+1,
                            sampling_grid.shape[1],
                            sampling_grid.shape[2],
                            sampling_grid.shape[3]])
        half_di[1:-1,:,:,:] = (sampling_grid[1:,:,:,:]-sampling_grid[:-1,:,:,:])/2
        half_di[0,:,:,:] = half_di[1,:,:,:]
        half_di[-1,:,:,:] = half_di[-2,:,:,:]
        
        half_dj = np.zeros([sampling_grid.shape[0],
                            sampling_grid.shape[1]+1,
                            sampling_grid.shape[2],
                            sampling_grid.shape[3]])
        half_dj[:,1:-1,:,:] = (sampling_grid[:,1:,:,:]-sampling_grid[:,:-1,:,:])/2
        half_dj[:,0,:,:] = half_dj[:,1,:,:]
        half_dj[:,-1,:,:] = half_dj[:,-2,:,:]
        
        half_dk = np.zeros([sampling_grid.shape[0],
                            sampling_grid.shape[1],
                            sampling_grid.shape[2]+1,
                            sampling_grid.shape[3]])
        half_dk[:,:,1:-1,:] = (sampling_grid[:,:,1:,:]-sampling_grid[:,:,:-1,:])/2
        half_dk[:,:,0,:] = half_dk[:,:,1,:]
        half_dk[:,:,-1,:] = half_dk[:,:,-2,:]
        
        # Next, we need vectors from the center to each corner of the block. There are 8 corners,
        # but like with the face centers, only half need to be caculated explicitly, since the
        # rest can be optained by flipping the sign of the opposite vector form an ajacent block.
        
        corner_mmm = np.zeros([sampling_grid.shape[0]+1,
                               sampling_grid.shape[1]+1,
                               sampling_grid.shape[2]+1,
                               sampling_grid.shape[3]])
        corner_mmm[1:-1, 1:-1, 1:-1, :] = (sampling_grid[:-1,:-1,:-1,:] - sampling_grid[1:,1:,1:,:])/2
        corner_mmm[0,:,:,:] = corner_mmm[1,:,:,:]
        corner_mmm[-1,:,:,:] = corner_mmm[-2,:,:,:]
        corner_mmm[:,0,:,:] = corner_mmm[:,1,:,:]
        corner_mmm[:,-1,:,:] = corner_mmm[:,-2,:,:]
        corner_mmm[:,:,0,:] = corner_mmm[:,:,1,:]
        corner_mmm[:,:,-1,:] = corner_mmm[:,:,-2,:]
        
        corner_mmp = np.zeros([sampling_grid.shape[0]+1,
                               sampling_grid.shape[1]+1,
                               sampling_grid.shape[2]+1,
                               sampling_grid.shape[3]])
        corner_mmp[1:-1, 1:-1, 1:-1, :] = (sampling_grid[:-1,:-1,1:,:] - sampling_grid[1:,1:,:-1,:])/2
        corner_mmp[0,:,:,:] = corner_mmp[1,:,:,:]
        corner_mmp[-1,:,:,:] = corner_mmp[-2,:,:,:]
        corner_mmp[:,0,:,:] = corner_mmp[:,1,:,:]
        corner_mmp[:,-1,:,:] = corner_mmp[:,-2,:,:]
        corner_mmp[:,:,0,:] = corner_mmp[:,:,1,:]
        corner_mmp[:,:,-1,:] = corner_mmp[:,:,-2,:]
        
        corner_mpp = np.zeros([sampling_grid.shape[0]+1,
                               sampling_grid.shape[1]+1,
                               sampling_grid.shape[2]+1,
                               sampling_grid.shape[3]])
        corner_mpp[1:-1, 1:-1, 1:-1, :] = (sampling_grid[:-1,1:,1:,:] - sampling_grid[1:,:-1,:-1,:])/2
        corner_mpp[0,:,:,:] = corner_mpp[1,:,:,:]
        corner_mpp[-1,:,:,:] = corner_mpp[-2,:,:,:]
        corner_mpp[:,0,:,:] = corner_mpp[:,1,:,:]
        corner_mpp[:,-1,:,:] = corner_mpp[:,-2,:,:]
        corner_mpp[:,:,0,:] = corner_mpp[:,:,1,:]
        corner_mpp[:,:,-1,:] = corner_mpp[:,:,-2,:]
        
        corner_mpm = np.zeros([sampling_grid.shape[0]+1,
                               sampling_grid.shape[1]+1,
                               sampling_grid.shape[2]+1,
                               sampling_grid.shape[3]])
        corner_mpm[1:-1, 1:-1, 1:-1, :] = (sampling_grid[:-1,1:,:-1,:] - sampling_grid[1:,:-1,1:,:])/2
        corner_mpm[0,:,:,:] = corner_mpm[1,:,:,:]
        corner_mpm[-1,:,:,:] = corner_mpm[-2,:,:,:]
        corner_mpm[:,0,:,:] = corner_mpm[:,1,:,:]
        corner_mpm[:,-1,:,:] = corner_mpm[:,-2,:,:]
        corner_mpm[:,:,0,:] = corner_mpm[:,:,1,:]
        corner_mpm[:,:,-1,:] = corner_mpm[:,:,-2,:]
        
        # cross products for bottom of cube (pointing out the bottom (-z)).
        cross_mcm = np.cross( corner_mmm[:-1,:-1,:-1,:], corner_mpm[:-1,1:,:-1,:],  axis=3)
        cross_cmm = np.cross(-corner_mpp[1:,:-1,:-1,:],  corner_mmm[:-1,:-1,:-1,:], axis=3)
        cross_pcm = np.cross(-corner_mmp[1:,1:,:-1,:],  -corner_mpp[1:,:-1,:-1,:],  axis=3)
        cross_cpm = np.cross( corner_mpm[:-1,1:,:-1,:], -corner_mmp[1:,1:,:-1,:],   axis=3)
        
        # cross products for top of cube (pointing out the top (+z)).
        cross_mcp = -np.cross( corner_mmp[:-1,:-1,1:,:], corner_mpp[:-1,1:,1:,:],  axis=3)
        cross_cmp = -np.cross(-corner_mpm[1:,:-1,1:,:],  corner_mmp[:-1,:-1,1:,:], axis=3)
        cross_pcp = -np.cross(-corner_mmm[1:,1:,1:,:],  -corner_mpm[1:,:-1,1:,:],  axis=3)
        cross_cpp = -np.cross( corner_mpp[:-1,1:,1:,:], -corner_mmm[1:,1:,1:,:],   axis=3)
        
        # cross products for the other edges (pointing in the right-handed direction about z).
        cross_mmc = np.cross( corner_mmp[:-1,:-1,1:,:], corner_mmm[:-1,:-1,:-1,:], axis=3)
        cross_pmc = np.cross(-corner_mpm[1:,:-1,1:,:], -corner_mpp[1:,:-1,:-1,:],  axis=3)
        cross_ppc = np.cross(-corner_mmm[1:,1:,1:,:],  -corner_mmp[1:,1:,:-1,:],   axis=3)
        cross_mpc = np.cross( corner_mpp[:-1,1:,1:,:], corner_mpm[:-1,1:,:-1,:],   axis=3)
        
        space_density = 0
        # volume contributions from bottom (-z-face-adjacent tetrahedra).
        space_density += np.abs(np.sum(-half_dk[:,:,:-1,:]*cross_mcm, axis=3)/6)
        space_density += np.abs(np.sum(-half_dk[:,:,:-1,:]*cross_cmm, axis=3)/6)
        space_density += np.abs(np.sum(-half_dk[:,:,:-1,:]*cross_pcm, axis=3)/6)
        space_density += np.abs(np.sum(-half_dk[:,:,:-1,:]*cross_cpm, axis=3)/6)
        
        # volume contributions from lower sides (tetrahedra adjacent to the last 4).
        space_density += np.abs(np.sum( half_di[:-1,:,:,:]*cross_mcm, axis=3)/6) # -x side
        space_density += np.abs(np.sum( half_dj[:,:-1,:,:]*cross_cmm, axis=3)/6) # -y side
        space_density += np.abs(np.sum(-half_di[1:,:,:,:]*cross_pcm,  axis=3)/6) # +x side
        space_density += np.abs(np.sum(-half_dj[:,1:,:,:]*cross_cpm,  axis=3)/6) # +y side
        
        # volume contributions from top (+z-face-adjacent tetrahedra).
        space_density += np.abs(np.sum( half_dk[:,:,1:,:]*cross_mcp, axis=3)/6)
        space_density += np.abs(np.sum( half_dk[:,:,1:,:]*cross_cmp, axis=3)/6)
        space_density += np.abs(np.sum( half_dk[:,:,1:,:]*cross_pcp, axis=3)/6)
        space_density += np.abs(np.sum( half_dk[:,:,1:,:]*cross_cpp, axis=3)/6)
        
        # volume contributions from upper sides (tetrahedra adjacent to the last 4).
        space_density += np.abs(np.sum( half_di[:-1,:,:,:]*cross_mcp, axis=3)/6) # -x side
        space_density += np.abs(np.sum( half_dj[:,:-1,:,:]*cross_cmp, axis=3)/6) # -y side
        space_density += np.abs(np.sum(-half_di[1:,:,:,:]*cross_pcp,  axis=3)/6) # +x side
        space_density += np.abs(np.sum(-half_dj[:,1:,:,:]*cross_cpp,  axis=3)/6) # +y side
        
        # volume contributions from side edges... couldn't think of a better description.
        space_density += np.abs(np.sum( half_di[:-1,:,:,:]*cross_mmc, axis=3)/6) # -x side
        space_density += np.abs(np.sum(-half_dj[:,:-1,:,:]*cross_mmc, axis=3)/6) # -y side
        space_density += np.abs(np.sum( half_dj[:,:-1,:,:]*cross_pmc, axis=3)/6) # -y side
        space_density += np.abs(np.sum( half_di[1:,:,:,:]*cross_pmc,  axis=3)/6) # +x side
        space_density += np.abs(np.sum(-half_di[1:,:,:,:]*cross_ppc,  axis=3)/6) # +x side
        space_density += np.abs(np.sum( half_dj[:,1:,:,:]*cross_ppc,  axis=3)/6) # +y side
        space_density += np.abs(np.sum(-half_dj[:,1:,:,:]*cross_mpc,  axis=3)/6) # +y side
        space_density += np.abs(np.sum(-half_di[:-1,:,:,:]*cross_mpc, axis=3)/6) # -x side
        
        return space_density
