import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

in_depth_path = '1a54a2319e87bd4071d03b466c72ce41_0_0_0.npy'

def single_depth_2_pc(in_depth_path):
    depth = np.load(in_depth_path)
    depth = np.asarray(depth, dtype=np.float32)
    plt.imshow(depth)
    plt.show()

    h = depth.shape[0]
    w = depth.shape[1]

    fov = 49.124/2  # degree
    fx = w/(2.0*np.tan(fov/180.0*np.pi))
    fy = h/(2.0*np.tan(fov/180.0*np.pi))
    k = np.array([[fx, 0, w/2],
                  [0, fy, h/2],
                  [0, 0, 1]], dtype=np.float32)

    xyz_pc = []
    for hi in range(h):
        for wi in range(w):
            if depth[hi, wi]>5 or depth[hi, wi]==0.0:
                depth[hi, wi] =0.0
                continue
            x = -(wi - w/2)*depth[hi, wi]/fx
            y = -(hi - h/2)*depth[hi, wi]/fy
            z = depth[hi, wi]
            xyz_pc.append([x, y, z])

    print ("pc num:", len(xyz_pc))
    xyz_pc = np.asarray(xyz_pc, dtype=np.float16)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xyz_pc[:,0], xyz_pc[:,1], xyz_pc[:,2], c='r',marker='o')
    plt.show()

    return xyz_pc

def voxelization(pc_25d):
    vox_res = 256

    x_max = max(pc_25d[:, 0]); x_min = min(pc_25d[:, 0])
    y_max = max(pc_25d[:, 1]); y_min = min(pc_25d[:, 1])
    z_max = max(pc_25d[:, 2]); z_min = min(pc_25d[:, 2])
    step = round(max([x_max - x_min, y_max - y_min, z_max - z_min]) / (vox_res - 1), 4)
    x_d_s = int((x_max - x_min) / step)
    y_d_s = int((y_max - y_min) / step)
    z_d_s = int((z_max - z_min) / step)

    vox = np.zeros((x_d_s+1, y_d_s+1, z_d_s+1, 1), dtype=np.int8)
    for k, p in enumerate(pc_25d):
        if k % 50000 == 0:
            print (k)
        ##### voxlization 25d
        xd = int((p[0] - x_min) / step)
        yd = int((p[1] - y_min) / step)
        zd = int((p[2] - z_min) / step)
        if xd >= vox_res or yd >= vox_res or zd >= vox_res:
            print ("xd>=vox_res or yd>=vox_res or zd>=vox_res")
            continue
        if xd > x_d_s or yd > y_d_s or zd > z_d_s:
            print ("xd>x_d_s or yd>y_d_s or zd>z_d_s")
            continue

        vox[xd, yd, zd, 0] = 1
    ####
    plotFromVoxels(vox)

    save = False
    if save:
        np.savez_compressed('1a54a2319e87bd4071d03b466c72ce41_0_0_0.npz', vox_25d)

    return vox

def plotFromVoxels(voxels):
    if len(voxels.shape) > 3:
        x_d = voxels.shape[0]
        y_d = voxels.shape[1]
        z_d = voxels.shape[2]
        v = voxels[:, :, :, 0]
        v = np.reshape(v, (x_d, y_d, z_d))
    else:
        v = voxels
    x, y, z = v.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')
    plt.show()

#########################
if __name__ == '__main__':
    pc_25d = single_depth_2_pc(in_depth_path)
    vox_25d = voxelization(pc_25d)

