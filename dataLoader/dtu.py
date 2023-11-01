
import torch
import cv2 as cv
import numpy as np
import os
from glob import glob
from .ray_utils import *
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation



# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def focus_pt_fn(poses):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def draw_render_path(poses, avg_poses, render_poses):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = render_poses[:, :3, 3]
    
    ax.scatter(data[:,0], data[:,1], data[:,2])
    ax.scatter(poses[:, :3, 3][:,0], poses[:, :3, 3][:,1], poses[:, :3, 3][:,2])
    ax.scatter(avg_poses[:3, 3][0], avg_poses[:3, 3][1], avg_poses[:3, 3][2])
    ax.view_init(elev=30, azim=0)
    plt.savefig("render_path.png")
    ani = FuncAnimation(fig, lambda i: ax.view_init(elev=30, azim=i), frames=np.arange(0, 360, 1), interval=10)
    ani.save('scan114.gif', writer='imagemagick', fps=30)  
    # breakpoint()

def generate_spiral_path_dtu(poses, n_frames=120, n_rots=4, zrate=1, perc=60):
    """Calculates a forward facing spiral path for rendering for DTU."""

    # Get radii for spiral path using 60th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), perc, 0) * 1
    radii = np.concatenate([radii, [1.]]) 

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    z_axis = focus_pt_fn(poses)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        render_poses.append(viewmatrix(z_axis, up, position, True))
    render_poses = np.stack(render_poses, axis=0)
    # Draw render_path 
    # draw_render_path(poses, cam2world, render_poses)
    return render_poses

def rescale_poses(poses):
    """Rescales camera poses according to maximum x/y/z value."""
    s = np.max(np.abs(poses[:, :3, -1]))
    out = np.copy(poses)
    out[:, :3, -1] /= s
    return out

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
    return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)

def recenter_poses(poses):
    """Recenter poses around the origin."""
    # breakpoint()
    trans = np.array([[-1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
    # trans[1, 1] = 1
    # trans[2, 2] = 1
    # trans[0, 3] = 1
    # rotation = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    # homogeneous_rotation_matrix = np.eye(4)
    # homogeneous_rotation_matrix[:3, :3] = rotation
    # poses = poses @ homogeneous_rotation_matrix
    cam2world = poses_avg(poses)
    poses = np.linalg.inv(pad_poses(cam2world)) @ poses
    return poses


class DTUDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1, is_stack=False, frame_num=None, hold_every=8):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.split = split
        self.root_dir = datadir
        self.is_stack = is_stack
        self.downsample = downsample
        self.white_bg = True
        self.camera_dict = np.load(os.path.join(self.root_dir, 'cameras.npz'))

        self.img_wh = (int(400 / downsample), int(300 / downsample))
        self.hold_every = hold_every
        
        self.near_far = np.array([0.0, 0.5])
        self.frame_num = frame_num
        # self.frame_len = len(frame_num)
        self.scene_name = datadir.split("/")[-1]
        self.dataset_name = datadir.split("/")[-2]
        # self.scan = os.path.basename(datadir)
        # self.split = split
        #
        # self.img_wh = (int(640 / downsample), int(512 / downsample))
        # self.downsample = downsample
        #
        # self.scale_factor = 1.0 / 200
        # self.define_transforms()

        # self.scene_bbox = np.array([[-1.01, -1.01, -1.01], [1.01,  1.01,  1.01]])
        # self.near_far = np.array([2.125, 4.525])*200/self.scale_mats_np[0,0,0]
        #
        # self.re_centerMat = np.array([[0.311619, -0.853452, 0.417749, -1.4379079],
        #                               [0.0270351, 0.44742498, 0.893913, -2.801856],
        #                               [-0.949823, -0.267266, 0.162499, -0.35806254],
        #                               [0., 0., 0., 1.]])
        self.read_meta()
        self.get_bbox()

    # def define_transforms(self):
    #     self.transform = T.ToTensor()

    def get_bbox(self):
        object_bbox_min = np.array([-1.0, -1.0, -1.0, 1.0])
        object_bbox_max = np.array([ 1.0,  1.0,  1.0, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.root_dir, 'cameras.npz'))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.scene_bbox = torch.from_numpy(np.stack((object_bbox_min[:3, 0],object_bbox_max[:3, 0]))).float()
        # self.near_far = [2.125, 4.525]

    def gen_rays_at(self, intrinsic, c2w, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        W,H = self.img_wh
        tx = torch.linspace(0, W - 1, W // l)+0.5
        ty = torch.linspace(0, H - 1, H // l)+0.5
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        intrinsic_inv = torch.inverse(intrinsic)
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(c2w[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = c2w[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1).reshape(-1,3), rays_v.transpose(0, 1).reshape(-1,3)

    def pre_calculate_nearest_pose(self, img_list):
        num_camera_pose = len(img_list)
        
        nearest_dist = np.full(len(self.poses), np.inf) # index; input_pose_index, output: its nearest_pose_index
        nearest_pose = np.full(len(self.poses), -1)
        
        dist = 0
        cur, next = -1, -1
        for i in range(num_camera_pose - 1):
            cur = img_list[i]
            for j in range(i + 1, num_camera_pose):
                next = img_list[j]
                dist = np.linalg.norm(self.poses[cur][:3, 3] - self.poses[next][:3, 3])
                if dist < nearest_dist[cur]:
                    nearest_dist[cur] = dist
                    nearest_pose[cur] = next
                if dist < nearest_dist[next]:
                    nearest_dist[next] = dist
                    nearest_pose[next] = cur
        return nearest_pose
    
    def get_nearest_pose(self, c2w, img_list, i):
        # calculate neighbor poses
        min_distance = -1
        for j in img_list:
            if j == i:
                continue
            
            distance = (torch.sum(((c2w[:3,3] - self.poses[j,:3,3])**2)))**0.5
            if min_distance == -1 or distance < min_distance:
                min_distance = distance
                nearest_id = j
        return nearest_id

    def read_meta(self):
        
        images_lis = sorted(glob(os.path.join(self.root_dir, 'image/*.png')))
        images_np = np.stack([cv.resize(cv.imread(im_name),self.img_wh) for im_name in images_lis]) / 255.0
        # masks_lis = sorted(glob(os.path.join(self.root_dir, 'mask/*.png')))
        # masks_np = np.stack([cv.resize(cv.imread(im_name),self.img_wh) for im_name in masks_lis])>128

        rgbs = torch.from_numpy(images_np.astype(np.float32)[...,[2,1,0]])  # [n_images, H, W, 3]
        # self.all_masks  = torch.from_numpy(masks_np>0)   # [n_images, H, W, 3]
        self.img_wh = [rgbs.shape[2],rgbs.shape[1]]
        W,H = self.img_wh

        # world_mat is a projection matrix from world to image
        n_images = len(images_lis) 
        world_mats_np = [self.camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        self.scale_mats_np = [self.camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        
        # load intrinsics & poses from all imgs
        self.intrinsics, self.poses = [],[]
        for img_idx, (scale_mat, world_mat) in enumerate(zip(self.scale_mats_np, world_mats_np)):
            print(img_idx)
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic, c2w = load_K_Rt_from_P(None, P)
            # c2w = torch.from_numpy(c2w).float()
            # intrinsic = torch.from_numpy(intrinsic).float()
            intrinsic[:2] /= self.downsample
            
            self.poses.append(c2w)
            self.intrinsics.append(intrinsic)
        self.intrinsics, self.poses = np.stack(self.intrinsics), np.stack(self.poses)
        
        # self.poses = recenter_poses(self.poses)
        self.poses = rescale_poses(self.poses)
        
        # import matplotlib.pyplot as plt
        # from matplotlib.animation import FuncAnimation
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(0, 0, 0, c='orange')
        # ax.scatter(self.poses[:, :3, 3][:,0], self.poses[:, :3, 3][:,1], self.poses[:, :3, 3][:,2], c='r')
        # ax.quiver(self.poses[:, :3, 3][:,0], self.poses[:, :3, 3][:,1], self.poses[:, :3, 3][:,2], self.poses[:, :3, 2][:, 0], self.poses[:, :3, 2][:, 1], self.poses[:, :3, 2][:, 2], color='r', length=1,
        #             arrow_length_ratio=0.1, alpha=0.1, zorder=1)

        # self.poses = recenter_poses(self.poses)
        # ax.scatter(self.poses[:, :3, 3][:,0], self.poses[:, :3, 3][:,1], self.poses[:, :3, 3][:,2], c='g')
        # ax.quiver(self.poses[:, :3, 3][:,0], self.poses[:, :3, 3][:,1], self.poses[:, :3, 3][:,2], self.poses[:, :3, 2][:, 0], self.poses[:, :3, 2][:, 1], self.poses[:, :3, 2][:, 2], color='g', length=1,
        #             arrow_length_ratio=0.1, alpha=0.1, zorder=1)
        
        # self.poses = rescale_poses(self.poses)
        # ax.scatter(self.poses[:, :3, 3][:,0], self.poses[:, :3, 3][:,1], self.poses[:, :3, 3][:,2], c='b')
        # ax.quiver(self.poses[:, :3, 3][:,0], self.poses[:, :3, 3][:,1], self.poses[:, :3, 3][:,2], self.poses[:, :3, 2][:, 0], self.poses[:, :3, 2][:, 1], self.poses[:, :3, 2][:, 2], color='b', length=1,
        #             arrow_length_ratio=0.1, alpha=0.1, zorder=1)
        # ani = FuncAnimation(fig, lambda i: ax.view_init(elev=30, azim=i), frames=np.arange(0, 360, 1), interval=10)
        # ani.save('scan103.gif', writer='imagemagick', fps=30)  
        # breakpoint()
        
        # load img list 
        if self.frame_num is not None and len(self.frame_num) > 0:
            img_list = self.frame_num
            self.frame_len = len(img_list)
        else:
            i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
            img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))
            self.frame_len = len(img_list)
        
        # build rendering path
        N_views = 60
        #### 2 view spiral path smapling
        print(self.split)
        if self.frame_num is not None and len(self.frame_num) > 0:
            if self.split == 'train' or 'novel':
                self.render_path = generate_spiral_path_dtu(self.poses[:, :3, :][self.frame_num], n_frames=N_views)
            elif self.split == 'test':
                self.render_path = generate_spiral_path_dtu(self.poses[:, :3, :][self.frame_num], n_frames=N_views, n_rots=1, zrate=0)
        else:
            self.render_path = generate_spiral_path_dtu(self.poses[:, :3, :], n_frames=N_views, n_rots=1, zrate=0)
        #### All view spiral path smapling
        # self.render_path = generate_spiral_path_dtu(self.poses[:, :3, :].numpy(), n_frames=N_views)
        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.poses = torch.from_numpy(self.poses).float()
        
        self.all_rays = []
        self.all_rgbs = []
        # self.all_ids = []
        # self.all_nearest_ids = []
        # self.all_depths = []
        # self.all_depth_weights = []
        if self.split != 'novel':
            # self.frameid2_startpoints_in_allray = [-10] * self.poses.shape[0] # -10 represent None
            # cnt = 0
            for i in img_list:
                c2w = self.poses[i]
                intrinsic = self.intrinsics[i]

                # center = intrinsic[:2,-1]
                # directions = get_ray_directions(H, W, [intrinsic[0,0], intrinsic[1,1]], center=center)  # (h, w, 3)
                # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
                # rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
                rays_o, rays_d = self.gen_rays_at(intrinsic,c2w)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                self.all_rgbs += [rgbs[i].reshape(-1, 3)]
                
                
                # # get sparse depth from csv (generated by ViP-NeRF saprse depth genrator)
                # depth = -torch.ones(H, W)
                # if self.split == "train":
                #     SD = load_sparse_depth(self.dataset_name, self.scene_name, self.frame_len, i, int(self.downsample))
                #     for j in range(len(SD)):
                #         depth[round(SD.y[j]), round(SD.x[j])] = SD.depth[j]
                #     depth = depth.view(-1)
                # self.all_depths += [depth]
                # # get nearest frame of current frame
                # nearest_id = self.get_nearest_pose(c2w, img_list, i)
                # cur_ids = torch.full([rays_o.shape[0]], i)
                # self.all_ids += [cur_ids]
                # self.all_nearest_ids += [torch.ones_like(cur_ids).int() * nearest_id]
                # self.frameid2_startpoints_in_allray[i] = cnt * cur_ids.shape[0] - 1
                # cnt += 1 
        # change poses shape from (n, 4, 4) to (n, 3, 4)
        self.poses = self.poses[:, :3, :]
        
        if self.split == 'novel':
            cnt = 0
            self.frameid2_startpoints_in_allray = [-10] * self.render_path.shape[0]
            for i,  c2w in enumerate(self.render_path):
                c2w = torch.FloatTensor(c2w)
                intrinsic = self.intrinsics[0]
                # get rays
                rays_o, rays_d = self.gen_rays_at(intrinsic,c2w)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                # get nearest training frame of current frame
                cur_ids = torch.full([rays_o.shape[0]], i)
                self.all_ids += [cur_ids]
                nearest_id = self.get_nearest_pose(c2w, img_list, i)
                self.all_nearest_ids += [torch.ones_like(cur_ids).int() * nearest_id]
                self.frameid2_startpoints_in_allray[i] = cnt * cur_ids.shape[0] - 1
                cnt += 1
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_ids = torch.cat(self.all_ids, 0).to(torch.int)
            self.all_nearest_ids = torch.cat(self.all_nearest_ids, 0).to(torch.int)
        else:
            if not self.is_stack:
                self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)
                # self.all_depths = torch.cat(self.all_depths, 0)
                # self.all_ids = torch.cat(self.all_ids, 0).to(torch.int)
                # self.all_nearest_ids = torch.cat(self.all_nearest_ids, 0).to(torch.int)
            else:
                self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
                # self.all_depths = torch.stack(self.all_depths, 0).reshape(-1,*self.img_wh[::-1], 1)
                # self.all_ids = torch.stack(self.all_ids, 0).to(torch.int)
                # self.all_nearest_ids = torch.stack(self.all_nearest_ids, 0).to(torch.int)
        
        f = (self.intrinsics[0][0, 0] + self.intrinsics[0][1, 1]) / 2
        self.focal = [f, f]
        self.center = [self.intrinsics[0][0, 2], self.intrinsics[0][1, 2]]
        # ray directions for all pixels, same for all images (same H, W, focal)
        # self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)
        # self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        
        print(f'{self.split} dataLoader loaded', len(self.all_rays), 'rays')
    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        else:
            sample = {'rays': self.all_rays[idx]}
        return sample