import sys
sys.path.append('.')

import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d

MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

class_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

class ScanNetObjPairScoreClsForTest(data.Dataset):

    def __init__(self, split_set='train',
        use_color=False, use_height=False, augment=False, debug=False, scene_name=None):

        self.data_path = '/home1/peisheng/3detr/scannet_data/scannet/scannet_train_detection_data'
        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))
        if split_set=='all':
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join('/home1/peisheng/3detr/scannet_data/scannet/meta_data',
                'scannetv2_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            # print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        else:
            print('illegal split name')
            return

        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment

        # the code above is for scenes, the code below is for objects
        self.object_path = '/home1/peisheng/occulusion_sim/DODA/scannet_object_complete_partial'
        # loop through all the scenes
        self.scene_object_dict = {}

        # remove scene0444_01 from the list because there is an error with camera locs
        if 'scene0444_01' in self.scan_names:
            self.scan_names.remove('scene0444_01')

        # filter self.scan_names to the one that contains the scene_name
        self.scene_name = scene_name
        self.scan_names = [sname for sname in self.scan_names if sname == self.scene_name]
        print('scan_names:', self.scan_names)

        for scan_name in self.scan_names:
            current_file_name = os.path.join(self.object_path, scan_name + '.npy')
            scene_objects = np.load(current_file_name, allow_pickle=True).item()
            self.scene_object_dict[scan_name] = scene_objects

        # scene_objects is a dictionary where the key is the index of the object in the scene, and the value is a dictionary
        # with 2 keys: 'complete' and 'partial', where 'complete' has a value of a numpy array of the indexes of points for the
        # complete object, and 'partial' has a value of a list of numpy arrays of the indexes of points for the partial objects.
        # the length of the dataset is the number of all the partial objects in all the scenes.

        # construct a mapping from dataset index to scene name and complete and partial object index
        # the mapping should be (scene name, complete object index, partial object index)
        self.index_to_scene_object = []
        for scan_name in self.scan_names:
            scene_objects = self.scene_object_dict[scan_name]
            for k, v in scene_objects.items():
                self.index_to_scene_object.append((scan_name, k))

        print('number of objects:', len(self.index_to_scene_object))

        if debug:
            self.index_to_scene_object = self.index_to_scene_object[:10] # debug

    def __getitem__(self, index):
        # for testing, we just need the complete object point cloud

        scan_name = self.index_to_scene_object[index][0]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name)+'_vert.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy')

        scan_name, complete_index = self.index_to_scene_object[index]
        complete_object_point_index = self.scene_object_dict[scan_name][complete_index]['complete']

        # index of the object is the complete object index. Get the class label of the object
        class_label = instance_bboxes[complete_index][-1]
        # convert to int
        class_label = int(class_label)
        # to tensor
        class_label = torch.tensor(class_label, dtype=torch.long)

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
        else:
            point_cloud = mesh_vertices[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0

        complete_object_point_cloud = point_cloud[complete_object_point_index]

        score = 1.
        score = torch.tensor(score, dtype=torch.float32)

        # normalize complete object point cloud
        complete_object_point_cloud[:,0:3], complete_centroid, complete_m = self.normalize_point_cloud(complete_object_point_cloud[:,0:3])

        complete_pc = self.random_sample(complete_object_point_cloud, 16384)

        return torch.from_numpy(complete_pc), torch.from_numpy(complete_pc), score, class_label

    def __len__(self):
        return len(self.index_to_scene_object) # 154751
    # 20 times more than shapenet, so should use 1/20 epoch number, and save every 1 epoch.

    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]

    def normalize_point_cloud(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc, centroid, m


if __name__ == "__main__":
    dataset = ScanNetObjPair('train')
    print(len(dataset))
    import pdb; pdb.set_trace()
    partial_pc, complete_pc = dataset[0]
    print(partial_pc.shape)
    print(complete_pc.shape)
    print(partial_pc)
    print(complete_pc)
