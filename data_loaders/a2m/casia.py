import pickle as pkl
import numpy as np
import pandas as pd
import os
from .dataset import Dataset


class CasiaPoses(Dataset):

    def __init__(self, datapath="/home/c3-0/globus/praveen/CASIA-B/casia-b_pose_train_valid.csv", split="train", **kargs):
        self.dataname = "casia"
        ### ALL 'action' VARAIBLES ARE SUBJECTS ###
        pd.set_option('display.max_columns', None)
        self.datapath = datapath
        #print(datapath)
        super().__init__(**kargs)
        self.video_list = [x for x in sorted(os.listdir('/squash/CASIA-B-lz4/OpenPose/DatasetB-1/')) if 'bkgrd' not in x]
        print(len(self.video_list))
        self.skeleton_csv = pd.read_csv(datapath)
        #print(self.skeleton_csv.head())
        

        
        #print(f'lens: {len(np.unique(self.skeleton_csv["image_name"][:-11]))}, {len(self.video_list)}, {np.unique(self.skeleton_csv["image_name"][-24:-11])}')

        self._actions = [int(x[:3])-1 for x in self.video_list]
        #print(self.skeleton_list[0])
        print(self._actions[0])
        print(len(self._actions))
        #print(self._actions)

        total_num_actions = 62
        self.num_actions = total_num_actions

        self._train = list(range(len(self.video_list)))

        keep_actions = np.arange(0, total_num_actions)
        #print(keep_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanact12_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        video = self.video_list[ind]
        filtered_df = self.skeleton_csv[self.skeleton_csv['image_name'].str.contains(f'{video}')]
        joints3D = filtered_df.iloc[frame_ix].values[:, 1:].reshape(-1, 17, 3)[:, :, :-1].astype(float)
        z = np.ones([60, 17, 1])
        joints3D = np.append(joints3D, z, axis=2)
        return joints3D

    def _load_rotvec(self, ind, frame_ix):
        video = self.video_list[ind]
        filtered_df = self.skeleton_csv[self.skeleton_csv['image_name'].str.contains(f'{video}')]
        pose = filtered_df.iloc[frame_ix].values[:, 1:].reshape(-1, 17, 3)[:, :, :-1].astype(float)
        print(f'pose: {pose}, {pose.shape}, {pose.dtype}')
        return pose


humanact12_coarse_action_enumerator = {}

for i in range(62):
    humanact12_coarse_action_enumerator[i] = f'sub{i+1}'
    
    
if __name__ == '__main__':
    loader = DataLoader(CasiaPoses, batch_size=1, shuffle=True, num_workers=8, drop_last=True)
    
    