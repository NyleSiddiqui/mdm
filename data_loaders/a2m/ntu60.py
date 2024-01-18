import pickle as pkl
import numpy as np
import os
from .dataset import Dataset


class NTU60Poses(Dataset):
    dataname = "ntu60"

    def __init__(self, datapath="/home/siddiqui/Action_Biometrics-RGB/frame_data/ntu_rgbd_120_skeletons/", split="train", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)
        
        self.skeleton_list = [x for x in sorted(os.listdir(datapath)) if x.endswith('.npy') and int(x[17:20]) < 11]
        print(len(self.skeleton_list))

        

        self._actions = [int(x[17:20])-1 for x in self.skeleton_list]
        self._views = [str(int(x[5:8]))+str(int(x[13:16])) for x in self.skeleton_list]
        print(np.unique(self._actions), np.unique(self._views))


        total_num_actions = 10
        total_num_views = 6
        self.num_actions = total_num_actions
        self.num_views = total_num_views
        self.num_conditions = [self.num_actions, self.num_views]

        self._train = list(range(len(self.skeleton_list)))

        keep_actions = np.arange(0, total_num_actions)
        keep_views = np.arange(0, total_num_views)


        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}
        
        self._view_to_label = {v: k for k, v in ntu_view_enumerator.items()}
        self._label_to_view = {k: v for k, v in ntu_view_enumerator.items()}
        
        
        self._action_classes = ntu_action_enumerator
        self._view_classes = ntu_view_enumerator

    def _load_joints3D(self, ind, frame_ix):
        #print(f'3D: {self._joints[ind][frame_ix].shape}')
        joints3D = np.load(os.path.join(self.datapath, self.skeleton_list[ind]), allow_pickle=True).item()['skel_body0'][frame_ix]
        #print(joints3D.shape)
        return joints3D

    def _load_rotvec(self, ind, frame_ix):
        #print(f'entered, {self._pose[ind][frame_ix].shape}')
        pose = np.load(os.path.join(self.datapath, self.skeleton_list[ind]), allow_pickle=True).item()['skel_body0'][frame_ix]
        #print(f'reshape, {pose.shape}')
        return pose


ntu_action_enumerator = {
    0: "drink water",
    1: "eat meal",
    2: "brush teeth",
    3: "brush hair",
    4: "drop",
    5: "pick up",
    6: "throw",
    7: "sit down",
    8: "stand up",
    9: "clapping",
    10: "reading",
    11: "writing",
    12: "tear up paper",
    13: "put on jacket",
    14: "take off jacket",
    15: "put on shoe",
    16: "take off shoe",
    17: "put on glasses",
    18: "take off glasses",
    19: "put on hat",
    20: "take off hat",
    21: "cheer up",
    22: "hand waving",
    23: "kicking something",
    24: "reach in pocket",
    25: "hopping",
    26: "jump up",
    27: "phone call",
    28: "play with phone",
    29: "type on keyboard",
    30: "point",
    31: "take a selfie",
    32: "check time",
    33: "rub two hands",
    34: "nod head",
    35: "shake head",
    36: "wipe face",
    37: "salute",
    38: "put palms together",
    39: "cross hands",
    40: "sneeze",
    41: "stagger",
    42: "fall down",
    43: "headache",
    44: "chest pain",
    45: "back pain",
    46: "neck pain",
    47: "vomit",
    48: "fan self",
#    49: "put on headphones",
#    50: "take off headphones",
#    51: "shoot basket",
#    52: "bounce ball",
#    53: "tennis swing",
#    54: "juggle",
#    55: "hush",
#    56: "flick hair",
#    57: "thumb up",
#    58: "thumb down",
#    59: "OK sign",
    
    
}

ntu_view_enumerator = {
    0: '11',
    1: '12',
    2: '21',
    3: '22',
    4: '31',
    5: '32',
}

