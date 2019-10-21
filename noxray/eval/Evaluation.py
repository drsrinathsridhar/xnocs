import argparse, sys, glob, os, json

import numpy as np
import cv2

from tk3dv.nocstools import datastructures as ds 
from EvalUtils import nocs2pc, read_nocs_map, filter_nocs_map, dict2np

class Evaluation(object):
    ''' Base class for evaluations to run. Loads GT NOCS map data.'''
    def __init__(self, args):
        self.parse_args(args)
        # load in some initial results data
        if self.nocs_view_type == 'single' or self.nocs_view_type=='multi':
            self.nocs_results = NOCSResults(self.nocs_path, self.nocs_view_type, pred_filter=self.pred_filter)

    def parse_args(self, args):
        parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

        # root of the NOCs results to load ground truth data from (nocs maps and camera pose)
        parser.add_argument('--nocs-results', required=True, help='root of the nocs results data')
        parser.add_argument('--type', help='result type (single or mutli) view.', choices=['single', 'multi'], required=True)
        parser.add_argument('--pred-filter', help='filter to apply to predicted NOCS maps. [bilateral, median]', default=None)
        parser.add_argument('--no-save-pc', dest='no_save_pc', action='store_true', help='Do not save the output point clouds for each examples')
        parser.set_defaults(no_save_pc=True)

        flags, _ = parser.parse_known_args(args)
        print(flags)

        self.nocs_path = flags.nocs_results
        self.nocs_view_type = flags.type
        self.no_save_pc = flags.no_save_pc
        self.pred_filter = flags.pred_filter

    def run(self):
        print(self.nocs_results.get_camera_pose(0))
        pred00 = self.nocs_results.get_nox00_pred(0)
        print(pred00.shape)
        print(self.nocs_results.get_nox01_pred(0).shape)
        print(self.nocs_results.get_nox00_gt(0).shape)
        print(self.nocs_results.get_nox01_gt(0).shape)
        print([x.shape for x in self.nocs_results.get_pc_pred(0)])
        print([x.shape for x in self.nocs_results.get_pc_pred(0, layer=0)])
        print([x.shape for x in self.nocs_results.get_pc_pred(0, layer=1)])
        print([x.shape for x in self.nocs_results.get_pc_pred(0, nocs_maps=pred00)])
        print([x.shape for x in self.nocs_results.get_pc_gt(0)])
        print([x.shape for x in self.nocs_results.get_pc_gt(0, layer=0)])
        print([x.shape for x in self.nocs_results.get_pc_gt(0, layer=1)])

class NOCSResults(object):
    ''' Structure to hold results from single view NOCS model. Indexes by model not by individual frames.'''
    def __init__(self, root, view_type, pred_filter=None):
        '''
        Constructs a NOCSResults object. If provided, the pred_filter (None, 'median', 'bilateral') will
        be applied to predicted NOCS maps before returned (i.e. in get_nox_*_pred_data).
        '''
        self.root = root
        self.pred_filter = pred_filter
        if pred_filter not in [None, 'median', 'bilateral']:
            print('Filter type ' + str(pred_filter) + ' not supported. No filter will be applied.')
            self.pred_filter = None
        if not os.path.exists(self.root):
            print('[NOCSSingleResults.__init__] Could not find results data at given path!')
            return

        # read in metadata list of models
        meta_path = os.path.join(self.root, 'model_info.txt')
        self.meta_info = []
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as meta_file:
                self.meta_info = meta_file.read().split('\n')
                if self.meta_info[-1] == '': # check for trailing new line
                    self.meta_info = self.meta_info[:-1]
            if len(self.meta_info) == 0:
                print('[NOCSSingleResults.__init__] Could not find meta info!')
            self.meta_info = [info.split('/') for info in self.meta_info]
            self.meta_info = [(info[0], info[1], int(info[2])) for info in self.meta_info]
        else:
            print('No meta info file! Cannot evaluate!')
            return

        # collect indices from the same model (assuming file in order)
        self.model_map = [] # maps model number to indices in data arrays
        cur_model = self.meta_info[0][1]
        cur_model_list = []
        for i, info in enumerate(self.meta_info):
            model = info[1]
            if i > 0 and model != cur_model:
                # push the one we were collecting
                self.model_map.append(cur_model_list)
                # setup new
                cur_model_list = [i]
                cur_model = model
            else:
                cur_model_list.append(i)
        self.model_map.append(cur_model_list)

        # the number of models
        self.length = len(self.model_map)

        # read in camera pos/rot data
        poses = sorted(glob.glob(self.root + '/*_pose.json'))
        self.cam_pos = np.zeros((len(poses), 3))
        self.cam_rot = np.zeros((len(poses), 4))
        for i, pose_path in enumerate(poses):
            with open(pose_path, 'r') as pose_file:
                cur_pose = json.load(pose_file)
            if len(cur_pose) == 0:
                print('Could not read pose info for' + pose_path)
            self.cam_pos[i] = dict2np(cur_pose['position'])
            self.cam_rot[i] = dict2np(cur_pose['rotation'])

        # everything else we will store a path to and read in lazily as needed
        # needs to be sorted in frame and then view order to match model info!
        if view_type == 'single':
            sort_by_frame = lambda file_path: int(file_path.split('/')[-1].split('_')[1])
        elif view_type == 'multi':
            sort_by_frame = lambda file_path: (int(file_path.split('/')[-1].split('_')[1]), int(file_path.split('/')[-1].split('_')[3]))
        self.color00 = sorted(glob.glob(self.root + '/*_color00.png'), key=sort_by_frame)
        self.color01_gt = sorted(glob.glob(self.root + '/*_color01_gt.png'), key=sort_by_frame)
        self.mask00_gt = sorted(glob.glob(self.root + '/*_mask00_gt.png'), key=sort_by_frame)
        self.mask01_gt = sorted(glob.glob(self.root + '/*_mask01_gt.png'), key=sort_by_frame)
        self.mask00_pred = sorted(glob.glob(self.root + '/*_mask00_pred.png'), key=sort_by_frame)
        self.mask01_pred = sorted(glob.glob(self.root + '/*_mask01_pred.png'), key=sort_by_frame)
        self.nox00_gt = sorted(glob.glob(self.root + '/*_nox00_gt.png'), key=sort_by_frame)
        self.nox01_gt = sorted(glob.glob(self.root + '/*_nox01_gt.png'), key=sort_by_frame)
        self.nox00_pred = sorted(glob.glob(self.root + '/*_nox00_pred.png'), key=sort_by_frame)
        self.nox01_pred = sorted(glob.glob(self.root + '/*_nox01_pred.png'), key=sort_by_frame)

        # frame_ordering = [int(f.split('/')[-1].split('_')[1]) for f in self.color00]
        # print(frame_ordering)

        #print(self.nox00_gt)
        #print(self.nox00_pred)

    def get_model_info(self, idx):
        return [self.meta_info[i] for i in self.model_map[idx]]
    
    def get_camera_pose(self, idx):
        ''' returns camera poses for all frames of model at idx '''
        data_inds = self.model_map[idx]
        poses = [self.get_camera_pose_data(data_idx) for data_idx in data_inds]
        return poses

    def get_camera_pose_data(self, idx):
        ''' returns tuple (position, quaternion) for single data idx '''
        if idx < self.cam_pos.shape[0]:
            return (self.cam_pos[idx], self.cam_rot[idx])
        else:
            return None

    def get_nox00_pred(self, idx):
        data_inds = self.model_map[idx]
        nox00_pred_list = [self.get_nox00_pred_data(i) for i in data_inds]
        return np.stack(nox00_pred_list, axis=0)
    
    def get_nox00_pred_data(self, idx):
        if idx >= len(self.nox00_pred):
            return None
        out_nocs_map = read_nocs_map(self.nox00_pred[idx])
        if self.pred_filter is not None:
            out_nocs_map = filter_nocs_map(out_nocs_map, self.pred_filter)
        return out_nocs_map

    def get_nox01_pred(self, idx):
        data_inds = self.model_map[idx]
        nox01_pred_list = [self.get_nox01_pred_data(i) for i in data_inds]
        return np.stack(nox01_pred_list, axis=0)

    def get_nox01_pred_data(self, idx):
        if idx >= len(self.nox01_pred):
            return None
        out_nocs_map = read_nocs_map(self.nox01_pred[idx])
        if self.pred_filter is not None:
            out_nocs_map = filter_nocs_map(out_nocs_map, self.pred_filter)
        return out_nocs_map

    def get_nox00_gt(self, idx):
        data_inds = self.model_map[idx]
        nox00_gt_list = [self.get_nox00_gt_data(i) for i in data_inds]
        return np.stack(nox00_gt_list, axis=0)

    def get_nox00_gt_data(self, idx):
        if idx >= len(self.nox00_gt):
            return None
        return read_nocs_map(self.nox00_gt[idx])

    def get_nox01_gt(self, idx):
        data_inds = self.model_map[idx]
        nox01_gt_list = [self.get_nox01_gt_data(i) for i in data_inds]
        return np.stack(nox01_gt_list, axis=0)

    def get_nox01_gt_data(self, idx):
        if idx >= len(self.nox01_gt):
            return None
        return read_nocs_map(self.nox01_gt[idx])

    def get_pc_pred(self, idx, layer=None, nocs_maps=None):
        ''' 
        Returns predicted point clouds for the model at idx.
        Can optionally pass in the predicted nocs maps which will be used
        to build the point clouds (for improved efficiency). 
        '''
        if nocs_maps is not None:
            frame_inds = range(nocs_maps.shape[0])
            pc_pred_list = [self.get_pc_pred_data(-1, layer, nocs_map=nocs_maps[i]) for i in frame_inds]
        else:
            data_inds = self.model_map[idx]
            pc_pred_list = [self.get_pc_pred_data(i, layer) for i in data_inds]
        return pc_pred_list
        
    def get_pc_pred_data(self, idx, layer=None, nocs_map=None):
        if nocs_map is not None:
            return nocs2pc([nocs_map])

        nocs_map_list = []
        if layer == None or layer == 0:
            in_nocs_map = read_nocs_map(self.nox00_pred[idx])
            if self.pred_filter is not None:
                in_nocs_map = filter_nocs_map(in_nocs_map, self.pred_filter)
            nocs_map_list.append(in_nocs_map)
        if layer == None or layer == 1:
            in_nocs_map = read_nocs_map(self.nox01_pred[idx])
            if self.pred_filter is not None:
                in_nocs_map = filter_nocs_map(in_nocs_map, self.pred_filter)
            nocs_map_list.append(in_nocs_map)
        return nocs2pc(nocs_map_list)

    def get_pc_gt(self, idx, layer=None):
        data_inds = self.model_map[idx]
        pc_gt_list = [self.get_pc_gt_data(i, layer) for i in data_inds]
        return pc_gt_list

    def get_pc_gt_data(self, idx, layer=None):
        nocs_map_list = []
        nocs_map0 = read_nocs_map(self.nox00_gt[idx])
        if layer == None or layer == 0:
            nocs_map_list.append(nocs_map0)
        nocs_map1 = read_nocs_map(self.nox01_gt[idx])
        if layer == None or layer == 1:
            nocs_map_list.append(nocs_map1)
        return nocs2pc(nocs_map_list)

class NOCSMultiResults(object):
    ''' Structure to hold results from multi-view NOCS model '''
    def __init__(self, root):
        self.root = root
        

if __name__=='__main__':
    evaluation = Evaluation(sys.argv[1:])
    evaluation.run()
