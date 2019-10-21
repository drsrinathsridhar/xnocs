import argparse, sys, os, random, csv

import numpy as np
import scipy.io
from transforms3d import quaternions

from NOCSEval import NOCSEvaluation
from chamfer_metric import ChamferMetric
from EvalUtils import prune_pc, viz_pcs, normalize_pc, open3d_icp, save_pc_multi

class R2N2Evaluation(NOCSEvaluation):
    ''' Evaluates R2N2's output for point cloud. '''
    def __init__(self, args):
        super().__init__(args)
        self.r2n2_results = R2N2Results(self.r2n2_path, limit=self.data_limit, all=self.using_all)

    def parse_args(self, args):
        super().parse_args(args)

        parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        
        parser.add_argument('--r2n2-results', required=True, help='root of the r2m2 results data (not this must be an individual category directory')
        parser.add_argument('--limit', type=int, default=None)
        parser.add_argument('--all', dest='all', action='store_true', help='Using ll data for evaluation')
        parser.set_defaults(all=False)
        flags, _ = parser.parse_known_args(args)

        self.r2n2_path = flags.r2n2_results
        self.data_limit = flags.limit
        self.using_all = flags.all
    
    def run(self):
        ''' Run through each model and evaluate point cloud prediction. '''

        # R2N2 models are already aligned
        # uncomment to viz result point clouds
        self.viz_results(30)

        # now do the actual evaluation
        # R2n2 only evaluated a fixed N number of views at a time and outputs a single point cloud
        multi_view_err = []
        chamfer = ChamferMetric(device='cuda')

        for i in range(self.r2n2_results.length):
            r2n2_id = self.r2n2_results.meta_info[i]
            nocs_id = self.nocs_results.get_model_info(i)[0][1]
            if r2n2_id != nocs_id:
                print('ERROR EVALUATION DATA IS NOT CONSISTENT BETWEEN R2N2 AND NOCS!!')
                quit()
            if not self.no_pc_eval:
                # now we evaluate the estimated point cloud
                # get GT point cloud (union of all available views)
                gt_pc_frames = self.nocs_results.get_pc_gt(i)
                gt_pc = np.concatenate(gt_pc_frames, axis=0)
                if gt_pc.shape[0] > self.max_pts:
                    gt_pc = prune_pc(gt_pc, self.max_pts)
                gt_pc -= np.mean(gt_pc, axis=0)

                # for multi-view we only do a single evaluation on the point cloud
                # from all views combined
                multi_pc = self.r2n2_results.pcs[i]
                if multi_pc.shape[0] > self.max_pts:
                    multi_pc = prune_pc(multi_pc, self.max_pts)
                # print(multi_pc.shape)
                cur_err = chamfer.chamfer_dist(multi_pc, gt_pc)
                multi_view_err.append(cur_err)
                if not self.no_save_pc:
                    cur_inf = self.nocs_results.get_model_info(i)[0]
                    save_pc_multi(multi_pc, self.pc_out_path, cur_inf)

            if i % 20 == 0:
                self.log('Finished evaluating model ' + str(i) + '...')

        with open(os.path.join(self.out_path, 'multi_view_results.csv'), 'w') as res_out:
            err_writer = csv.writer(res_out, delimiter=',')
            err_writer.writerow(['cat', 'model', 'chamfer'])
            for i, err in enumerate(multi_view_err):
                info = self.nocs_results.get_model_info(i)
                cat, model_id = info[0][0:2]
                err_writer.writerow([cat, model_id, err])
        self.log('Mean multi view err: %06f' % (np.mean(np.array(multi_view_err))))


    def viz_results(self, num_models):
        ''' Save visualizations of the first num_models point clouds '''
        for i in range(num_models):
            # same GT for every point cloud in this model
            gt_pc_frames = self.nocs_results.get_pc_gt(i)
            gt_pc = np.concatenate(gt_pc_frames, axis=0)
            if gt_pc.shape[0] > self.max_pts:
                gt_pc = prune_pc(gt_pc, self.max_pts)
            gt_pc -= np.mean(gt_pc, axis=0)

            viz_pcs([gt_pc], os.path.join(self.out_path, 'gt_pc' + str(i) + '.png'))

            pred_pc = self.r2n2_results.pcs[i]
            viz_pcs([pred_pc], os.path.join(self.out_path, 'pred_pc' + str(i) + '.png'))

class R2N2Results(object):
    ''' Structure to hold results from R2N2 model. Indexed by model not by individual frames. '''
    def __init__(self, root, all=False, limit=None):
        self.root = root
        if not os.path.exists(self.root):
            print('Could not find R2N2 results data at given path!')
            return

        self.meta_info = [] # model id
        self.pcs = [] # point clouds (5, N, 3)

        voxel_width = 1.0 / 32

        # all datat is stored in npz files, just read in and store in memory
        cats = [self.root]
        if all:
            cats = sorted([os.path.join(self.root, f) for f in os.listdir(self.root)])
        for cat in cats:
            all_results = sorted([os.path.join(cat, f) for f in os.listdir(cat) if f.split('.')[-1] == 'npz'])
            for cnt, npz_path in enumerate(all_results):
                if limit is not None and cnt >= limit:
                    break
                npz_arr = np.load(npz_path)
                full_voxel = npz_arr['full_voxel_grid']
                surf_voxel = npz_arr['surface_voxel_grid']
                # create point cloud from surface voxels
                pt_inds = np.nonzero(surf_voxel > 0.5)
                idx_arr = np.stack(pt_inds, axis=1)
                pc = idx_arr*voxel_width
                pc -= np.mean(pc, axis=0)
                if pc.shape[0] == 0:
                    # dummpy point cloud
                    pc = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
                    print('Empty point cloud...creating dummy.')
                pc = normalize_pc(pc)
                self.pcs.append(pc)
                self.meta_info.append(npz_path.split('/')[-1].split('.')[0])
        
        self.length = len(self.meta_info)

if __name__=='__main__':
    evaluation = R2N2Evaluation(sys.argv[1:])
    evaluation.run()
