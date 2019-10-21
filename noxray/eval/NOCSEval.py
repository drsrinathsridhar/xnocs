import argparse, sys, os, random, csv

import numpy as np
import cv2
from transforms3d import quaternions
from transforms3d.axangles import axangle2mat

from tk3dv.nocstools import datastructures as ds
from tk3dv.nocstools import calibration

from Evaluation import Evaluation
from chamfer_metric import ChamferMetric
from EvalUtils import prune_pc, save_pc_single, save_pc_multi, write_nocs_map
from EvalUtils import quaternion_from_campos, quaternion_multiply, quaternion_conjugate

NUM_CAM_SOLVE_PTS = 100
MAX_PC_PTS_DEFAULT = 8000

class NOCSEvaluation(Evaluation):
    ''' Evaluates our model's output for point cloud and camera pose. '''
    def __init__(self, args):
        super().__init__(args)
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        with open(os.path.join(self.out_path, 'log.txt'), 'w') as f:
            f.write('Eval log:\n')
        if not self.no_save_pc:
            self.pc_out_path = os.path.join(self.out_path, 'pcs')
            if not os.path.exists(self.pc_out_path):
                os.mkdir(self.pc_out_path)

    def parse_args(self, args):
        super().parse_args(args)
        parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        
        parser.add_argument('--out', required=True, help='root of the output directory to save results')
        parser.add_argument('--no-camera', dest='no_camera', action='store_true')
        parser.set_defaults(no_camera=False)
        parser.add_argument('--dpc-camera', dest='dpc_camera', action='store_true')
        parser.set_defaults(dpc_camera=False)
        parser.add_argument('--no-pc', dest='no_pc', action='store_true')
        parser.set_defaults(no_pc=False)
        parser.add_argument('--save-filtered', dest='save_filtered', action='store_true', help='If given along with a pred-filter, then filtered NOCS maps will be saved in nocs-path')
        parser.set_defaults(save_filtered=False)
        parser.add_argument('--max-points', type=int, default=MAX_PC_PTS_DEFAULT, help='Maximum number of allowed points in any ground truth or predicted point cloud. If over this number will randoly downsample.')
        if self.nocs_view_type == 'single':
            # need to know how many views to use in evaluation (can use multiple numbers)
            parser.add_argument('--nviews', type=int, nargs='+', help='Number of views to evaluate for aggregated single view evaluation (1 view is always performed).')

        flags, _ = parser.parse_known_args(args)

        self.out_path = flags.out
        self.max_pts = flags.max_points
        self.no_camera_eval = flags.no_camera
        self.dpc_camera_eval = flags.dpc_camera
        self.no_pc_eval = flags.no_pc
        self.save_filtered = flags.save_filtered
        if self.nocs_view_type == 'single':
            self.num_views = flags.nviews
            if self.num_views is None:
                self.num_views = []
    
    def run(self):
        ''' Run through each model and evaluate point cloud prediction and camera pose. '''
        cam_pos_err = []
        cam_rot_err = []
        dpc_rot_err = []
        if self.nocs_view_type == 'single':
            single_view_err = []
            single_nview_errs = []
            for n in self.num_views:
                single_nview_errs.append([])
        elif self.nocs_view_type == 'multi':
            multi_view_err = []

        for i in range(self.nocs_results.length):
            # will need these for both
            nocs00_maps = self.nocs_results.get_nox00_pred(i)
            nocs01_maps = self.nocs_results.get_nox01_pred(i)
            nocs00_pcs = self.nocs_results.get_pc_pred(-1, nocs_maps=nocs00_maps)
            nocs01_pcs = self.nocs_results.get_pc_pred(-1, nocs_maps=nocs01_maps)

            if self.pred_filter is not None and self.save_filtered:
                # frame_971_view_00_nox00_gt.png
                # frame is just i (model index)
                # view is model_info[2]
                info = self.nocs_results.get_model_info(i)
                frame_idx = i
                if self.nocs_view_type == 'multi':
                    for view_idx in range(len(info)):
                        # save front and back pred
                        cur_front_path = 'frame_%03d_view_%02d_nox00_pred_filtered.png' % (frame_idx, view_idx)
                        cur_front_path = os.path.join(self.nocs_results.root, cur_front_path)
                        cur_back_path = 'frame_%03d_view_%02d_nox01_pred_filtered.png' % (frame_idx, view_idx)
                        cur_back_path = os.path.join(self.nocs_results.root, cur_back_path)
                        # print(cur_front_path)
                        # print(cur_back_path)
                        write_nocs_map(nocs00_maps[view_idx], cur_front_path)
                        write_nocs_map(nocs01_maps[view_idx], cur_back_path)
                elif self.nocs_view_type == 'single':
                    views_per_model = len(info)
                    for view_idx in range(len(info)):
                        global_frame_idx = i*views_per_model + view_idx
                        # save front and back pred
                        cur_front_path = 'frame_%03d_nox00_pred_filtered.png' % (global_frame_idx)
                        cur_front_path = os.path.join(self.nocs_results.root, cur_front_path)
                        cur_back_path = 'frame_%03d_nox01_pred_filtered.png' % (global_frame_idx)
                        cur_back_path = os.path.join(self.nocs_results.root, cur_back_path)
                        # print(cur_front_path)
                        # print(cur_back_path)
                        write_nocs_map(nocs00_maps[view_idx], cur_front_path)
                        write_nocs_map(nocs01_maps[view_idx], cur_back_path)

            if not self.no_camera_eval:
                # first evaluate camera pose, do this for every frame
                #   for both single-view and multi-view evaluation
                # solve for camera pose using nocs maps and point clouds
                cam_poses = self.nocs_results.get_camera_pose(i)
                for j in range(nocs00_maps.shape[0]):
                    nocs00_map = nocs00_maps[j]
                    nocs01_map = nocs01_maps[j]
                    nocs00_pc = nocs00_pcs[j]
                    nocs01_pc = nocs01_pcs[j]

                    # solve using both layers jointly
                    _, _, R_pred, P_pred, flip = self.estimateCameraPoseFromNM([nocs00_map, nocs01_map], [nocs00_pc, nocs01_pc], N=NUM_CAM_SOLVE_PTS)
                    # estimated position is w.r.t to (0.0, 0.0, 0.0) not NOCS center and in right-handed coords
                    P_pred += np.array([-0.5, -0.5, -0.5])
                    P_pred[0] *= -1.0
                    # change coordinates depending on if prediction is flipped
                    R_flip = axangle2mat(np.array([1.0, 0.0, 0.0]), np.radians(180), is_normalized=True)
                    R_noflip = axangle2mat(np.array([0.0, 1.0, 0.0]), np.radians(180), is_normalized=True)
                    # get GT pose
                    P_gt, gt_quat = cam_poses[j]
                    gt_quat = np.array([gt_quat[0], gt_quat[1], gt_quat[2], gt_quat[3]])
                    R_gt = quaternions.quat2mat(gt_quat).T

                    # compare (transform unit x-axis)
                    unit_dir = np.array([1.0, 0.0, 0.0])
                    # predicted tranform
                    R_pred = np.dot(R_pred, R_flip)
                    if not flip:
                        R_pred = np.dot(R_noflip, R_pred)
                    pred_dir = np.dot(R_pred, unit_dir)
                    pred_pos = P_pred
                    # ground truth transform
                    gt_dir = np.dot(R_gt, unit_dir)
                    gt_pos = P_gt

                    cur_pos_err = np.linalg.norm(gt_pos - pred_pos)
                    cam_pos_err.append(cur_pos_err)
                    cur_angle_err = np.degrees(np.arccos(np.dot(gt_dir, pred_dir)))
                    cam_rot_err.append(cur_angle_err)

                    # print('Model %d Frame %d:' % (i, j))
                    # print('Flip?:', flip)
                    # print('Pred Pos:', pred_pos)
                    # print('Pred X-dir:', pred_dir)
                    # print('GT Pos:', gt_pos)
                    # print('GT X-dir:', gt_dir)
                    # print('=======================================================')
                    # print('Pos err:', cur_pos_err)
                    # print('rot err:', cur_angle_err)
                    # print('=======================================================')

                    # if the option is turned on, also evaluate camera pose for comparison with DPC
                    # this method uses quaternion differences and only measures angle difference
                    if self.dpc_camera_eval:
                        gt_pos_unity, _ = cam_poses[j]
                        gt_pos_blender = np.array([-gt_pos_unity[0], -gt_pos_unity[2], gt_pos_unity[1]])
                        gt_quat = quaternion_from_campos(gt_pos_blender)
                        gt_quat /= np.linalg.norm(gt_quat)

                        pred_pos_unity = pred_pos
                        pred_pos_blender = np.array([-pred_pos_unity[0], -pred_pos_unity[2], pred_pos_unity[1]])
                        pred_quat = quaternion_from_campos(pred_pos_blender)
                        pred_quat /= np.linalg.norm(pred_quat)
                        
                        # look at the quaternion difference between GT and pred
                        gt_conj = quaternion_conjugate(gt_quat)
                        q_diff = quaternion_multiply(gt_conj, pred_quat)
                        ang_diff = 2 * np.arccos(q_diff[0])
                        if ang_diff > np.pi:
                            ang_diff -= 2*np.pi

                        dpc_rot_err.append(np.degrees(np.fabs(ang_diff)))

            if not self.no_pc_eval:
                # now we evaluate the estimated point cloud
                # get GT point cloud (union of all available views)
                gt_pc_frames = self.nocs_results.get_pc_gt(i)
                gt_pc = np.concatenate(gt_pc_frames, axis=0)
                if gt_pc.shape[0] > self.max_pts:
                    gt_pc = prune_pc(gt_pc, self.max_pts)

                chamfer = ChamferMetric(device='cuda')
                if self.nocs_view_type == 'single':
                    # for single-view we do evaluation for point cloud from every view
                    #   and then evaluate any additional passed in nviews by taking the union
                    #   of point clouds from the first N predicted views.
                    for j in range(nocs00_maps.shape[0]):
                        single_pc = np.concatenate([nocs00_pcs[j], nocs01_pcs[j]], axis=0)
                        if single_pc.shape[0] > self.max_pts:
                            single_pc = prune_pc(single_pc, self.max_pts)
                        # calculate chamfer distance
                        cur_err = chamfer.chamfer_dist(single_pc, gt_pc)
                        single_view_err.append(cur_err)
                        # print('SingleChamferErr (1 view):', cur_err)
                        if not self.no_save_pc:
                            cur_inf = self.nocs_results.get_model_info(i)[j]
                            save_pc_single(single_pc, self.pc_out_path, cur_inf)
                    # then multi-views (using the first n views)
                    for k, n in enumerate(self.num_views):
                        if n > nocs00_maps.shape[0]:
                            continue
                        # aggregate point cloud
                        pc_list = []
                        for j in range(n):
                            pc_list += [nocs00_pcs[j], nocs01_pcs[j]]
                        multi_pc = np.concatenate(pc_list, axis=0)
                        if multi_pc.shape[0] > self.max_pts:
                            multi_pc = prune_pc(multi_pc, self.max_pts)
                        cur_err = chamfer.chamfer_dist(multi_pc, gt_pc)
                        single_nview_errs[k].append(cur_err)
                        # print('SingleChamferErr (%d views): %f' % (n, cur_err))
                elif self.nocs_view_type == 'multi':
                    # for multi-view we only do a single evaluation on the point cloud
                    # from all views combined
                    pc_list = nocs00_pcs + nocs01_pcs
                    multi_pc = np.concatenate(pc_list, axis=0)
                    if multi_pc.shape[0] > self.max_pts:
                        multi_pc = prune_pc(multi_pc, self.max_pts)
                    # print(multi_pc.shape)
                    cur_err = chamfer.chamfer_dist(multi_pc, gt_pc)
                    multi_view_err.append(cur_err)
                    # print('MultiChamferErr (%d views): %f' % (len(nocs00_pcs), cur_err))
                    if not self.no_save_pc:
                        cur_inf = self.nocs_results.get_model_info(i)[0]
                        save_pc_multi(multi_pc, self.pc_out_path, cur_inf)

            if i % 20 == 0:
                self.log('Finished evaluating model ' + str(i) + '...')

        # save evaluation statistics and do any aggregation
        with open(os.path.join(self.out_path, 'cam_pose_results.csv'), 'w') as cam_out:
            err_writer = csv.writer(cam_out, delimiter=',')
            err_writer.writerow(['cat', 'model', 'frame', 'pos', 'rot'])
            for i, (pos_err, rot_err) in enumerate(zip(cam_pos_err, cam_rot_err)):
                info = self.nocs_results.meta_info[i]
                err_writer.writerow([info[0], info[1], info[2], pos_err, rot_err])
        if not self.no_camera_eval:
            self.log('Mean cam pos err: %06f' % (np.mean(np.array(cam_pos_err))))
            self.log('Mean cam rot err: %06f' % (np.mean(np.array(cam_rot_err))))

        if self.dpc_camera_eval:
            with open(os.path.join(self.out_path, 'dpc_cam_rot_results.csv'), 'w') as cam_out:
                err_writer = csv.writer(cam_out, delimiter=',')
                err_writer.writerow(['cat', 'model', 'frame', 'rot'])
                for i, rot_err in enumerate(dpc_rot_err):
                    info = self.nocs_results.meta_info[i]
                    err_writer.writerow([info[0], info[1], info[2], rot_err])
            self.log('Mean dpc cam rot err: %06f' % (np.mean(np.array(dpc_rot_err))))
            all_errors = np.array(dpc_rot_err)
            correct = all_errors < 30.0
            num_predictions = correct.shape[0]
            accuracy = np.count_nonzero(correct) / num_predictions
            median_error = np.sort(all_errors)[num_predictions // 2]
            self.log("accuracy: %f, median angular error: %f" % (accuracy, median_error))

        if self.nocs_view_type == 'single':
            with open(os.path.join(self.out_path, 'single_view_results.csv'), 'w') as res_out:
                err_writer = csv.writer(res_out, delimiter=',')
                err_writer.writerow(['cat', 'model', 'frame', 'chamfer'])
                for i, err in enumerate(single_view_err):
                    info = self.nocs_results.meta_info[i]
                    err_writer.writerow([info[0], info[1], info[2], err])
            self.log('Mean single view err: %06f' % (np.mean(np.array(single_view_err))))
            if len(self.num_views) != 0:
                with open(os.path.join(self.out_path, 'single_nview_results.csv'), 'w') as res_out:
                    err_writer = csv.writer(res_out, delimiter=',')
                    err_writer.writerow(['cat', 'model', 'nviews', 'chamfer'])
                    for i in range(self.nocs_results.length):
                        info = self.nocs_results.get_model_info(i)
                        cat, model_id = info[0][0:2]
                        for j in range(len(self.num_views)):
                            if len(single_nview_errs[j]) != 0:
                                err = single_nview_errs[j][i]
                                err_writer.writerow([cat, model_id, self.num_views[j], err])
                for i, err_list in enumerate(single_nview_errs):
                    if len(single_nview_errs[i]) != 0:
                        self.log('Mean %d-view err: %06f' % (self.num_views[i], np.mean(np.array(err_list))))
        elif self.nocs_view_type == 'multi':
            with open(os.path.join(self.out_path, 'multi_view_results.csv'), 'w') as res_out:
                err_writer = csv.writer(res_out, delimiter=',')
                err_writer.writerow(['cat', 'model', 'nviews', 'chamfer'])
                for i, err in enumerate(multi_view_err):
                    info = self.nocs_results.get_model_info(i)
                    cat, model_id = info[0][0:2]
                    nviews = len(info)
                    err_writer.writerow([cat, model_id, nviews, err])
            self.log('Mean multi view err: %06f' % (np.mean(np.array(multi_view_err))))

        # self.log('Finished evaluation!')


    def estimateCameraPoseFromNM(self, NOCSMapList, NOCSPtsList, N=None, Intrinsics=None):
        x_list = []
        X_list = []
        for NOCSMap, NOCSPts in zip(NOCSMapList, NOCSPtsList):
            ValidIdx = np.where(np.all(NOCSMap != [255, 255, 255], axis=-1)) # row, col

            # Create correspondences tuple list
            x = np.array([ValidIdx[1], ValidIdx[0]]) # row, col ==> u, v
            # Convert image coordinates from top left to bottom right (See Figure 6.2 in HZ)
            x[0, :] = NOCSMap.shape[1] - x[0, :]
            x[1, :] = NOCSMap.shape[0] - x[1, :]

            X = NOCSPts.T
            
            x_list.append(x)
            X_list.append(X)
        # combine all correspondences
        x = np.concatenate(x_list, axis=1)
        X = np.concatenate(X_list, axis=1)

        # Enough to do pose estimation from a subset of points but randomly distributed in the image
        MaxN = x.shape[1]
        if N is not None:
            MaxN = min(N, x.shape[1])
        RandIdx = [i for i in range(0, x.shape[1])]
        random.shuffle(RandIdx)
        RandIdx = RandIdx[:MaxN]
        x = x[:, RandIdx]
        X = X[:, RandIdx]

        Corr = []
        for i in range(0, max(X.shape)):
            Corr.append((x[:, i], X[:, i]))

        p, c, k, r, Flip = calibration.calculateCameraParameters(Corr)

        # print('Full estimate:\n')
        # print('R:\n', r, '\n')
        # print('C:\n', c, '\n')
        # print('K:\n', k, '\n\n')

        useEstimatedK = True
        if useEstimatedK == False and Intrinsics is not None:
            # Use passed K to refine r, c
            rc = np.linalg.inv(Intrinsics.Matrix) @ p
            r = rc[:, :-1]
            c = -r.T @ rc[:, -1]
            k = Intrinsics.Matrix

        # print('K-based estimate:\n')
        # print('R:\n', r, '\n')
        # print('C:\n', c, '\n')
        # print('K:\n', k, '\n\n')
        # print(c, '\n\n')
        # print('K:\n', k, '\n\n')
        # print(r, '\n\n')

        return p, k, r, c, Flip

    def log(self, string):
        print(string)
        with open(os.path.join(self.out_path, 'log.txt'), 'a') as f:
            f.write(string + '\n')


if __name__=='__main__':
    evaluation = NOCSEvaluation(sys.argv[1:])
    evaluation.run()
