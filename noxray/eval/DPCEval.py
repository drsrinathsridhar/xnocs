import argparse, sys, os, random, csv

import numpy as np
import scipy.io
from transforms3d import quaternions

from NOCSEval import NOCSEvaluation
from chamfer_metric import ChamferMetric
from EvalUtils import prune_pc, viz_pcs, normalize_pc, open3d_icp, save_pc_single, save_pc_multi
from EvalUtils import quaternion_from_campos, quaternion_multiply, quaternion_conjugate, as_rotation_matrix, quatWAvgMarkley


class DPCEvaluation(NOCSEvaluation):
    ''' Evaluates differentiable point cloud's output for point cloud and camera pose. '''
    def __init__(self, args):
        super().__init__(args)
        self.dpc_results = DPCResults(self.dpc_path, limit=self.data_limit)

    def parse_args(self, args):
        super().parse_args(args)

        parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        
        parser.add_argument('--dpc-results', required=True, help='root of the dpc results data')
        parser.add_argument('--limit', type=int, default=None)

        flags, _ = parser.parse_known_args(args)

        self.dpc_path = flags.dpc_results
        self.data_limit = flags.limit
        if self.num_views is None:
            self.num_views = []
    
    def run(self):
        ''' Run through each model and evaluate point cloud prediction and camera pose. '''

        # need to first do an alignment between the predicted and ground truth reference frames
        default_num_align_models = 20
        num_align_models = min([default_num_align_models, self.dpc_results.length])
        self.log('Performing alignment...')
        final_align_quat = self.find_global_alignment(num_align_models)
        # final_align_quat = np.array([ 0.02164055,  0.08669936,  0.02188158, -0.99575906]) # chair GT for debug
        final_align_quat_conj = quaternion_conjugate(final_align_quat)
        print(final_align_quat)
        final_align_R = as_rotation_matrix(final_align_quat)
        # print(final_align_R)

        # now do the actual evaluation
        cam_rot_errs = []
        single_view_err = []
        single_nview_errs = []
        for n in self.num_views:
            single_nview_errs.append([])
        chamfer = ChamferMetric(device='cuda')

        for i in range(self.dpc_results.length):
            if not self.no_camera_eval:
                # convert GT camera pose to blender quaternions
                # compare angle of difference quaternion to outuput
                dpc_id = self.dpc_results.meta_info[i]
                nocs_id = self.nocs_results.get_model_info(i)[0][1]
                if dpc_id != nocs_id:
                    print('ERROR EVALUATION DATA IS NOT CONSISTENT BETWEEN DPC AND NOCS!!')
                    quit()
                cam_poses = self.dpc_results.camera_poses[i]
                for j in range(cam_poses.shape[0]):
                    gt_pos_unity = self.nocs_results.get_camera_pose(i)[j][0]
                    gt_pos_blender = np.array([-gt_pos_unity[0], -gt_pos_unity[2], gt_pos_unity[1]])
                    gt_quat = quaternion_from_campos(gt_pos_blender)
                    gt_quat /= np.linalg.norm(gt_quat)
                    
                    pred_quat = cam_poses[j]
                    pred_quat /= np.linalg.norm(pred_quat)
                    aligned_pred_quat = quaternion_multiply(pred_quat, final_align_quat_conj)
                    
                    # look at the quaternion difference between GT and pred
                    gt_conj = quaternion_conjugate(gt_quat)
                    q_diff = quaternion_multiply(gt_conj, aligned_pred_quat)
                    ang_diff = 2 * np.arccos(q_diff[0])
                    if ang_diff > np.pi:
                        ang_diff -= 2*np.pi

                    cam_rot_errs.append(np.degrees(np.fabs(ang_diff)))

            if not self.no_pc_eval:
                # now we evaluate the estimated point cloud
                # get GT point cloud (union of all available views)
                gt_pc_frames = self.nocs_results.get_pc_gt(i)
                gt_pc = np.concatenate(gt_pc_frames, axis=0)
                if gt_pc.shape[0] > self.max_pts:
                    gt_pc = prune_pc(gt_pc, self.max_pts)
                gt_pc -= np.mean(gt_pc, axis=0)

                # for single-view we do evaluation for point cloud from every view
                #   and then evaluate any additional passed in nviews by taking the union
                #   of point clouds from the first N predicted views.
                pred_pcs = self.dpc_results.pcs[i]
                aligned_pcs = [] #np.zeros((pred_pcs.shape[0], min(pred_pcs.shape[1], self.max_pts), 3))
                for j in range(len(pred_pcs)):
                    single_pc = pred_pcs[j]
                    if single_pc.shape[0] > self.max_pts:
                        single_pc = prune_pc(single_pc, self.max_pts)
                    # center and align
                    single_pc -= np.mean(single_pc, axis=0)
                    single_pc = np.dot(final_align_R, single_pc.T).T
                    single_pc = normalize_pc(single_pc)
                    aligned_pcs.append(single_pc)
                    # calculate chamfer distance
                    cur_err = chamfer.chamfer_dist(single_pc, gt_pc)
                    single_view_err.append(cur_err)
                    # print('SingleChamferErr (1 view):', cur_err)
                    if not self.no_save_pc:
                        cur_inf = self.nocs_results.get_model_info(i)[j]
                        save_pc_single(single_pc, self.pc_out_path, cur_inf)
                # then multi-views (using the first n views)
                for k, n in enumerate(self.num_views):
                    if n > len(pred_pcs):
                        continue
                    # aggregate point cloud
                    multi_pc = np.concatenate(aligned_pcs[:n], axis=0)
                    # multi_pc = aligned_pcs[:n].reshape((n*aligned_pcs.shape[1], 3))
                    if multi_pc.shape[0] > self.max_pts:
                        multi_pc = prune_pc(multi_pc, self.max_pts)
                    cur_err = chamfer.chamfer_dist(multi_pc, gt_pc)
                    single_nview_errs[k].append(cur_err)
                    # print('SingleChamferErr (%d views): %f' % (n, cur_err))

            if i % 20 == 0:
                self.log('Finished evaluating model ' + str(i) + '...')
                
        # calculate accuracy (% under 30 degrees) and median degree error
        all_errors = np.array(cam_rot_errs)
        correct = all_errors < 30.0
        num_predictions = correct.shape[0]
        accuracy = np.count_nonzero(correct) / num_predictions
        median_error = np.sort(all_errors)[num_predictions // 2]
        self.log("accuracy: %f, median angular error: %f" % (accuracy, median_error))

        # save evaluation statistics and do any aggregation
        with open(os.path.join(self.out_path, 'cam_pose_results.csv'), 'w') as cam_out:
            err_writer = csv.writer(cam_out, delimiter=',')
            err_writer.writerow(['cat', 'model', 'frame', 'rot'])
            for i, rot_err in enumerate(cam_rot_errs):
                info = self.nocs_results.meta_info[i] # can use meta info from here b/c in same order
                err_writer.writerow([info[0], info[1], info[2], rot_err])
        self.log('Mean cam rot err: %06f' % (np.mean(np.array(cam_rot_errs))))

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
                for i in range(self.dpc_results.length):
                    info = self.nocs_results.get_model_info(i)
                    cat, model_id = info[0][0:2]
                    for j in range(len(self.num_views)):
                        err = single_nview_errs[j][i]
                        err_writer.writerow([cat, model_id, self.num_views[j], err])
            for i, err_list in enumerate(single_nview_errs):
                self.log('Mean %d-view err: %06f' % (self.num_views[i], np.mean(np.array(err_list))))


    def find_global_alignment(self, num_align_models):
        ''' Finds a rigid rotation between the network output and the ground truth coordinate system. '''
        # use first N models or how every many are available
        # heavily based on differentiable point clouds code
        views_per_model = len(self.dpc_results.pcs[0])
        rmse_vals = np.zeros((num_align_models, views_per_model)) # 5 views per model
        align_quats = np.zeros((num_align_models, views_per_model, 4))
        for i in range(num_align_models):
            # same GT for every point cloud in this model
            gt_pc_frames = self.nocs_results.get_pc_gt(i)
            gt_pc = np.concatenate(gt_pc_frames, axis=0)
            if gt_pc.shape[0] > self.max_pts:
                gt_pc = prune_pc(gt_pc, self.max_pts)
            gt_pc -= np.mean(gt_pc, axis=0)

            # viz_pcs([gt_pc], os.path.join(self.out_path, 'gt_pc' + str(i) + '.png'))

            pred_pcs = self.dpc_results.pcs[i]
            cam_poses = self.dpc_results.camera_poses[i]
            for j in range(len(pred_pcs)):
                pc_pred = pred_pcs[j]
                if pc_pred.shape[0] > self.max_pts:
                    pc_pred = prune_pc(pc_pred, self.max_pts)
                pc_pred -= np.mean(pc_pred, axis=0)
                pc_pred = normalize_pc(pc_pred)

                # first rough align with camera quats
                gt_pos_unity = self.nocs_results.get_camera_pose(i)[j][0]
                gt_pos_blender = np.array([-gt_pos_unity[0], -gt_pos_unity[2], gt_pos_unity[1]])
                gt_quat = quaternion_from_campos(gt_pos_blender)
                gt_quat /= np.linalg.norm(gt_quat)
                
                pred_quat = cam_poses[j]
                pred_quat /= np.linalg.norm(pred_quat)

                quat_unrot = quaternion_multiply(quaternion_conjugate(gt_quat), pred_quat)
                R_init = as_rotation_matrix(quat_unrot)

                # then refine with ICP
                reg_p2p = open3d_icp(pc_pred, gt_pc, init_rotation=R_init)
                # print(reg_p2p.inlier_rmse)
                rmse_vals[i, j] = reg_p2p.inlier_rmse
                T = np.array(reg_p2p.transformation)
                R = T[:3, :3]
                assert (np.fabs(np.linalg.det(R) - 1.0) <= 0.0001)
                align_quat = quaternions.mat2quat(R)
                align_quats[i, j] = align_quat

                # chair GT for debugging
                # R = np.array([[-0.98402981,  0.04689179, -0.1717163 ], 
                #                 [-0.03930331, -0.99810577, -0.04733001],
                #                 [-0.17361041, -0.03982512, 0.98400883]])

                # aligned_pc_pred = np.dot(R, pc_pred.T).T
                # viz_pcs([aligned_pc_pred], os.path.join(self.out_path, 'aligned_pred_pc' + str(i) + str(j) + '.png'))
                # viz_pcs([gt_pc, aligned_pc_pred], os.path.join(self.out_path, 'overlay' + str(i) + str(j) + '.png'))

        num_to_estimate = int(num_align_models*0.75)
        num_to_keep = int(views_per_model*0.6)
        self.log('Using ' + str(num_to_keep) + ' views from the ' + str(num_to_estimate) + ' best models.')
        # remove outliers from each model based on rmse
        filtered_rmse = np.zeros((num_align_models, num_to_keep))
        filtered_quats = np.zeros((num_align_models, num_to_keep, 4))
        for i in range(num_align_models):
            sort_inds = np.argsort(rmse_vals[i,:])
            sort_inds = sort_inds[:num_to_keep]
            filtered_rmse[i] = rmse_vals[i, sort_inds]
            filtered_quats[i] = align_quats[i, sort_inds, :]
        # now remove worst performing models
        mean_rmse = np.mean(filtered_rmse, axis=1)
        sort_inds = np.argsort(mean_rmse)
        sort_inds = sort_inds[:num_to_estimate]
        ref_rots = filtered_quats[sort_inds,:,:]
        ref_rots = ref_rots.reshape((-1, 4))
        print(ref_rots)
        # find the mean rotation to get our final alignment rotation
        final_align_quat = quatWAvgMarkley(ref_rots)

        return final_align_quat

class DPCResults(object):
    ''' Structure to hold results from DPC model. Indexed by model not by individual frames. '''
    def __init__(self, root, limit=None):
        self.root = root
        if not os.path.exists(self.root):
            print('Could not find DPC results data at given path!')
            return

        self.meta_info = [] # model id
        self.camera_poses = [] # camera quaternions (5, 4)
        self.pcs = [] # point clouds (5, N, 3)

        # all data is stored in scipy mats so just read all results into memory
        all_mats = sorted([os.path.join(self.root, f) for f in os.listdir(self.root)])
        for i, mat_path in enumerate(all_mats):
            if limit is not None and i >= limit:
                break
            cur_mat = scipy.io.loadmat(mat_path)
            cur_pts = cur_mat['points']
            cur_cam_pose = cur_mat['camera_pose']

            # filter pc to remove noisy outliers
            pts_list = []
            for j in range(cur_pts.shape[0]):
                pc = cur_pts[j]
                good_inds = np.where(np.abs(pc[:,0]) < 0.4)[0]
                pc = pc[good_inds, :]
                good_inds = np.where(np.abs(pc[:,1]) < 0.4)[0]
                pc = pc[good_inds, :]
                good_inds = np.where(np.abs(pc[:,2]) < 0.4)[0]
                pc = pc[good_inds, :]
                pts_list.append(pc)

            self.camera_poses.append(cur_cam_pose)
            self.pcs.append(pts_list)
            self.meta_info.append(mat_path.split('/')[-1].split('_')[0])

        self.length = len(self.meta_info)  

if __name__=='__main__':
    evaluation = DPCEvaluation(sys.argv[1:])
    evaluation.run()
