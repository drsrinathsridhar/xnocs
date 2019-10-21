import sys, os

import numpy as np

from DPCEval import DPCEvaluation
from EvalUtils import viz_pcs, prune_pc, normalize_pc, quaternion_conjugate, as_rotation_matrix

def main(args):
    dpc_eval = DPCEvaluation(args)
    images_out = os.path.join(dpc_eval.out_path, 'images')
    if not os.path.exists(images_out):
        os.mkdir(images_out)
    pcl_out = os.path.join(dpc_eval.out_path, 'pcl')
    if not os.path.exists(pcl_out):
        os.mkdir(pcl_out)

    # need to first do an alignment between the predicted and ground truth reference frames
    default_num_align_models = 20
    num_align_models = min([default_num_align_models, dpc_eval.dpc_results.length])
    dpc_eval.log('Performing alignment...')
    final_align_quat = dpc_eval.find_global_alignment(num_align_models)
    # final_align_quat = np.array([ 0.02164055,  0.08669936,  0.02188158, -0.99575906]) # chair GT for debug
    final_align_quat_conj = quaternion_conjugate(final_align_quat)
    print(final_align_quat)
    final_align_R = as_rotation_matrix(final_align_quat)
    
    for i in range(20): #dpc_eval.dpc_results.length):
        dpc_id = dpc_eval.dpc_results.meta_info[i]
        nocs_id = dpc_eval.nocs_results.get_model_info(i)[0][1]
        if dpc_id != nocs_id:
            print('ERROR EVALUATION DATA IS NOT CONSISTENT BETWEEN DPC AND NOCS!!')
            quit()

        # get GT point cloud (union of all available views)
        gt_pc_frames = dpc_eval.nocs_results.get_pc_gt(i)
        gt_pc = np.concatenate(gt_pc_frames, axis=0)
        if gt_pc.shape[0] > dpc_eval.max_pts:
            gt_pc = prune_pc(gt_pc, dpc_eval.max_pts)
        gt_pc -= np.mean(gt_pc, axis=0)

        viz_pcs([gt_pc], os.path.join(images_out, dpc_id + '_gt_pc.png'))
        np.save(os.path.join(pcl_out, dpc_id + '_gt_pc'), gt_pc)

        nocs00_maps = dpc_eval.nocs_results.get_nox00_pred(i)
        nocs01_maps = dpc_eval.nocs_results.get_nox01_pred(i)
        nocs00_pcs = dpc_eval.nocs_results.get_pc_pred(-1, nocs_maps=nocs00_maps)
        nocs01_pcs = dpc_eval.nocs_results.get_pc_pred(-1, nocs_maps=nocs01_maps)

        pred_pcs = dpc_eval.dpc_results.pcs[i]
        for j in range(len(pred_pcs)):
            # DPC
            single_pc = pred_pcs[j]
            if single_pc.shape[0] > dpc_eval.max_pts:
                single_pc = prune_pc(single_pc, dpc_eval.max_pts)
            # center and align
            single_pc -= np.mean(single_pc, axis=0)
            single_pc = np.dot(final_align_R, single_pc.T).T
            single_pc = normalize_pc(single_pc)

            viz_pcs([single_pc], os.path.join(images_out, dpc_id + '_dpc_pc_' + str(j) + '.png'))
            np.save(os.path.join(pcl_out, dpc_id + '_dpc_pc_' + str(j)), single_pc)

            # NOCS
            single_pc = np.concatenate([nocs00_pcs[j], nocs01_pcs[j]], axis=0)
            if single_pc.shape[0] > dpc_eval.max_pts:
                single_pc = prune_pc(single_pc, dpc_eval.max_pts)
            single_pc -= np.mean(single_pc, axis=0)
            
            viz_pcs([single_pc], os.path.join(images_out, dpc_id + '_nocs_pc_' + str(j) + '.png'))            
            np.save(os.path.join(pcl_out, dpc_id + '_nocs_pc_' + str(j)), single_pc)       

if __name__=='__main__':
    main(sys.argv[1:])
