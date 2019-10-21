import sys, os

import numpy as np

from R2N2Eval import R2N2Evaluation
from EvalUtils import viz_pcs, prune_pc, normalize_pc, quaternion_conjugate, as_rotation_matrix

def main(args):
    r2n2_eval = R2N2Evaluation(args)
    images_out = os.path.join(r2n2_eval.out_path, 'images')
    if not os.path.exists(images_out):
        os.mkdir(images_out)
    pcl_out = os.path.join(r2n2_eval.out_path, 'pcl')
    if not os.path.exists(pcl_out):
        os.mkdir(pcl_out)
    
    for i in range(100): #dpc_eval.dpc_results.length):
        r2n2_id = r2n2_eval.r2n2_results.meta_info[i]
        nocs_id = r2n2_eval.nocs_results.get_model_info(i)[0][1]
        if r2n2_id != nocs_id:
            print('ERROR EVALUATION DATA IS NOT CONSISTENT BETWEEN R2N2 AND NOCS!!')
            quit()

        # get GT point cloud (union of all available views)
        gt_pc_frames = r2n2_eval.nocs_results.get_pc_gt(i)
        #print(len(gt_pc_frames))
        gt_pc = np.concatenate(gt_pc_frames, axis=0)
        if gt_pc.shape[0] > r2n2_eval.max_pts:
            gt_pc = prune_pc(gt_pc, r2n2_eval.max_pts)
        gt_pc -= np.mean(gt_pc, axis=0)

        viz_pcs([gt_pc], os.path.join(images_out, r2n2_id + '_gt_pc.png'))
        np.save(os.path.join(pcl_out, r2n2_id + '_gt_pc'), gt_pc)

        nocs00_maps = r2n2_eval.nocs_results.get_nox00_pred(i)
        nocs01_maps = r2n2_eval.nocs_results.get_nox01_pred(i)
        nocs00_pcs = r2n2_eval.nocs_results.get_pc_pred(-1, nocs_maps=nocs00_maps)
        nocs01_pcs = r2n2_eval.nocs_results.get_pc_pred(-1, nocs_maps=nocs01_maps)
        #print(len(nocs00_pcs))
        #print(len(nocs01_pcs))

        # R2N2
        multi_pc = r2n2_eval.r2n2_results.pcs[i]
        if multi_pc.shape[0] > r2n2_eval.max_pts:
            multi_pc = prune_pc(multi_pc, r2n2_eval.max_pts)

        viz_pcs([multi_pc], os.path.join(images_out, r2n2_id + '_r2n2_pc.png'))
        np.save(os.path.join(pcl_out, r2n2_id + '_r2n2_pc'), multi_pc)

        # NOCS
        pc_list = nocs00_pcs + nocs01_pcs
        multi_pc = np.concatenate(pc_list, axis=0)
        if multi_pc.shape[0] > r2n2_eval.max_pts:
            multi_pc = prune_pc(multi_pc, r2n2_eval.max_pts)
        multi_pc -= np.mean(multi_pc, axis=0)
            
        viz_pcs([multi_pc], os.path.join(images_out, r2n2_id + '_nocs_pc.png'))            
        np.save(os.path.join(pcl_out, r2n2_id + '_nocs_pc'), multi_pc)       

if __name__=='__main__':
    main(sys.argv[1:])
