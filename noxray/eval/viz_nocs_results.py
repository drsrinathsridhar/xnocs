import sys, os

import numpy as np

from NOCSEval import NOCSEvaluation
from EvalUtils import viz_pcs, prune_pc

def main(args):
    nocs_eval = NOCSEvaluation(args)
    images_out = os.path.join(nocs_eval.out_path, 'images')
    if not os.path.exists(images_out):
        os.mkdir(images_out)
    pcl_out = os.path.join(nocs_eval.out_path, 'pcl')
    if not os.path.exists(pcl_out):
        os.mkdir(pcl_out)
    
    for i in range(10): #dpc_eval.dpc_results.length):
        nocs_id = nocs_eval.nocs_results.get_model_info(i)[0][1]
       
        # get GT point cloud (union of all available views)
        gt_pc_frames = nocs_eval.nocs_results.get_pc_gt(i)
        #print(len(gt_pc_frames))
        gt_pc = np.concatenate(gt_pc_frames, axis=0)
        if gt_pc.shape[0] > nocs_eval.max_pts:
            gt_pc = prune_pc(gt_pc, nocs_eval.max_pts)
        gt_pc -= np.mean(gt_pc, axis=0)

        viz_pcs([gt_pc], os.path.join(images_out, nocs_id + '_gt_pc.png'))
        np.save(os.path.join(pcl_out, nocs_id + '_gt_pc'), gt_pc)

        nocs00_maps = nocs_eval.nocs_results.get_nox00_pred(i)
        nocs01_maps = nocs_eval.nocs_results.get_nox01_pred(i)
        nocs00_pcs = nocs_eval.nocs_results.get_pc_pred(-1, nocs_maps=nocs00_maps)
        nocs01_pcs = nocs_eval.nocs_results.get_pc_pred(-1, nocs_maps=nocs01_maps)
        #print(len(nocs00_pcs))
        #print(len(nocs01_pcs))

        # NOCS
        if nocs_eval.nocs_view_type == 'single':
            for j in range(nocs00_maps.shape[0]):
                single_pc = np.concatenate([nocs00_pcs[j], nocs01_pcs[j]], axis=0)
                if single_pc.shape[0] > nocs_eval.max_pts:
                    single_pc = prune_pc(single_pc, nocs_eval.max_pts)
                single_pc -= np.mean(single_pc, axis=0)

                viz_pcs([single_pc], os.path.join(images_out, nocs_id + '_nocs_pc_view' + str(j) + '.png'))            
                np.save(os.path.join(pcl_out, nocs_id + '_nocs_pc_view' + str(j)), single_pc)
        elif nocs_eval.nocs_view_type == 'multi':
            pc_list = nocs00_pcs + nocs01_pcs
            multi_pc = np.concatenate(pc_list, axis=0)
            if multi_pc.shape[0] > nocs_eval.max_pts:
                multi_pc = prune_pc(multi_pc, nocs_eval.max_pts)
            multi_pc -= np.mean(multi_pc, axis=0)
            
            viz_pcs([multi_pc], os.path.join(images_out, nocs_id + '_nocs_pc.png')) 
            np.save(os.path.join(pcl_out, nocs_id + '_nocs_pc'), multi_pc)

if __name__=='__main__':
    main(sys.argv[1:])
