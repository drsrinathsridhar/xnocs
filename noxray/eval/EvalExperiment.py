import subprocess, argparse, sys, os, multiprocessing, shutil

# from NOCSEval import NOCSEvaluation

parser = argparse.ArgumentParser()

# root of the experiment (should contain the GPU** directories)
parser.add_argument('--root', required=True, help='root of the finished experiments (should contain the GPU0* dirs)')
parser.add_argument('--single-idx', default=None, type=int, help='will only evaluate the runs at this gpu idx')
parser.add_argument('--type', help='experiment type (single or mutli) view.', choices=['single', 'multi'], required=True)
parser.add_argument('--comparison', dest='comparison', action='store_true')
parser.set_defaults(comparison=False)
parser.add_argument('--camera', dest='eval_camera', action='store_true')
parser.set_defaults(eval_camera=False)
parser.add_argument('--skip-validation', dest='skip_val', action='store_true')
parser.set_defaults(skip_val=False)
parser.add_argument('--pred-filter', help='filter to apply to predicted NOCS maps. [bilateral, median]', default=None)



flags, _ = parser.parse_known_args()

exp_root = flags.root
exp_type = flags.type
skip_val = flags.skip_val
single_idx = flags.single_idx
eval_camera = flags.eval_camera
script_dir = os.path.dirname(os.path.realpath(__file__))
is_comparison = flags.comparison
pred_filter = flags.pred_filter

multi_views_to_test = [1, 2, 3, 5, 10]
variable_view_str = '--enable-variable-set-size'
num_views_str = '--set-size='
expt_name_str = '--expt-name='

val_script = '../nxm/configs/runConfigDir.py'
if exp_type == 'multi':
    val_script = '../mv_nxm/configs/runConfigDir.py'

def evaluate_experiment(exp_path):
    ''' Runs validation for each run in the experiment, and then runs the evaluation '''
    gpu_dir = exp_path
    gpu_runs = sorted([os.path.join(gpu_dir, f) for f in os.listdir(gpu_dir) if os.path.isdir(os.path.join(gpu_dir, f))])
    # print(gpu_runs)

    if exp_type == 'multi' and not skip_val:
        # deal with variable view experiments
        # we make a copy of these for each number of views we want to evaluate so we can do it all separately
        new_gpu_runs = []
        for i, run_path in enumerate(gpu_runs):
            expt_base_name = run_path.split('/')[-1]
            if expt_base_name.find('_EVAL_COPY_') != -1:
                # this is a data copy from a previous run, skip it to avoid recursive copy
                print(expt_base_name + ' looks like data from a previous multi-view evaluation, skipping...')
                continue
            if expt_base_name.find('SUBMISSION') != -1:
                # this is an old copy from the submission runs, don't want to copy this
                print(expt_base_name + ' looks like old Submission data, skipping...')
                continue
            # find any variable view experiments
            with open(run_path + '.config', 'r') as run_cfg_file:
                cfg_str = run_cfg_file.read()
            var_idx = cfg_str.find(variable_view_str)
            if var_idx == -1:
                # not variable
                continue
            print('Found variable run: ' + expt_base_name)
            print('Setting up variable multi-view evaluation...')
            max_views_idx = cfg_str.find(num_views_str)
            max_views = int(cfg_str[max_views_idx + len(num_views_str)])
            if max_views == 1:
                if cfg_str[max_views_idx + len(num_views_str) + 1] == '0':
                    max_views = 10
                elif cfg_str[max_views_idx + len(num_views_str) + 1] == '2':
                    max_views = 12
            print('Max views: ' + str(max_views))
            # copy the data for each evaluation less than max views
            # only need to copy the latest checkpoint, so find this first
            checkpoint_names = sorted([f for f in os.listdir(run_path) if f[0] != '.' and f.split('.')[-1] == 'tar'])
            latest_ckpnt_name = checkpoint_names[-1] # names include timestamp so last is most recent
            latest_ckpnt_path = os.path.join(run_path, latest_ckpnt_name)
            for num_views_eval in multi_views_to_test:
                if num_views_eval >= max_views:
                    continue # will let original folder be max views eval
                new_expt_name = expt_base_name + '_EVAL_COPY_' + str(num_views_eval) + '_VIEWS'
                print('Creating ' + new_expt_name + '...')
                # copy config
                dest_cfg_path = os.path.join(exp_path, new_expt_name + '.config')
                shutil.copyfile(run_path + '.config', dest_cfg_path)
                # edit config with new experiment name and set size (append to end)
                with open(dest_cfg_path, 'a') as new_cfg_file:
                    new_cfg_file.write(expt_name_str + new_expt_name + '\n')
                    new_cfg_file.write(num_views_str + str(num_views_eval) + '\n')
                # now copy over actual data
                dest_dir_path = os.path.join(exp_path, new_expt_name)
                dest_ckpnt_path = os.path.join(dest_dir_path, latest_ckpnt_name)
                if not os.path.exists(dest_dir_path):
                    # shutil.copytree(run_path, dest_data_path) OLD
                    os.mkdir(dest_dir_path)
                    shutil.copyfile(latest_ckpnt_path, dest_ckpnt_path)
                    # save for our evaluation
                    new_gpu_runs.append(dest_dir_path)
                else:
                    print('Data already exists, not copying again.')
                
        gpu_runs += new_gpu_runs

    # filter gpu_runs to remove SUBMISSION directories
    filtered_gpu_runs = []
    for run_dir in gpu_runs:
        if run_dir.find('SUBMISSION') != -1:
            print('Removing submission dir ' + run_dir + ' from evaluation...')
        else:
            filtered_gpu_runs.append(run_dir)
    gpu_runs = filtered_gpu_runs

    # first run validation
    val_script = '../nxm/configs/runConfigDir.py'
    if exp_type == 'multi':
        val_script = '../mv_nxm/configs/runConfigDir.py'
    val_args = ['python', val_script, '-n', gpu_dir, '--no-train']
    # print(val_args)
    if not skip_val:
        subprocess.run(val_args)
    else:
        print('Skipping model validation...')

    # now the evaluations
    print('Starting evaluation for ' + gpu_dir.split('/')[-1])
    for run_path in gpu_runs:
        val_out_path = os.path.join(run_path, 'ValResults')
        eval_out_path = os.path.join(run_path, 'EvalResults')
        if pred_filter is not None and pred_filter in ['median', 'bilateral']:
            eval_out_path += '_filtered'
        # create directory to output the evaluation
        if not os.path.exists(eval_out_path):
            os.mkdir(eval_out_path)
        # write a config to that directory and create the evaluator
        config = ['--nocs-results', val_out_path, '--type', exp_type, '--out', eval_out_path]
        if not eval_camera:
            config += ['--no-camera']
        if is_comparison:
            config += ['--dpc_camera']
        if exp_type == 'single':
            config += ['--nviews', '2', '3', '5', '10']
        if pred_filter:
            if pred_filter in ['median', 'bilateral']:
                config += ['--pred-filter', pred_filter, '--save-filtered']
        config_path = os.path.join(eval_out_path, 'eval.cfg')
        with open(config_path, 'w') as cfg_file:
            cfg_file.write('\n'.join(config))

        # have to run in seperate process for GPU reasons
        subprocess.Popen(['python', 'NOCSEval.py', '@' + config_path])


def main():
    # find all experiments (GPU directories) and their contained runs
    if not os.path.exists(exp_root):
        print('Cannot find experiment at directory ' + exp_root)
        return

    # all GPU directories
    all_exps = sorted([os.path.join(exp_root, f) for f in os.listdir(exp_root) if os.path.isdir(os.path.join(exp_root, f))])
    if single_idx is not None:
        all_exps = [all_exps[single_idx]]
    # print(all_exps)
    print('Running evaluations using ' + val_script)
    print('Starting evaluations...')
    pool_size = multiprocessing.cpu_count()
    print('Using ' + str(pool_size) + ' workers.')
    pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
    pool.map(evaluate_experiment, all_exps)
    pool.close()
    pool.join()  
        

if __name__=='__main__':
    main()