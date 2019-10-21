import sys, os, random, itertools, argparse
import numpy as np

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))

from tk3dv.ptTools import ptUtils, ptNets
import nxm_config

Parser = argparse.ArgumentParser(description='Create configs for grid searching.', fromfile_prefix_chars='@')
ArgGroup = Parser.add_argument_group()
ArgGroup.add_argument('-n', '--base-name', help='Enter the base experiment name.', required=True)
ArgGroup.add_argument('-o', '--output-dir', help='Specify *relative* output directory (relative to config file).', required=True)

Modes = ['train']
Datasets = ['ShapeNetCOCODataset'] # ['ShapeNetCOCODataset', 'HandRigDataset']
BatchSizes = [2] #1, 2, 4, 8] # LABEL
LearningRates = [0.0001] #[0.1, 0.01, 0.001, 0.0001]#[, 0.00005, 0.00001]  # LABEL
Archs = [ 'SegNetSkip' ]# ['SegNet', 'SegNetSkip']  # LABEL
Categories = ['cars', 'airplanes', 'chairs', 'all']  # LABEL
nEpochs = 50
nGPUs = 8  # LABEL
Seed = 0  # LABEL
SaveFreq = 5 # int(nEpochs / 40)
DataLimit = -1
ValDataLimit = -1
InputDir = '~/input/DPC_ComparisonData'
OutputDir = '.'
MaskStatus = [''] # ['', '--no-mask']  # LABEL
HallucinateStatus = ['', '--no-color-hallucinate']  # LABEL
Losses = ['l2'] # ['l2', 'chamfer']

# Problem-specific config
NOCSConfigList = ['--mode={}', '--gpu={}', '--seed={}', '--data-limit={}', '--load-memory', '--dataset={}', '--arch={}', '--val-data-limit={}', '--category={}', '{}', '{}', '--loss={}']
# Network config
NetConfigList = ['--learning-rate={}', '--batch-size={}', '--expt-name={}', '--input-dir={}', '--rel-output-dir={}', '--epochs={}', '--save-freq={}']

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()
    ptUtils.printArgs(Args)

    OutputDir = Args.output_dir

    GridSearchParams = list(itertools.product(Modes, BatchSizes, LearningRates, Datasets, Archs, Categories, MaskStatus, HallucinateStatus, Losses))
    print('[ INFO ]: Generating {} config files and writing to {}.'.format(len(GridSearchParams), Args.base_name))

    ptUtils.makeDir(Args.base_name)
    SubDirCtr = 0
    for Ctr, Config in enumerate(GridSearchParams, 0):
        GPUID = Ctr%nGPUs
        # Create directory each time we hit GPU00
        if GPUID == 0:
            CurrentSubDir = os.path.join(Args.base_name, Args.base_name + '-ExpGroup_' + str(SubDirCtr).zfill(2))
            SubDirCtr += 1
            ptUtils.makeDir(CurrentSubDir)
            GPUDirs = []
            for n in range(nGPUs):
                GPUDirs.append(os.path.join(CurrentSubDir, 'GPU' + str(n).zfill(2)))
                ptUtils.makeDir(GPUDirs[-1])

        NewNOCSConfig = NOCSConfigList.copy()
        NewNOCSConfig[0] = NewNOCSConfig[0].replace('{}', str(Config[0])) # mode
        NewNOCSConfig[1] = NewNOCSConfig[1].replace('{}', str(Ctr%nGPUs)) # GPU
        NewNOCSConfig[2] = NewNOCSConfig[2].replace('{}', str(Seed))  # seed
        if DataLimit is not None:
            NewNOCSConfig[3] = NewNOCSConfig[3].replace('{}', str(DataLimit))  # data-limit
        else:
            NewNOCSConfig[3] = ''
        NewNOCSConfig[5] = NewNOCSConfig[5].replace('{}', str(Config[3]))  # dataset
        NewNOCSConfig[6] = NewNOCSConfig[6].replace('{}', str(Config[4]))  # architecture
        if ValDataLimit is not None:
            NewNOCSConfig[7] = NewNOCSConfig[7].replace('{}', str(ValDataLimit))  # val-data-limit
        else:
            NewNOCSConfig[7] = ''
        NewNOCSConfig[8] = NewNOCSConfig[8].replace('{}', str(Config[5])) # category
        NewNOCSConfig[9] = NewNOCSConfig[9].replace('{}', str(Config[6])) # mask
        NewNOCSConfig[10] = NewNOCSConfig[10].replace('{}', str(Config[7]))  # color hallucinate
        NewNOCSConfig[11] = NewNOCSConfig[11].replace('{}', str(Config[8]))  # Loss

        NOCSConfig = nxm_config.nxm_config(NewNOCSConfig)

        LR = '{0:.6f}'.format(Config[2])
        ExptName = Args.base_name
        # ExptName += '-BS_' + str(Config[1]) +'-LR_' + LR + '-Seed_' + str(Seed) + '-GPU_' + str(GPUID).zfill(2)
        # LABELS = BS, Arch, Category, LR, GPU, seed, Mask, Color halluncinate, Peel status
        ExptName += '-BS_' + str(Config[1]) + '-Arch_' + str(Config[4]) + '-' + str(Config[5]) + '-LR_' + LR + '-Seed_' + str(Seed)
        if Config[7] is not '--no-color-hallucinate':
            ExptName += '_PeeledColor'
        if Config[6] is not '--no-mask':
            ExptName += '_WithMask'
        ExptName += '-GPU_' + str(GPUID).zfill(2)

        NewNetConfig = NetConfigList.copy()
        NewNetConfig[0] = NewNetConfig[0].replace('{}', LR)  # learning-rate
        NewNetConfig[1] = NewNetConfig[1].replace('{}', str(Config[1]))  # batch-size
        NewNetConfig[2] = NewNetConfig[2].replace('{}', str(ExptName))  # expt-name
        NewNetConfig[3] = NewNetConfig[3].replace('{}', str(InputDir))  # input-dir
        # CurrOutputDir = os.path.join(OutputDir, GPUDirs[GPUID])
        # NewNetConfig[4] = NewNetConfig[4].replace('{}', str(CurrOutputDir))  # output-dir
        NewNetConfig[4] = NewNetConfig[4].replace('{}', str(OutputDir))  # output-dir
        NewNetConfig[5] = NewNetConfig[5].replace('{}', str(nEpochs))  # epochs
        NewNetConfig[6] = NewNetConfig[6].replace('{}', str(SaveFreq))  # save-freq

        NetConfig = ptNets.ptNetExptConfig(NewNetConfig)
        print('--------------------------------')

        ConfigFilePath = os.path.join(GPUDirs[GPUID], ExptName + '.config')
        NOCSConfig.serialize(ConfigFilePath, isAppend=False)
        NetConfig.serialize(ConfigFilePath, isAppend=True)

    print('[ INFO ]: Finished generating {} config files.'.format(len(GridSearchParams)))