import sys, os, argparse, glob, pickle, operator
from pprint import pprint

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))
from shutil import copy2

from tk3dv.ptTools import ptUtils, ptNets
import runConfigDir

Parser = argparse.ArgumentParser(description='View results from all pickle files in an experiment directory.', fromfile_prefix_chars='@')
ArgGroup = Parser.add_argument_group()
ArgGroup.add_argument('-n', '--base-dir', help='Enter the base directory name.', required=True)
ArgGroup.add_argument('-o', '--output-dir', help='Copy all config files corresponding to N top results to another directory sorted by GPU ID.', required=False)
ArgGroup.add_argument('-t', '--top', help='Specify the number of top entries to show/copy. Default is all.', type=int, default=-1, required=False)

HEADER = '-'*200

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()
    ptUtils.printArgs(Args)

    StatusDataFiles = glob.iglob(Args.base_dir + '/**/*.pk', recursive=True)
    ExptLossHistories = dict()
    TopConfigs = []
    for PkF in StatusDataFiles:
        with open(PkF, 'rb') as handle:
            LocExpt = pickle.load(handle)
            ExptLossHistories.update(LocExpt)

    runConfigDir.printStats(ExptLossHistories, Args.top)

    Top = Args.top if Args.top > 0 else len(ExptLossHistories)
    if Args.output_dir is not None:
        if os.path.exists(Args.output_dir) == False:
            raise RuntimeError('Directory not found {}'.format(Args.output_dir))
        LastLossDict = dict()
        for Item in ExptLossHistories:
            LastLossDict[Item] = ExptLossHistories[Item][-1]
        SortedExpt = sorted(LastLossDict.items(), key=operator.itemgetter(1))

        for Ctr, Tup in enumerate(SortedExpt, 0):
            if Ctr>=Top: break
            Loss = Tup[-1]
            ShortItem = os.path.basename(Tup[0])
            print('[ INFO ]: {} Copying {} to {}'.format(Ctr+1, ShortItem, Args.output_dir))
            copy2(Tup[0], Args.output_dir)
