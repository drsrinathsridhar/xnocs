import sys, os, argparse, glob, pickle, operator, traceback
from pprint import pprint

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))

from tk3dv.ptTools import ptUtils, ptNets
import mv_nxm

Parser = argparse.ArgumentParser(description='Queue up all config files in a directory and run one after another.', fromfile_prefix_chars='@')
ArgGroup = Parser.add_argument_group()
ArgGroup.add_argument('-n', '--base-dir', help='Enter the base directory name.', required=True)
ArgGroup.add_argument('--no-train', help='Perform only validation.', action='store_true')
Parser.set_defaults(no_train=False)

HEADER = '-'*200

def printStats(ExptDict, Range=None):
    if Range is None or Range < 0:
        Range = len(ExptDict)
    Range = Range if Range < len(ExptDict) else len(ExptDict)
    LastLossDict = dict()
    for Item in ExptDict:
        ShortItem = os.path.basename(Item)
        LastLossDict[ShortItem] = ExptDict[Item][-1]

    SortedExpt = sorted(LastLossDict.items(), key=operator.itemgetter(1))
    print('[ INFO ]: Top {} experiments so far:'.format(Range))
    for i in range(0, Range):
        print(SortedExpt[i])

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()
    ptUtils.printArgs(Args)

    AllConfigs = glob.glob(os.path.join(Args.base_dir, '*.config'))
    AllConfigs.sort()

    if len(AllConfigs) <= 0:
        print('[ WARN ]: No config files found. Aborting.')
        exit()

    NetConfig = ptNets.ptNetExptConfig(['@' + AllConfigs[0]])
    OutDir = ptUtils.expandTilde(NetConfig.Args.output_dir)
    StoreFileName = NetConfig.Args.expt_name
    StatusDataFile = os.path.join(OutDir, StoreFileName + '.pk')

    ExptLossHistories = dict()
    if os.path.exists(StatusDataFile):
        with open(StatusDataFile, 'rb') as handle:
            ExptLossHistories = pickle.load(handle)
            printStats(ExptLossHistories)

    for Ctr, Config in enumerate(AllConfigs):
        print(HEADER)
        print('[ INFO ]: Running experiment {}/{} ...'.format(Ctr+1, len(AllConfigs)))
        if Config not in ExptLossHistories:
            try:
                if Args.no_train == False:
                    ExptLossHistories[Config] = mv_nxm.main(['@' + Config])
                    with open(StatusDataFile, 'wb') as handle:
                        pickle.dump(ExptLossHistories, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    print('[ INFO ]: Skipping training as requested. Proceeding to validation')
            except Exception as e:
                print('[ WARN ]: Unable to run', Config, 'exception', e, 'Continuing...')
                print(traceback.format_exc())
        else:
            print('[ INFO ]:', Config, 'already done. Proceeding to validation.')

        # Run validation
        try:
            print('[ INFO ]: Generating visualizations of final model in ValResults.')
            ValLoss = mv_nxm.main(['@' + Config, '--mode=val', '--test-samples=1000000'])
        except Exception as e:
            print('[ WARN ]: Unable to run', Config, 'validation. Exception', e, 'Continuing...')
            print(traceback.format_exc())

    print('[ INFO ]: All experiments in {} done.'.format(Args.base_dir))