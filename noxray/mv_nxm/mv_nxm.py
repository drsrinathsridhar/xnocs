import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import sys, os, cv2, math, json

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '../../models'))
import SetSegNet
from tk3dv.ptTools import ptUtils

sys.path.append(os.path.join(FileDirPath, '../loaders'))
import MShapeNetCOCODataset

sys.path.append(os.path.join(FileDirPath, './configs'))
import mv_nxm_config

sys.path.append(os.path.join(FileDirPath, '../eval'))
from chamfer_metric import ChamferMetric

from MShapeNetCOCODataset import MShapeNetCOCODataset as MSNCD
from ShapeNetCOCODataset import ShapeNetCOCODataset as SNCD

def validate(Args, LossFunc, TestDataLoader, Net, TestDevice, OutDir=None):
    TestNet = Net.to(TestDevice)
    nSamples = min(Args.test_samples, len(TestDataLoader))
    print('[ INFO ]: Testing on', nSamples, 'samples')

    if os.path.exists(OutDir) == False:
        os.makedirs(OutDir)
    ModelInfoPath = os.path.join(OutDir, 'model_info.txt')
    if os.path.exists(ModelInfoPath):
        os.remove(ModelInfoPath)

    ValLosses = []
    ValDists = []
    Tic = ptUtils.getCurrentEpochTime()
    for i, (Data, Targets) in enumerate(TestDataLoader, 0):  # Get each batch
        if i > (nSamples-1): break
        DataTD = ptUtils.sendToDevice(Data, TestDevice)
        TargetsTD = ptUtils.sendToDevice(Targets, TestDevice)

        Output = TestNet.forward(DataTD)
        Loss = LossFunc(Output, TargetsTD)
        ValLosses.append(Loss.item())

        # print(Output.size())
        # print(len(Targets))
        # print(Targets[0].size())
        # exit()

        if OutDir is not None:
            SetSize = Output.size(1) # unsqueezed
            for s in range(0, SetSize):
                # output GT camera information
                PoseList = Targets[1]
                Pose = PoseList[s]
                # convert from torch
                PosDict = Pose['position']
                for Comp in PosDict.keys():
                    PosDict[Comp] = float(PosDict[Comp].item())
                RotDict = Pose['rotation']
                for Comp in RotDict.keys():
                    RotDict[Comp] = float(RotDict[Comp].item())
                Pose['position'] = PosDict
                Pose['rotation'] = RotDict
                with open(os.path.join(OutDir, 'frame_{}_view_{}_pose.json').format(str(i).zfill(3), str(s).zfill(2)), 'w') as JSONFile:
                    JSONFile.write(json.dumps(Pose))
                
                CurrTargets = Targets[0][:, s, :, :, :].squeeze()
                CurrData = Data[:, s, :, :, :].squeeze()
                CurrOutput = Output[:, s, : ,:, :].squeeze()
                GTItems = SNCD.convertData(ptUtils.sendToDevice(CurrData, 'cpu'), ptUtils.sendToDevice([CurrTargets, None], 'cpu'))
                PredItems = SNCD.convertData(ptUtils.sendToDevice(CurrData, 'cpu'), ptUtils.sendToDevice([CurrOutput.detach(), None], 'cpu'), isMaskNOX=True)
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_color00.png').format(str(i).zfill(3), str(s).zfill(2)),
                            cv2.cvtColor(GTItems[0], cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_nox00_gt.png').format(str(i).zfill(3), str(s).zfill(2)),
                            cv2.cvtColor(GTItems[2], cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_nox00_pred.png').format(str(i).zfill(3), str(s).zfill(2)),
                            cv2.cvtColor(PredItems[2], cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_nox01_gt.png').format(str(i).zfill(3), str(s).zfill(2)),
                            cv2.cvtColor(GTItems[3], cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_nox01_pred.png').format(str(i).zfill(3), str(s).zfill(2)),
                            cv2.cvtColor(PredItems[3], cv2.COLOR_BGR2RGB))
                if len(PredItems) == 6:
                    cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_mask00_gt.png').format(str(i).zfill(3), str(s).zfill(2)), GTItems[4])
                    cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_mask00_pred.png').format(str(i).zfill(3), str(s).zfill(2)), PredItems[4])
                    cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_mask01_gt.png').format(str(i).zfill(3), str(s).zfill(2)), GTItems[5])
                    cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_mask01_pred.png').format(str(i).zfill(3), str(s).zfill(2)), PredItems[5])
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_color01_gt.png').format(str(i).zfill(3), str(s).zfill(2)),
                            cv2.cvtColor(GTItems[1], cv2.COLOR_BGR2RGB))
                if PredItems[1] is not None:
                    cv2.imwrite(os.path.join(OutDir, 'frame_{}_view_{}_color01_pred.png').format(str(i).zfill(3), str(s).zfill(2)),
                                cv2.cvtColor(PredItems[1], cv2.COLOR_BGR2RGB))

                # # calculate chamfer metric
                # EvalMetric = ChamferMetric(torch_device=TestDevice)
                # ValDists.append(EvalMetric.chamfer_dist_nocs(PredItems[2:4], GTItems[2:4]))

                # write out meta info (category_id/model_id/frame(view)_num for each output frame)
                if Args.dataset == 'MShapeNetCOCODataset':
                    modelid = TestDataLoader.dataset.ModelList[i]
                    catid = TestDataLoader.dataset.ModelDirPaths[modelid].split('/')[-1]
                    MetaInfo = '%s/%s/%s' % (catid, modelid, str(s).zfill(8))
                    # print(MetaInfo)
                    with open(ModelInfoPath, 'a+') as InfoFile:
                        InfoFile.write(MetaInfo + '\n')


        # Print stats
        Toc = ptUtils.getCurrentEpochTime()
        Elapsed = math.floor((Toc - Tic) * 1e-6)
        done = int(50 * (i + 1) / len(TestDataLoader))
        sys.stdout.write(('\r[{}>{}] test loss - {:.16f}, test dist - {:.8f}, elapsed - {}')
                         .format('=' * done, '-' * (50 - done), np.mean(np.asarray(ValLosses)), np.mean(np.asarray(ValDists)), 
                                 ptUtils.getTimeDur(Elapsed)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    
    # write out chamfer results
    # DistStr = '\n'.join(['{:.16f}'.format(Dist) for Dist in ValDists])
    # with open(os.path.join(OutDir, 'chamfer_results.txt'), 'w') as f:
    #     f.write(DistStr)
    # np.save(os.path.join(OutDir, 'chamfer_results'), np.asarray(ValDists))
    
    # plt.show()

    return ValLosses

def main(Args):
    if Args[0][0] == '@':
        ExpandedPath = ptUtils.expandTilde(Args[0][1:])
        Args[0] = '@' + ExpandedPath
    Config = mv_nxm_config.mv_nxm_config(Args)

    if Config.Args.arch not in ['SetSegNet', 'SetSegNetSkip', 'SetSegNetSkipPermEq']:
        raise RuntimeError('Unsupported architecture ' + Config.Args.arch)
    if Config.Args.dataset not in ['MShapeNetCOCODataset']:
        raise RuntimeError('Unsupported dataset ' + Config.Args.dataset)

    if Config.Args.seed is not None:
        ptUtils.seedRandom(Config.Args.seed)

    DataLimit = Config.Args.data_limit if Config.Args.data_limit > 0 else None
    ValDataLimit = Config.Args.val_data_limit if Config.Args.val_data_limit > 0 else None
    ValOnTrain = True if Config.Args.force_test_on_train else False
    if ValOnTrain:
        print(' [ WARN ]: Testing on train data. This should be used for debugging purposes only.')

    useVariableSetSize = Config.Args.enable_variable_set_size
    if useVariableSetSize:
        print('[ INFO ]: Using variable set sizes.')

    DeviceList, MainGPUID = ptUtils.setupGPUs(Config.Args.gpu)
    print('[ INFO ]: Using {} GPUs with IDs {}'.format(len(DeviceList), DeviceList))
    Device = ptUtils.setDevice(MainGPUID)

    if Config.Args.arch == 'SetSetSegNet':
        NOCSNet = SetSegNet.SetSegNet(n_classes=Config.nOutChannels, Args=Args, DataParallelDevs=DeviceList,
                                   withSkipConnections=False, enablePermEq=False)
    elif Config.Args.arch == 'SetSegNetSkip':
        NOCSNet = SetSegNet.SetSegNet(n_classes=Config.nOutChannels, Args=Args, DataParallelDevs=DeviceList,
                                      withSkipConnections=True, enablePermEq=False)
    elif Config.Args.arch == 'SetSegNetSkipPermEq':
        NOCSNet = SetSegNet.SetSegNet(n_classes=Config.nOutChannels, Args=Args, DataParallelDevs=DeviceList,
                                withSkipConnections=True, enablePermEq=True)

    if Config.Args.mode == 'train':
        if Config.Args.dataset == 'MShapeNetCOCODataset':
            TrainData = MSNCD(root=NOCSNet.Config.Args.input_dir, train=True, download=True
                                                    , limit=DataLimit, category=Config.Args.category, loadMask=Config.WithMask, imgSize=Config.ImageSize, small=Config.Args.use_small_dataset
                                                                  , setSize=Config.Args.set_size, isVariableSetSize=Config.Args.enable_variable_set_size)
            ValData = MSNCD(root=NOCSNet.Config.Args.input_dir, train=False, download=True
                                                      , limit=ValDataLimit, category=Config.Args.category, loadMask=Config.WithMask, imgSize=Config.ImageSize, small=Config.Args.use_small_dataset
                                                                , setSize=Config.Args.set_size, isVariableSetSize=Config.Args.enable_variable_set_size)
            if Config.WithMask == False:
                print('[ INFO ]: Using L2Loss function.')
                LossFunc = MSNCD.L2Loss(HallucinatePeeledColor = Config.HallucinatePeeledColor)
            else:
                if Config.Args.loss == 'l2':
                    print('[ INFO ]: Using L2MaskLoss function.')
                    LossFunc = MSNCD.L2MaskLoss(HallucinatePeeledColor = Config.HallucinatePeeledColor)
        print('[ INFO ]: Training data has', len(TrainData), 'samples.')
        print('[ INFO ]: Validation data has', len(ValData), 'samples.')
        TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=NOCSNet.Config.Args.batch_size, shuffle=True, num_workers=4)
        ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=1, # Using the smallest batch size #NOCSNet.Config.Args.batch_size,
                                                      shuffle=True, num_workers=4)

        NOCSNet.fit(TrainDataLoader, Objective=LossFunc, TrainDevice=Device, ValDataLoader=ValDataLoader)
    elif Config.Args.mode == 'val':
        NOCSNet.loadCheckpoint()

        if Config.Args.dataset == 'MShapeNetCOCODataset':
            ValData = MSNCD(root=NOCSNet.Config.Args.input_dir, train=ValOnTrain, download=True
                                                      , limit=ValDataLimit, category=Config.Args.category, loadMask=Config.WithMask, imgSize=Config.ImageSize, small=Config.Args.use_small_dataset
                                                                , setSize=Config.Args.set_size, isVariableSetSize=False)
            if Config.WithMask == False:
                print('[ INFO ]: Using L2Loss function.')
                LossFunc = MSNCD.L2Loss(HallucinatePeeledColor = Config.HallucinatePeeledColor)
            else:
                if Config.Args.loss == 'l2':
                    print('[ INFO ]: Using L2MaskLoss function.')
                    LossFunc = MSNCD.L2MaskLoss(HallucinatePeeledColor = Config.HallucinatePeeledColor)

        ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=1, shuffle=False, num_workers=4)
        print('[ INFO ]: Validation data has', len(ValDataLoader), 'samples.')

        validate(Config.Args, LossFunc, ValDataLoader, NOCSNet, Device, OutDir=os.path.join(NOCSNet.ExptDirPath, 'ValResults'))
    elif Config.Args.mode == 'test':
        print('[ WARN ]: Not yet implemented.')

    return NOCSNet.LossHistory

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('[ ERR ]: Please use -h flag for usage instructions.')
        exit()

    main(sys.argv[1:])
