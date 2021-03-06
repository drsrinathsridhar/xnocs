import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import sys, os, cv2, math, json

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '../../models'))
import SegNet
from tk3dv.ptTools import ptUtils

sys.path.append(os.path.join(FileDirPath, '../loaders'))
import ShapeNetCOCODataset

sys.path.append(os.path.join(FileDirPath, './configs'))
import nxm_config

sys.path.append(os.path.join(FileDirPath, '../eval'))
from chamfer_metric import ChamferMetric

def validate(Args, LossFunc, TestDataLoader, Net, TestDevice, OutDir=None):
    TestNet = Net.to(TestDevice)
    nSamples = min(Args.test_samples, len(TestDataLoader))
    print('[ INFO ]: Testing on', nSamples, 'samples')

    if os.path.exists(OutDir) == False:
        os.makedirs(OutDir)

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

        if OutDir is not None:
            # ImageNP = np.uint8((np.squeeze(ptUtils.torch2np(torch.squeeze(Data)))) * 255)
            # cv2.imwrite(os.path.join(OutDir, 'frame_{}_color00.png').format(str(i).zfill(3)),
            #             cv2.cvtColor(ImageNP, cv2.COLOR_BGR2RGB))

            # output GT camera information
            _, Pose = ptUtils.sendToDevice(Targets, 'cpu')
            # convert from torch
            PosDict = Pose['position']
            for Comp in PosDict.keys():
                PosDict[Comp] = float(PosDict[Comp].item())
            RotDict = Pose['rotation']
            for Comp in RotDict.keys():
                RotDict[Comp] = float(RotDict[Comp].item())
            Pose['position'] = PosDict
            Pose['rotation'] = RotDict
            with open(os.path.join(OutDir, 'frame_{}_pose.json').format(str(i).zfill(3)), 'w') as JSONFile:
                JSONFile.write(json.dumps(Pose))

            GTItems = ShapeNetCOCODataset.ShapeNetCOCODataset.convertData(ptUtils.sendToDevice(Data, 'cpu'), ptUtils.sendToDevice(Targets, 'cpu'))
            PredItems = ShapeNetCOCODataset.ShapeNetCOCODataset.convertData(ptUtils.sendToDevice(Data, 'cpu'), ptUtils.sendToDevice([Output.detach(), None], 'cpu'), isMaskNOX=True)
            cv2.imwrite(os.path.join(OutDir, 'frame_{}_color00.png').format(str(i).zfill(3)),
                        cv2.cvtColor(GTItems[0], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(OutDir, 'frame_{}_nox00_gt.png').format(str(i).zfill(3)),
                        cv2.cvtColor(GTItems[2], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(OutDir, 'frame_{}_nox00_pred.png').format(str(i).zfill(3)),
                        cv2.cvtColor(PredItems[2], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(OutDir, 'frame_{}_nox01_gt.png').format(str(i).zfill(3)),
                        cv2.cvtColor(GTItems[3], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(OutDir, 'frame_{}_nox01_pred.png').format(str(i).zfill(3)),
                        cv2.cvtColor(PredItems[3], cv2.COLOR_BGR2RGB))
            if len(PredItems) == 6:
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_mask00_gt.png').format(str(i).zfill(3)), GTItems[4])
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_mask00_pred.png').format(str(i).zfill(3)), PredItems[4])
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_mask01_gt.png').format(str(i).zfill(3)), GTItems[5])
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_mask01_pred.png').format(str(i).zfill(3)), PredItems[5])
            cv2.imwrite(os.path.join(OutDir, 'frame_{}_color01_gt.png').format(str(i).zfill(3)),
                        cv2.cvtColor(GTItems[1], cv2.COLOR_BGR2RGB))
            if PredItems[1] is not None:
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_color01_pred.png').format(str(i).zfill(3)),
                            cv2.cvtColor(PredItems[1], cv2.COLOR_BGR2RGB))

            # calculate chamfer metric
            # EvalMetric = ChamferMetric(torch_device=TestDevice)
            # ValDists.append(EvalMetric.chamfer_dist_nocs(PredItems[2:4], GTItems[2:4]))

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

    # write out meta info (category_id/model_id/frame_num for each output frame)
    if Args.dataset == 'ShapeNetCOCODataset':
        MetaInfo = '\n'.join(TestDataLoader.dataset.ModelPaths)
        with open(os.path.join(OutDir, 'model_info.txt'), 'w') as InfoFile:
            InfoFile.write(MetaInfo)
    
    # plt.show()

    return ValLosses

def main(Args):
    if Args[0][0] == '@':
        ExpandedPath = ptUtils.expandTilde(Args[0][1:])
        Args[0] = '@' + ExpandedPath
    Config = nxm_config.nxm_config(Args)

    if Config.Args.arch not in ['SegNet', 'SegNetSkip']:
        raise RuntimeError('Unsupported architecture ' + Config.Args.arch)
    if Config.Args.dataset not in ['ShapeNetCOCODataset']:
        raise RuntimeError('Unsupported dataset ' + Config.Args.dataset)

    if Config.Args.seed is not None:
        ptUtils.seedRandom(Config.Args.seed)

    DataLimit = Config.Args.data_limit if Config.Args.data_limit > 0 else None
    ValDataLimit = Config.Args.val_data_limit if Config.Args.val_data_limit > 0 else None
    ValOnTrain = True if Config.Args.force_test_on_train else False
    if ValOnTrain:
        print(' [ WARN ]: Testing on train data. This should be used for debugging purposes only.')

    DeviceList, MainGPUID = ptUtils.setupGPUs(Config.Args.gpu)
    print('[ INFO ]: Using {} GPUs with IDs {}'.format(len(DeviceList), DeviceList))
    Device = ptUtils.setDevice(MainGPUID)

    if Config.Args.arch == 'SegNet':
        NOCSNet = SegNet.SegNet(n_classes=Config.nOutChannels, Args=Args, DataParallelDevs=DeviceList)
    elif Config.Args.arch == 'SegNetSkip':
        NOCSNet = SegNet.SegNet(n_classes=Config.nOutChannels, Args=Args, DataParallelDevs=DeviceList,
                                withSkipConnections=True)

    if Config.Args.mode == 'train':
        if Config.Args.dataset == 'ShapeNetCOCODataset':
            TrainData = ShapeNetCOCODataset.ShapeNetCOCODataset(root=NOCSNet.Config.Args.input_dir, train=True, download=True
                                                    , limit=DataLimit, category=Config.Args.category, loadMask=Config.WithMask, imgSize=Config.ImageSize, small=Config.Args.use_small_dataset)
            ValData = ShapeNetCOCODataset.ShapeNetCOCODataset(root=NOCSNet.Config.Args.input_dir, train=False, download=True
                                                      , limit=ValDataLimit, category=Config.Args.category, loadMask=Config.WithMask, imgSize=Config.ImageSize, small=Config.Args.use_small_dataset)
            if Config.WithMask == False:
                print('[ INFO ]: Using L2Loss function.')
                LossFunc = ShapeNetCOCODataset.ShapeNetCOCODataset.L2Loss(HallucinatePeeledColor = Config.HallucinatePeeledColor)
            else:
                if Config.Args.loss == 'l2':
                    print('[ INFO ]: Using L2MaskLoss function.')
                    LossFunc = ShapeNetCOCODataset.ShapeNetCOCODataset.L2MaskLoss(HallucinatePeeledColor = Config.HallucinatePeeledColor)
                else:
                    print('[ INFO ]: Using ChamferMaskLoss function.')
                    LossFunc = ShapeNetCOCODataset.ShapeNetCOCODataset.ChamferMaskLoss(HallucinatePeeledColor = Config.HallucinatePeeledColor)
        print('[ INFO ]: Training data has', len(TrainData), 'samples.')
        print('[ INFO ]: Validation data has', len(ValData), 'samples.')
        TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=NOCSNet.Config.Args.batch_size, shuffle=True, num_workers=4)
        ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=1, # Using the smallest batch size #NOCSNet.Config.Args.batch_size,
                                                      shuffle=True, num_workers=4)

        NOCSNet.fit(TrainDataLoader, Objective=LossFunc, TrainDevice=Device, ValDataLoader=ValDataLoader)
    elif Config.Args.mode == 'val':
        NOCSNet.loadCheckpoint()

        if Config.Args.dataset == 'ShapeNetCOCODataset':
            ValData = ShapeNetCOCODataset.ShapeNetCOCODataset(root=NOCSNet.Config.Args.input_dir, train=ValOnTrain, download=True
                                                     , limit=DataLimit, category=Config.Args.category, loadMask=Config.WithMask, imgSize=Config.ImageSize, small=Config.Args.use_small_dataset)
            if Config.WithMask == False:
                print('[ INFO ]: Using L2Loss function.')
                LossFunc = ShapeNetCOCODataset.ShapeNetCOCODataset.L2Loss(HallucinatePeeledColor = Config.HallucinatePeeledColor)
            else:
                if Config.Args.loss == 'l2':
                    print('[ INFO ]: Using L2MaskLoss function.')
                    LossFunc = ShapeNetCOCODataset.ShapeNetCOCODataset.L2MaskLoss(HallucinatePeeledColor = Config.HallucinatePeeledColor)
                else:
                    print('[ INFO ]: Using ChamferMaskLoss function.')
                    LossFunc = ShapeNetCOCODataset.ShapeNetCOCODataset.ChamferMaskLoss(HallucinatePeeledColor = Config.HallucinatePeeledColor)

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
