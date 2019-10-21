import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import json
import matplotlib.pyplot as plt
import os, sys, math, argparse, zipfile, glob, cv2, random, PIL, pickle
import time

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))
from tk3dv.ptTools import ptUtils
from torch import nn
from tk3dv.ptTools.loaders import CameraDataset
if torch.cuda.is_available():
    from tk3dv.extern.chamfer import ChamferDistance

# This is the basic loader that loads all data without any model ID separation of camera viewpoint knowledge
class ShapeNetCOCODataset(CameraDataset.CameraDataset):
    class L2Loss(nn.Module):
        def __init__(self, HallucinatePeeledColor=True):
            super().__init__()
            self.HallucinatePeeledColor = HallucinatePeeledColor

        def forward(self, output, target):
            return self.computeLoss(output, target)

        def computeLoss(self, output, target):
            if isinstance(output, list) or isinstance(output, tuple):
                OutNOX = output[0].clone().requires_grad_(True)
            else:
                OutNOX = output.clone().requires_grad_(True)
            if isinstance(target, list) or isinstance(target, tuple):
                TargetNOX = target[0]
            else:
                TargetNOX = target

            BatchSize = TargetNOX.size(0)
            nChannels = OutNOX.size(1)

            if nChannels == 8: # Mask is available but we skip it for loss computation
                NOXIdx = tuple(([0, 1, 2], [4, 5, 6]))
            elif nChannels == 11:  # Mask + PeeledColor available. We skip only mask for loss computation.
                NOXIdx = tuple(([0, 1, 2], [4, 5, 6], [8, 9, 10]))
            elif nChannels == 6: # No mask available
                NOXIdx = tuple(([0, 1, 2], [3, 4, 5]))
            elif nChannels == 9: # No mask available but peeled color is
                NOXIdx = tuple(([0, 1, 2], [3, 4, 5], [6, 7, 8]))
            else:
                raise RuntimeError('[ ERR ]: Unknown number of channels in input.')

            # TODO I think this could be much more efficient if we vectorize it (maybe not worth the time though)
            TotalLoss = 0
            for NOXIdRange in NOXIdx:
                DiffNorm = torch.norm(OutNOX[:, NOXIdRange, :, :] - TargetNOX[:, NOXIdRange, :, :], dim=1)  # Same size as WxH
                L2Loss = 0
                for i in range(0, BatchSize):
                    L2Loss += torch.mean(DiffNorm[i])
                TotalLoss += (L2Loss / BatchSize)

            TotalLoss /= len(NOXIdx)

            return TotalLoss

    class L2MaskLoss(nn.Module):
        Thresh = 0.7 # PARAM
        def __init__(self, Thresh=0.7, HallucinatePeeledColor=True, UseChamfer=False):
            super().__init__()
            self.MaskLoss = nn.BCELoss(size_average=True, reduce=True)
            self.Sigmoid = nn.Sigmoid()
            self.Thresh = Thresh
            self.HallucinatePeeledColor = HallucinatePeeledColor
            self.UseChamfer = UseChamfer
            if self.UseChamfer:
                self.ChamferDist = ChamferDistance()

        def forward(self, output, target):
            return self.computeLoss(output, target)

        def computeLoss(self, output, target):
            if isinstance(output, list) or isinstance(output, tuple):
                OutNOX = output[0].clone().requires_grad_(True)
            else:
                OutNOX = output.clone().requires_grad_(True)
            if isinstance(target, list) or isinstance(target, tuple):
                TargetNOX = target[0]
            else:
                TargetNOX = target

            BatchSize = TargetNOX.size(0)
            nChannels = OutNOX.size(1)
            if nChannels != 8 and nChannels != 11:
                print('[ WARN ]: nChannels is {}'.format(nChannels))
                raise RuntimeError('[ ERR ]: L2Mask loss can only be used when mask and/or Peeled color is loaded.')

            TotalLoss = 0
            Den = 2.0
            ChamferWeight = 10.0
            TotalLoss += self.computeMaskedL2Loss(OutNOX[:, 0:4, :, :], TargetNOX[:, 0:4, :, :])
            TotalLoss += self.computeMaskedL2Loss(OutNOX[:, 4:8, :, :], TargetNOX[:, 4:8, :, :])
            if self.UseChamfer:
                TotalLoss += ChamferWeight*self.computeMaskedChamferLoss(OutNOX[:, 0:4, :, :], TargetNOX[:, 0:4, :, :])
                TotalLoss += ChamferWeight*self.computeMaskedChamferLoss(OutNOX[:, 4:8, :, :], TargetNOX[:, 4:8, :, :])

            if self.HallucinatePeeledColor and nChannels == 11:
                Den = 3.0

                # Use same masked L2 loss but use the mask for the second NOX layer, for potential improvements on the mask
                OutNOXPC = torch.cat((OutNOX[:, 8:11, :, :], OutNOX[:, 7:8, :, :]), 1) # The 7:8 makes sure not to squeeze
                TargetNOXPC = torch.cat((TargetNOX[:, 8:11, :, :], TargetNOX[:, 7:8, :, :]), 1) # The 7:8 makes sure not to squeeze
                TotalLoss += self.computeMaskedL2Loss(OutNOXPC, TargetNOXPC)

                # # OR ######## Plain L2 loss, probably makes learning harder
                # DiffNorm = torch.norm(OutNOX[:, 8:, :, :] - TargetNOX[:, 8:, :, :], dim=1)  # Same size as WxH
                # L2Loss = 0
                # for i in range(0, BatchSize):
                #     L2Loss += torch.mean(DiffNorm[i])
                # TotalLoss += (L2Loss / BatchSize)

            TotalLoss /= Den

            return  TotalLoss

        def computeMaskedL2Loss(self, output, target):
            BatchSize = target.size(0)
            TargetMask = target[:, -1, :, :]
            OutMask = output[:, -1, :, :].clone().requires_grad_(True)
            OutMask = self.Sigmoid(OutMask)

            MaskLoss = self.MaskLoss(OutMask, TargetMask)

            TargetNOCS = target[:, :-1, :, :]
            OutNOCS = output[:, :-1, :, :].clone().requires_grad_(True)

            DiffNorm = torch.norm(OutNOCS - TargetNOCS, dim=1)  # Same size as WxH
            MaskedDiffNorm = torch.where(OutMask > self.Thresh, DiffNorm,
                                         torch.zeros(DiffNorm.size(), device=DiffNorm.device))
            NOCSLoss = 0
            for i in range(0, BatchSize):
                nNonZero = torch.nonzero(MaskedDiffNorm[i]).size(0)
                if nNonZero > 0:
                    NOCSLoss += torch.sum(MaskedDiffNorm[i]) / nNonZero
                else:
                    NOCSLoss += torch.mean(DiffNorm[i])

            MaskWeight = 0.7 # PARAM
            NOCSWeight = 0.3 # PARAM
            Loss = (MaskWeight*MaskLoss) + (NOCSWeight*(NOCSLoss / BatchSize))
            return Loss

        def computeMaskedChamferLoss(self, output, target):
            with torch.autograd.set_detect_anomaly(True):
                BatchSize = target.size(0)
                TargetMask = target[:, -1, :, :].clone()

                TargetNOCS = target[:, :-1, :, :].clone()
                OutNOCS = output[:, :-1, :, :].clone().requires_grad_(True)

                # use points within the ground truth mask
                TargetCond = TargetMask > self.Thresh

                # must go through each batch element individually since masked point clouds are different sizes
                ChamferLoss = 0
                for i in range(BatchSize): 
                    # get masked point cloud for this NOCS map
                    TargetPCL = TargetNOCS[i].view(3, -1).t()
                    TargetPCL = TargetPCL[TargetCond[i].view(-1)]
                    OutPCL = OutNOCS[i].clone().requires_grad_(True).view(3, -1).t()
                    OutPCL = OutPCL[TargetCond[i].view(-1)]

                    # expects [B, N, 3] size inputs
                    dist1, dist2 = self.ChamferDist(TargetPCL.unsqueeze(0), OutPCL.unsqueeze(0))
                    CurLoss = torch.mean(dist1) + torch.mean(dist2)
                    ChamferLoss = ChamferLoss + CurLoss

                return ChamferLoss / BatchSize


    class ChamferMaskLoss(nn.Module):
        def __init__(self,  Thresh=0.7, HallucinatePeeledColor=True):
            super().__init__()
            self.MaskLoss = nn.BCELoss(size_average=True, reduce=True)
            self.Sigmoid = nn.Sigmoid()
            self.Thresh = Thresh
            self.HallucinatePeeledColor = HallucinatePeeledColor
            self.ChamferDist = ChamferDistance()

        def forward(self, output, target):
            return self.computeLoss(output, target)

        def computeLoss(self, output, target):
            with torch.autograd.set_detect_anomaly(True):
                if isinstance(output, list) or isinstance(output, tuple):
                    OutNOX = output[0].clone().requires_grad_(True)
                else:
                    OutNOX = output.clone().requires_grad_(True)
                if isinstance(target, list) or isinstance(target, tuple):
                    TargetNOX = target[0]
                else:
                    TargetNOX = target

                nChannels = OutNOX.size(1)
                
                TotalLoss = 0
                Den = 2.0
                NOCSLoss = self.computeMaskedChamferLoss(OutNOX[:, 0:4, :, :], TargetNOX[:, 0:4, :, :])
                NOXLoss = self.computeMaskedChamferLoss(OutNOX[:, 4:8, :, :], TargetNOX[:, 4:8, :, :])
                PeelLoss = 0.0
                if self.HallucinatePeeledColor and nChannels == 11:
                    Den = 3.0

                    # Use same masked L2 loss but use the mask for the second NOX layer, for potential improvements on the mask
                    OutNOXPC = torch.cat((OutNOX[:, 8:11, :, :], OutNOX[:, 7:8, :, :]),
                                         1)  # The 7:8 makes sure not to squeeze
                    TargetNOXPC = torch.cat((TargetNOX[:, 8:11, :, :], TargetNOX[:, 7:8, :, :]),
                                            1)  # The 7:8 makes sure not to squeeze
                    PeelLoss = self.computeMaskedL2Loss(OutNOXPC, TargetNOXPC)

                # print('NOCSLoss:', NOCSLoss)
                # print('NOXLoss:', NOXLoss)
                # print('PeelLoss:', PeelLoss)
                NOCSWeight = 1.0
                NOXWeight = 1.0
                PeelWeight = 0.03
                TotalLoss = (NOCSWeight*NOCSLoss) + (NOXWeight*NOXLoss) +(PeelWeight* PeelLoss)
                TotalLoss = TotalLoss / Den

                return TotalLoss

        def computeMaskedChamferLoss(self, output, target):
            with torch.autograd.set_detect_anomaly(True):
                BatchSize = target.size(0)
                TargetMask = target[:, -1, :, :].clone()
                OutMask = output[:, -1, :, :].clone().requires_grad_(True)
                OutMask = self.Sigmoid(OutMask)

                MaskLoss = self.MaskLoss(OutMask, TargetMask)

                TargetNOCS = target[:, :-1, :, :].clone()
                OutNOCS = output[:, :-1, :, :].clone().requires_grad_(True)

                # use points within the predicted mask unless predicted mask is empty (then use ground truth)
                TargetCond = TargetMask > self.Thresh
                # OutCond = OutMask > self.Thresh

                # must go through each batch element individually since masked point clouds are different sizes
                ChamferLoss = 0
                for i in range(BatchSize):
                    # nNonZero = torch.nonzero(OutCond[i]).size(0)
                    # print(nNonZero)
                    # if nNonZero == 0:
                    #    OutCond[i] = TargetCond[i]

                    # get masked point cloud for this NOCS map
                    TargetPCL = TargetNOCS[i].view(3, -1).t()
                    TargetPCL = TargetPCL[TargetCond[i].view(-1)]
                    #TargetPCL = torch.index_select(TargetPCL, 0, torch.Tensor([0, 100, 200]).long().cuda()) #TargetPCL[0:2, :]
                    OutPCL = OutNOCS[i].clone().requires_grad_(True).view(3, -1).t()
                    OutPCL = OutPCL[TargetCond[i].view(-1)] #OutCond[i].clone().view(-1)] # have to clone not to affect original mask
                    #OutPCL = torch.index_select(OutPCL, 0, torch.Tensor([0, 100, 200]).long().cuda()) #, OutPCL[[0,500]:]
                    # print(TargetPCL)
                    # print(OutPCL)
                    # OutPCL.register_hook(print)

                    # expects [B, N, 3] size inputs
                    dist1, dist2 = self.ChamferDist(TargetPCL.unsqueeze(0), OutPCL.unsqueeze(0))
                    CurLoss = torch.mean(dist1) + torch.mean(dist2)
                    # print(CurLoss)
                    ChamferLoss = ChamferLoss + CurLoss

                MaskWeight = 2.0 # PARAM
                ChamferWeight = 1.0 # PARAM
                Loss = (MaskWeight * MaskLoss) + (ChamferWeight*(ChamferLoss / BatchSize))

                # print('MaskLoss:', MaskLoss)
                # print('ChamferLoss:', ChamferLoss)

                return Loss

        def computeMaskedL2Loss(self, output, target):
            BatchSize = target.size(0)
            TargetMask = target[:, -1, :, :]
            OutMask = output[:, -1, :, :].clone().requires_grad_(True)
            OutMask = self.Sigmoid(OutMask)

            MaskLoss = self.MaskLoss(OutMask, TargetMask)

            TargetNOCS = target[:, :-1, :, :]
            OutNOCS = output[:, :-1, :, :].clone().requires_grad_(True)

            DiffNorm = torch.norm(OutNOCS - TargetNOCS, dim=1)  # Same size as WxH
            MaskedDiffNorm = torch.where(OutMask > self.Thresh, DiffNorm,
                                         torch.zeros(DiffNorm.size(), device=DiffNorm.device))
            NOCSLoss = 0
            for i in range(0, BatchSize):
                nNonZero = torch.nonzero(MaskedDiffNorm[i]).size(0)
                if nNonZero > 0:
                    NOCSLoss += torch.sum(MaskedDiffNorm[i]) / nNonZero
                else:
                    NOCSLoss += torch.mean(DiffNorm[i])

            Loss = MaskLoss + (NOCSLoss / BatchSize)
            return Loss

    def __init__(self, root, train=True, download=True, transform=None, target_transform=None, trialrun=False,
                 imgSize=(640, 480), limit=None, loadMemory=False, loadMask=False, category='cars', peel=True, small=False):
        self.isSmall = small
        self.FileName = 'shapenetcoco_dataset_v1.zip'
        if self.isSmall:
            print('[ WARN ]: Using SMALL DATSET. Use only for debugging.')
            self.FileName = 'shapenetcoco_dataset_v1_small.zip'
        self.DataURL = 'https://storage.googleapis.com/stanford_share/Datasets/shapenetcoco_dataset_v1.zip'
        self.LoadMask = loadMask
        self.Category = category
        self.Synsets = {'all' : '**', 'airplanes' : '02691156', 'cars' : '02958343', 'chairs' : '03001627'}
        self.isPeel = peel
        if self.LoadMask:
            print('[ INFO ]: Loading masks in ShapeNetCOCODataset.')
        if self.isPeel == False:
            print('[ WARN ]: Depth peeled targets disabled. Will not load NOXRay maps and peeled color.')

        super().init(root, train, download, transform, target_transform, trialrun, imgSize, limit, loadMemory)
        self.loadData()

    def loadData(self):
        # First check if unzipped directory exists
        DatasetDir = os.path.join(ptUtils.expandTilde(self.DataDir), os.path.splitext(self.FileName)[0])
        if os.path.exists(DatasetDir) == False:
            DataPath = os.path.join(ptUtils.expandTilde(self.DataDir), self.FileName)
            if os.path.exists(DataPath) == False:
                if self.isDownload:
                    print('[ INFO ]: Downloading', DataPath)
                    ptUtils.downloadFile(self.DataURL, DataPath)

                if os.path.exists(DataPath) == False: # Not downloaded
                    raise RuntimeError('Specified data path does not exist: ' + DataPath)
            # Unzip
            with zipfile.ZipFile(DataPath, 'r') as File2Unzip:
                print('[ INFO ]: Unzipping.')
                File2Unzip.extractall(ptUtils.expandTilde(self.DataDir))

        AppendDir = '**'
        if self.Synsets[self.Category] is not None:
            AppendDir = self.Synsets[self.Category]

        print('[ INFO ]: Loading dataset subset:', self.Category)
        FilesPath = os.path.join(DatasetDir, 'val/')
        if self.isTrainData:
            FilesPath = os.path.join(DatasetDir, 'train/')

        if AppendDir == '**':
            GlobCache = os.path.join(FilesPath, 'all_glob.cache')
        FilesPath = FilesPath + AppendDir
        if AppendDir != '**':
            GlobCache = os.path.join(FilesPath, 'glob.cache')

        if os.path.exists(GlobCache):
            print('[ INFO ]: Loading from glob cache.')
            with open(GlobCache, 'rb') as fp:
                self.Color00 = pickle.load(fp)
                self.Color01 = pickle.load(fp)
                self.NOX00 = pickle.load(fp)
                self.NOX01 = pickle.load(fp)
                self.Pose = pickle.load(fp)
        else:
            print('[ INFO ]: Saving to glob cache.')
            self.Color00 = glob.glob(FilesPath + '/**/*Color_00*')
            self.Color01 = glob.glob(FilesPath + '/**/*Color_01*')
            self.NOX00 = glob.glob(FilesPath + '/**/*NOXRayTL_00*')
            self.NOX01 = glob.glob(FilesPath + '/**/*NOXRayTL_01*')
            self.Pose = glob.glob(FilesPath + '/**/*CameraPose*')

            self.Color00.sort()
            self.Color01.sort()
            self.NOX00.sort()
            self.NOX01.sort()
            self.Pose.sort()

            with open(GlobCache, 'wb') as fp:
                pickle.dump(self.Color00, fp)
                pickle.dump(self.Color01, fp)
                pickle.dump(self.NOX00, fp)
                pickle.dump(self.NOX01, fp)
                pickle.dump(self.Pose, fp)

        if not self.Color00 or not self.Color01 or not self.NOX00 or not self.NOX01 or not self.Pose:
            raise RuntimeError('[ ERR ]: Unable to glob data.')

        if len(self.Color00) == 0:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        if len(self.Color00) != len(self.Color01) or len(self.Color01) != len(self.NOX00) or len(self.NOX00) != len(
                self.NOX01) or len(self.NOX01) != len(self.Pose):
            raise RuntimeError('[ ERR ]: Data corrupted. Sizes do not match')

        print('[ INFO ]: Found {} items in dataset.'.format(len(self)))
        DatasetLength = self.DataLimit
        self.Color00 = self.Color00[:DatasetLength]
        self.Color01 = self.Color01[:DatasetLength]
        self.NOX00 = self.NOX00[:DatasetLength]
        self.NOX01 = self.NOX01[:DatasetLength]
        self.Pose = self.Pose[:DatasetLength]

        # meta info for outputting results (category_id/model_id/frame_num)
        self.ModelPaths = ['/'.join(Path.split('/')[-3:-1]) for Path in self.Color00]
        self.ModelPaths = [PrePath + '/' + Path.split('/')[-1].split('_')[1] for (PrePath, Path) in zip(self.ModelPaths, self.Color00)]

    def __len__(self):
        return len(self.Color00)

    @staticmethod
    def createMask(NOCSMap):
        LocNOCS = NOCSMap.type(torch.FloatTensor)

        Norm = torch.norm(LocNOCS, dim=0)
        ZeroT = torch.zeros((1, LocNOCS.size(1), LocNOCS.size(2)), dtype=LocNOCS.dtype, device=LocNOCS.device)
        OneT = torch.ones((1, LocNOCS.size(1), LocNOCS.size(2)), dtype=LocNOCS.dtype, device=LocNOCS.device) * 255 # Range: 0, 255
        Mask = torch.where(Norm >= 441.6729, ZeroT, OneT)  # 441.6729 == sqrt(3 * 255^2)
        return Mask.type(torch.FloatTensor)

    def loadImages(self, idx):
        Color00 = self.imread_rgb_torch(self.Color00[idx], Size=self.ImageSize).type(torch.FloatTensor)
        Color01 = self.imread_rgb_torch(self.Color01[idx], Size=self.ImageSize).type(torch.FloatTensor)
        if self.Transform is not None:
            Color00 = self.Transform(Color00)
            Color01 = self.Transform(Color01)

        NOX00 = self.imread_rgb_torch(self.NOX00[idx], Size=self.ImageSize).type(torch.FloatTensor)
        NOX01 = self.imread_rgb_torch(self.NOX01[idx], Size=self.ImageSize).type(torch.FloatTensor)
        if self.TargetTransform is not None:
            NOX00 = self.TargetTransform(NOX00)
            NOX01 = self.TargetTransform(NOX01)
        with open(self.Pose[idx], 'r') as JSONFile:
            Pose = json.load(JSONFile)

        if self.LoadMask:
            NOX00 = torch.cat((NOX00, self.createMask(NOX00)), 0).type(torch.FloatTensor)
            NOX01 = torch.cat((NOX01, self.createMask(NOX01)), 0).type(torch.FloatTensor)

        # Convert range to 0.0 - 1.0
        Color00 /= 255.0
        Color01 /= 255.0
        NOX00 /= 255.0
        NOX01 /= 255.0

        NOX = torch.cat((NOX00, NOX01), 0)

        return Color00, Color01, NOX, Pose

    def __getitem__(self, idx):
        RGB, PeeledColor, NOX, Pose = self.loadImages(idx)
        if self.isPeel:
            NOX = torch.cat((NOX, PeeledColor), 0) # Attach peeled color (3 channels) to end of NOX
        else:
            NOX = NOX[0:3, :, :] # Only NOCS, nothing else
        return RGB, (NOX, Pose)

    @staticmethod
    def applyMask(NOX, Thresh):
        # Input (only torch): 4-channels where first 3 are NOCS, last is mask
        # Output (numpy): 3-channels where the mask is used to mask out the NOCS
        if NOX.size()[0] != 4:
            raise RuntimeError('[ ERR ]: Input needs to be a 4 channel image.')

        NOCS = NOX[:3, :, :]
        Mask = NOX[3, :, :]

        MaskProb = ptUtils.torch2np(torch.squeeze(F.sigmoid(Mask)))
        Masked = np.uint8((MaskProb > Thresh) * 255)
        MaskedNOCS = ptUtils.torch2np(torch.squeeze(NOCS))
        MaskedNOCS[MaskProb <= Thresh] = 255

        return MaskedNOCS, Masked

    @staticmethod
    def convertData(RGB, Targets, isMaskNOX=False):
        NOX, Pose = Targets
        # Convert range to 0-255
        NOX = NOX.squeeze() * 255
        Color00 = ptUtils.torch2np(RGB.squeeze()).squeeze() * 255

        nChannels = NOX.size(0)
        PeeledColor = None
        if nChannels == 9 or nChannels == 11:
            PeeledColor = NOX[-3:, :, :]

        if PeeledColor is not None:
            PeeledColor = PeeledColor.squeeze()
            Color01 = ptUtils.torch2np(PeeledColor).squeeze()
        else:
            Color01 = None

        if nChannels == 8 or nChannels == 11:
            if isMaskNOX:
                NOX00, Mask00 = ShapeNetCOCODataset.applyMask(torch.squeeze(NOX[0:4, :, :]), Thresh=ShapeNetCOCODataset.L2MaskLoss.Thresh)
                NOX01, Mask01 = ShapeNetCOCODataset.applyMask(torch.squeeze(NOX[4:8, :, :]), Thresh=ShapeNetCOCODataset.L2MaskLoss.Thresh)
                if Color01 is not None:
                    Color01, DummyMask = ShapeNetCOCODataset.applyMask((torch.cat((NOX[-3:, :, :], NOX[7:8, :, :]), 0)).squeeze(), Thresh=ShapeNetCOCODataset.L2MaskLoss.Thresh)
            else:
                NOX00 = ptUtils.torch2np(NOX[0:3, :, :]).squeeze()
                Mask00 = ptUtils.torch2np(NOX[3, :, :]).squeeze()
                NOX01 = ptUtils.torch2np(NOX[4:7, :, :]).squeeze()
                Mask01 = ptUtils.torch2np(NOX[7, :, :]).squeeze()
            return Color00, Color01, NOX00, NOX01, Mask00, Mask01
        elif nChannels == 6 or nChannels == 9:
            NOX00 = ptUtils.torch2np(NOX[0:3, :, :]).squeeze()
            NOX01 = ptUtils.torch2np(NOX[3:6, :, :]).squeeze()
            return Color00, Color01, NOX00, NOX01

    def convertItem(self, idx):
        RGB, Targets = self[idx]

        return self.convertData(RGB, Targets)

    @staticmethod
    def saveData(Items, OutPath='.'):
        for Ctr, I in enumerate(Items, 0):
            if len(I.shape) == 3:
                I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(OutPath, 'item' + str(Ctr).zfill(4) + '.png'), I)


    def saveItem(self, idx, OutPath='.'):
        Items = Data.convertItem(idx)
        self.saveData(Items, OutPath)

    @staticmethod
    def imread_rgb_torch(Path, Size=None): # Use only for loading RGB images
        ImageCV = cv2.imread(Path, -1)
        # Discard 4th channel since we are loading as RGB
        if ImageCV.shape[-1] != 3:
            ImageCV = ImageCV[:, :, :3]

        ImageCV = cv2.cvtColor(ImageCV, cv2.COLOR_BGR2RGB)
        if Size is not None:
            ImageCV = cv2.resize(ImageCV, dsize=Size, interpolation=cv2.INTER_NEAREST)
        Image = ptUtils.np2torch(ImageCV) # Range: 0-255

        return Image

    def visualizeRandom(self, nSamples=10):
        nColsPerSample = 6 if self.LoadMask else 4
        nCols = nColsPerSample
        nRows = min(nSamples, 10)
        nTot = nRows * nCols

        RandIdx = random.sample(range(1, len(self)), nRows)

        fig = plt.figure(0, figsize=(10,2))
        ColCtr = 0
        for RowCtr, RandId in enumerate(RandIdx):
            if self.LoadMask:
                Color00, Color01, NOX00, NOX01, Mask00, Mask01 = self.convertItem(RandId)
            else:
                Color00, Color01, NOX00, NOX01 = self.convertItem(RandId)

            DivFact = 1
            if np.max(Color00) > 1:
                DivFact = 255

            plt.subplot(nRows, nCols, ColCtr + 1)
            plt.xticks([]), plt.yticks([]), plt.grid(False)
            plt.imshow(Color00 / DivFact)
            plt.subplot(nRows, nCols, ColCtr + 2)
            plt.xticks([]), plt.yticks([]), plt.grid(False)
            plt.imshow(Color01 / DivFact)
            plt.subplot(nRows, nCols, ColCtr + 3)
            plt.xticks([]), plt.yticks([]), plt.grid(False)
            plt.imshow(NOX00 / DivFact)
            plt.subplot(nRows, nCols, ColCtr + 4)
            plt.xticks([]), plt.yticks([]), plt.grid(False)
            plt.imshow(NOX01 / DivFact)
            if self.LoadMask:
                plt.subplot(nRows, nCols, ColCtr + 5)
                plt.xticks([]), plt.yticks([]), plt.grid(False)
                plt.imshow(Mask00, cmap='gray')
                plt.subplot(nRows, nCols, ColCtr + 6)
                plt.xticks([]), plt.yticks([]), plt.grid(False)
                plt.imshow(Mask01, cmap='gray')
            ColCtr = (RowCtr+1) * (nColsPerSample)
        plt.show()


Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--DataDir', help='Specify the location of the directory to download and store CameraDataset', required=True)
Parser.add_argument('-c', '--category', help='Name of category.', required=False, default='cars', choices=['cars', 'airplanes', 'chairs', 'all'])

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()

    Data = ShapeNetCOCODataset(root=Args.DataDir, train=True, download=True, loadMask=True, category=Args.category, imgSize=(320, 240), small=False)
    # Data.saveItem(1)
    Data.visualizeRandom(5)
    exit()

    if Data.LoadMask:
        if torch.cuda.is_available() == False:
            LossUnitTest = ShapeNetCOCODataset.L2MaskLoss(0.7)
        else:
            LossUnitTest = ShapeNetCOCODataset.ChamferMaskLoss(0.7)
    else:
        LossUnitTest = ShapeNetCOCODataset.L2Loss()
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=10, shuffle=True, num_workers=4)
    for i, (Data, Targets) in enumerate(DataLoader, 0):  # Get each batch
        # DataTD = ptUtils.sendToDevice(Targets, 'cpu')
        Loss = LossUnitTest(Targets, Targets)
        print('Loss:', Loss.item())
        # Loss.backward()
