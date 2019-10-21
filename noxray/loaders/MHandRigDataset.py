import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os, sys, math, argparse, zipfile, glob, cv2, random

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))
from ptTools import ptUtils
import MCOCOCarsDataset

# This is the advanced multi-view data loader that loads multiple images from each model ID
# Each item from this dataset is a tensor of size [SetSize, Channels, W, H] where setsize can be pre-specified of randomized (seed appropriately for reproducible results)
class MHandRigDataset(MCOCOCarsDataset.MCOCOCarsDataset):
    def __init__(self, root, train=True, download=True, transform=None, target_transform=None, trialrun=False, imgSize=(640, 480), limit=None, loadMemory=False, loadMask=False, setSize=None, isVariableSetSize=False, RandomizeViews=False, seed=0):
        self.FileName = 'hand_rig_dataset_v1.zip'
        self.DataURL = 'https://storage.googleapis.com/stanford_share/Datasets/hand_rig_dataset_v1.zip'
        self.LoadMask = loadMask
        self.SetSize = setSize
        self.isVariableSetSize = isVariableSetSize
        self.RandomizeViews = RandomizeViews
        self.Seed = seed
        if self.LoadMask:
            print('[ INFO ]: Loading masks in {}.'.format(self.__class__.__name__))
        if self.isVariableSetSize:
            print('[ INFO ]: Loading variable set sizes.')

        super().init(root, train, download, transform, target_transform, trialrun, imgSize, limit, loadMemory)
        self.loadData()
        random.seed(seed)

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

        self.DataPath = os.path.join(DatasetDir, 'val/')
        if self.isTrainData:
            self.DataPath = os.path.join(DatasetDir, 'train/')

        CameraIdxStr = str(0).zfill(2) # Using camera index 0 to find all frame IDs
        self.FrameList = glob.glob(self.DataPath + '/**/frame_*_cam_' + CameraIdxStr + '_color.png')
        self.FrameList.sort()
        for i in range(len(self.FrameList)):
            DirName = os.path.basename(os.path.dirname(self.FrameList[i]))
            FileName = os.path.basename(self.FrameList[i]).replace('_cam_00_color.png', '')
            self.FrameList[i] = os.path.join(DirName, FileName)
        # print(self.FrameList[0])

        if self.FrameList is None:
            raise RuntimeError('[ ERR ]: Unable to glob data.')
        if len(self.FrameList) == 0:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        print('[ INFO ]: Found {} items in dataset.'.format(len(self)))

        DatasetLength = self.DataLimit
        self.FrameList = self.FrameList[:DatasetLength]

    def __len__(self):
        return len(self.FrameList)

    def __getitem__(self, idx):
        RGBSet, NOCSSet = self.loadFrame(self.FrameList[idx], withMask=self.LoadMask)

        return RGBSet, NOCSSet

    def loadFrame(self, FrameID, withMask=False):
        FramePrefix = os.path.join(self.DataPath, FrameID)

        RGBList = glob.glob(FramePrefix + '_cam*color*')
        NOCSList = glob.glob(FramePrefix + '_cam*nocs*')
        if RGBList is None or NOCSList is None:
            raise RuntimeError('[ ERR ]: Unable to glob data.')

        if len(RGBList) != len(NOCSList) or len(RGBList) == 0:
            raise RuntimeError('[ ERR ]: Data corrupted. Sizes do not match or there is not data in the path {}.'.format(self.DataPath))

        nMaxViews = len(RGBList)
        if nMaxViews == 0:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        RGBList.sort()
        NOCSList.sort()

        if self.SetSize is None:
            ReqSetSize = nMaxViews
        else:
            ReqSetSize = min(self.SetSize, nMaxViews)

        # Each item is a tensor of size [SetSize, Channels, W, H]
        RGBSetList = []
        NOCSSetList = []
        if self.isVariableSetSize:
            nViews = random.randint(1, ReqSetSize) # both inclusive
        else:
            nViews = ReqSetSize
        FileIdx = [i for i in range(0, nMaxViews)]
        if self.RandomizeViews:
            random.shuffle(FileIdx)
        FileIdx = FileIdx[:nViews]
        for i in FileIdx:
            if withMask:
                RGB, NOCS, Mask = self.loadImages(RGBList[i], NOCSList[i])
                NOCS = torch.cat((NOCS, Mask), 0).type(torch.FloatTensor) # NOCS + Mask actually
            else:
                RGB, NOCS = self.loadImages(RGBList[i], NOCSList[i])
            RGBSetList.append(RGB.unsqueeze(0))
            NOCSSetList.append(NOCS.unsqueeze(0))

        RGBSet = torch.cat(RGBSetList, dim=0) # [SetSize, Channels, W, H]
        NOCSSet = torch.cat(NOCSSetList, dim=0)  # [SetSize, Channels, W, H]

        return RGBSet, NOCSSet

Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--DataDir', help='Specify the location of the directory to download and store CameraDataset', required=True)

if __name__ == '__main__':
    Args = Parser.parse_args()

    NormalizeTrans = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.))
    ])
    Data = MHandRigDataset(root=Args.DataDir, train=True, download=True, loadMask=True, setSize=5, isVariableSetSize=True, RandomizeViews=True, seed=0, transform=NormalizeTrans)
    Data.visualizeRandom(10)

    if Data.LoadMask:
        LossUnitTest = MCOCOCarsDataset.MCOCOCarsDataset.NOCSMaskLoss(0.7)
    else:
        LossUnitTest = MCOCOCarsDataset.MCOCOCarsDataset.NOCSLoss()
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=1, shuffle=True, num_workers=0, collate_fn=MCOCOCarsDataset.MCOCOCarsDataset.collate_fn)
    for i, (Data, Targets) in enumerate(DataLoader, 0):  # Get each batch
        print('Data list length:', (len(Data)))
        print('Target list length:', (len(Targets)))

        Loss = LossUnitTest(Targets, Targets)
        print('Loss:', Loss.item())