import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os, sys, math, argparse, zipfile, glob, cv2, random

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '.'))
import ptUtils

sys.path.append(os.path.join(FileDirPath, '../..'))
from ptTools.loaders import CameraDataset

class HandRigDataset(CameraDataset.CameraDataset):
    def __init__(self, root, train=True, download=True, transform=None, target_transform=None, trialrun=False, imgSize=(640, 480), limit=None, loadMemory=False, cameraIdx=-1, loadMask=False):
        self.CameraIdx = cameraIdx
        self.FileName = 'hand_rig_dataset_v1.zip'
        self.DataURL = 'https://storage.googleapis.com/stanford_share/Datasets/hand_rig_dataset_v1.zip'
        self.LoadMask = loadMask

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

        FilesPath = os.path.join(DatasetDir, 'val/')
        if self.isTrainData:
            FilesPath = os.path.join(DatasetDir, 'train/')

        CameraIdxStr = '*'
        if self.CameraIdx >= 0 and self.CameraIdx <= 4:
            CameraIdxStr = str(self.CameraIdx).zfill(2)

        print('[ INFO ]: Loading data for camera {}.'.format(CameraIdxStr))

        self.RGBList = glob.glob(FilesPath + '/**/frame_*_cam_' + CameraIdxStr + '_color.*')
        self.RGBList.sort()
        self.InstMaskList = glob.glob(FilesPath + '/**/frame_*_cam_' + CameraIdxStr + '_binmask.*')
        self.InstMaskList.sort()
        self.NOCSList = glob.glob(FilesPath + '/**/frame_*_cam_' + CameraIdxStr + '_nocs.*')
        self.NOCSList.sort()

        if self.RGBList is None or self.InstMaskList is None or self.NOCSList is None:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        if len(self.RGBList) != len(self.InstMaskList) or len(self.InstMaskList) != len(self.NOCSList):
            raise RuntimeError('[ ERR ]: Data corrupted. Sizes do not match')

        print('[ INFO ]: Found {} items in dataset.'.format(len(self)))
        DatasetLength = self.DataLimit
        self.RGBList = self.RGBList[:DatasetLength]
        self.InstMaskList = self.InstMaskList[:DatasetLength]
        self.NOCSList = self.NOCSList[:DatasetLength]

    def transform(self, RGB, NOCS, Mask=None):
        if self.Transform is not None:
            RGB = self.Transform(RGB)
        if self.TargetTransform is not None:
            NOCS = self.TargetTransform(NOCS)
            if Mask is not None:
                Mask = self.TargetTransform(Mask)

        if self.LoadMask:
            return RGB, NOCS, Mask
        return RGB, NOCS

    def loadImages(self, RGBFile, NOCSFile):
        RGB = self.imread_rgb_torch(RGBFile)
        NOCS = self.imread_rgb_torch(NOCSFile)

        if self.LoadMask:
            LocNOCS = NOCS
            LocNOCS = LocNOCS.type(torch.FloatTensor)

            Norm = torch.norm(LocNOCS, dim=0)
            ZeroT = torch.zeros((1, NOCS.size(1), NOCS.size(2)), dtype=LocNOCS.dtype, device=NOCS.device)
            OneT = torch.ones((1, NOCS.size(1), NOCS.size(2)), dtype=LocNOCS.dtype, device=NOCS.device)
            Mask = torch.where(Norm >= 441.6729, ZeroT, OneT) # 441.6729 == sqrt(3 * 255^2)

            # # TEMP, TESTING TODO FIXME
            # NOCS = self.imread_rgb_torch(NOCSFile).type(torch.FloatTensor) # NOT CORRECT TESTING
            RGB, NOCS, Mask = self.transform(RGB, NOCS, Mask)
            return RGB, NOCS.type(torch.FloatTensor), Mask.type(torch.FloatTensor)

        RGB, NOCS = self.transform(RGB, NOCS)
        return RGB, NOCS.type(torch.FloatTensor)

    def __getitem__(self, idx):
        if self.LoadMask:
            RGB, NOCS, Mask = self.loadImages(self.RGBList[idx], self.NOCSList[idx])
            NOCSMask = torch.cat((NOCS, Mask), 0).type(torch.FloatTensor)

            return RGB, NOCSMask

        RGB, NOCS = self.loadImages(self.RGBList[idx], self.NOCSList[idx])

        return RGB, NOCS

    @staticmethod
    def imread_rgb_torch(Path, Size=None): # Use only for loading RGB images
        ImageCV = cv2.imread(Path, -1)
        # Discard 4th channel since we are loading as RGB
        if ImageCV.shape[-1] != 3:
            ImageCV = ImageCV[:, :, :3]
        ImageCV = cv2.cvtColor(ImageCV, cv2.COLOR_BGR2RGB)
        if Size is not None:
            ImageCV = cv2.resize(ImageCV, dsize=Size, interpolation=cv2.INTER_NEAREST)
        Image = ptUtils.np2torch(ImageCV)
        return Image

Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--DataDir', help='Specify the location of the directory to download and store CameraDataset', required=True)

if __name__ == '__main__':
    Args = Parser.parse_args()

    NormalizeTrans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0., 0., 0.), (1., 1., 1.))]
    )
    Data = HandRigDataset(root=Args.DataDir, train=False, download=True, cameraIdx=-1)#, transform=NormalizeTrans)
    Data.visualizeRandom(4)
