import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import os, sys, math, argparse, zipfile, glob, cv2, random, pickle, json
from abc import abstractmethod

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))
from tk3dv.ptTools import ptUtils
from ShapeNetCOCODataset import ShapeNetCOCODataset as SNCD
from tk3dv.ptTools.loaders import CameraDataset


# This is the advanced multi-view data loader that loads multiple images from each model ID
# Each item from this dataset is a tensor of size [SetSize, Channels, W, H] where setsize can be pre-specified of randomized (seed appropriately for reproducible results)
class MShapeNetCOCODataset(CameraDataset.CameraDataset):
    def __init__(self, root, train=True, download=True, transform=None, target_transform=None, trialrun=False,
                 imgSize=(640, 480), limit=None, loadMemory=False, loadMask=False, category='cars', peel=True, small=False, setSize=None,
                 isVariableSetSize=False, RandomizeViews=False, seed=0):

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
            print('[ INFO ]: Loading masks in {}.'.format(self.__class__.__name__))
        if self.isPeel == False:
            print('[ WARN ]: Depth peeled targets disabled. Will not load NOXRay maps and peeled color.')

        self.SetSize = setSize
        self.isVariableSetSize = isVariableSetSize
        self.RandomizeViews = RandomizeViews
        self.Seed = seed
        if self.isVariableSetSize:
            print('[ INFO ]: Loading variable set sizes.')

        super().init(root, train, download, transform, target_transform, trialrun, imgSize, limit, loadMemory)
        self.loadData()
        random.seed(self.Seed)

    @abstractmethod
    def getModelList(DirPath):
        ModelList = [os.path.basename(x[0]) for x in os.walk(DirPath)]
        try:
            ModelList.remove(os.path.basename(DirPath)) # Remove base dir from walk
        except Exception as e:
            print('[ ERR ]: ModelList does not contain {}'.format(os.path.basename(DirPath)))
            print(ModelList)
            print('[ ERR ]', e)
        ModelList = list(filter(None, ModelList)) # Remove empty
        return ModelList

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

        print('[ INFO ]: Loading dataset subset:', self.Category)
        self.ModelList = []

        if self.Category == 'all': # Load all categories
            CatDirs = list(self.Synsets.values())
            CatDirs.remove('**')
        else:
            CatDirs = [self.Synsets[self.Category]]

        self.ModelDirPaths = dict()
        for CatDir in CatDirs:
            if CatDir is not None:
                DirPath = os.path.join(DatasetDir, 'val', CatDir)
                if self.isTrainData:
                    DirPath = os.path.join(DatasetDir, 'train', CatDir)
                DirCache = os.path.join(DirPath, 'dirs.cache')
                if os.path.exists(DirCache):
                    print('[ INFO ]: Loading from dirs cache:', DirCache)
                    with open(DirCache, 'rb') as fp:
                        ML = pickle.load(fp)
                else:
                    print('[ INFO ]: Saving to dirs cache:', DirCache)
                    ML = MShapeNetCOCODataset.getModelList(DirPath)
                    with open(DirCache, 'wb') as fp:
                        pickle.dump(ML, fp)
                self.ModelList.extend(ML)
                self.ModelDirPaths.update(dict(zip(ML, [DirPath] * len(ML))))
            else:
                raise RuntimeError('[ ERR ]: Unknown category passed.')

        self.ModelList.sort()
        # print('self.ModelList:', self.ModelList)

        if self.ModelList is None:
            raise RuntimeError('[ ERR ]: Unable to glob data.')
        if len(self.ModelList) == 0:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        print('[ INFO ]: Found {} items in dataset.'.format(len(self)))
        DatasetLength = self.DataLimit
        self.ModelList = self.ModelList[:DatasetLength]

    def __len__(self):
        return len(self.ModelList)

    def __getitem__(self, idx):
        RGBSet, PeeledColorSet, NOXSet, PoseSet = self.loadModelDir(self.ModelList[idx])
        if self.isPeel:
            NOXSet = torch.cat((NOXSet, PeeledColorSet), 1)  # Attach peeled color (3 channels) to end of NOX
        else:
            NOXSet = NOXSet[:, 0:3, :, :]  # Only NOCS, nothing else
        return RGBSet, (NOXSet, PoseSet)

    def loadModelDir(self, ModelID):
        DirPath = os.path.join(self.ModelDirPaths[ModelID], ModelID)

        Color00List = glob.glob(DirPath + '/*Color_00*')
        Color01List = glob.glob(DirPath + '/*Color_01*')
        NOX00List = glob.glob(DirPath + '/*NOXRayTL_00*')
        NOX01List = glob.glob(DirPath + '/*NOXRayTL_01*')
        PoseList = glob.glob(DirPath + '/*CameraPose*')
        Color00List.sort()
        Color01List.sort()
        NOX00List.sort()
        NOX01List.sort()
        PoseList.sort()

        if not Color00List or not Color01List or not NOX00List or not NOX01List or not PoseList:
            raise RuntimeError('[ ERR ]: Unable to glob data.')

        if len(Color00List) == 0:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        if len(Color00List) != len(Color01List) or len(Color01List) != len(NOX00List) or len(NOX00List) != len(NOX01List) or len(NOX01List) != len(PoseList):
            raise RuntimeError('[ ERR ]: Data corrupted. Sizes do not match')

        # print('[ INFO ]: Found {} items in {}.'.format(len(Color00), ModelID))
        nMaxViews = len(Color00List)
        if nMaxViews == 0:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        if self.SetSize is None:
            ReqSetSize = nMaxViews
        elif self.SetSize < 0:
            ReqSetSize = nMaxViews
        else:
            ReqSetSize = min(self.SetSize, nMaxViews)

        # Each item is a tensor of size [SetSize, Channels, W, H]
        Color00SetList = []
        Color01SetList = []
        NOXSetList = []
        PoseSetList = []
        if self.isVariableSetSize:
            nViews = random.randint(1, ReqSetSize) # both inclusive
        else:
            nViews = ReqSetSize
        FileIdx = [i for i in range(0, nMaxViews)]
        if self.RandomizeViews:
            random.shuffle(FileIdx)
        FileIdx = FileIdx[:nViews]
        for i in FileIdx:
            Color00, Color01, NOX, Pose = self.loadImages(Color00List[i], Color01List[i], NOX00List[i], NOX01List[i], PoseList[i])
            Color00SetList.append(Color00.unsqueeze(0))
            Color01SetList.append(Color01.unsqueeze(0))
            NOXSetList.append(NOX.unsqueeze(0))
            PoseSetList.append(Pose)

        Color00Set = torch.cat(Color00SetList, dim=0) # [SetSize, Channels, W, H]
        Color01Set = torch.cat(Color01SetList, dim=0)  # [SetSize, Channels, W, H]
        NOXSet = torch.cat(NOXSetList, dim=0)  # [SetSize, Channels, W, H]

        return Color00Set, Color01Set, NOXSet, PoseSetList

    def loadImages(self, Color00File, Color01File, NOX00File, NOX01File, PoseFile):
        Color00 = SNCD.imread_rgb_torch(Color00File, Size=self.ImageSize).type(torch.FloatTensor)
        Color01 = SNCD.imread_rgb_torch(Color01File, Size=self.ImageSize).type(torch.FloatTensor)
        if self.Transform is not None:
            Color00 = self.Transform(Color00)
            Color01 = self.Transform(Color01)

        NOX00 = SNCD.imread_rgb_torch(NOX00File, Size=self.ImageSize).type(torch.FloatTensor)
        NOX01 = SNCD.imread_rgb_torch(NOX01File, Size=self.ImageSize).type(torch.FloatTensor)
        if self.TargetTransform is not None:
            NOX00 = self.TargetTransform(NOX00)
            NOX01 = self.TargetTransform(NOX01)
        with open(PoseFile, 'r') as JSONFile:
            Pose = json.load(JSONFile)

        if self.LoadMask:
            NOX00 = torch.cat((NOX00, SNCD.createMask(NOX00)), 0).type(torch.FloatTensor)
            NOX01 = torch.cat((NOX01, SNCD.createMask(NOX01)), 0).type(torch.FloatTensor)

        # Convert range to 0.0 - 1.0
        Color00 /= 255.0
        Color01 /= 255.0
        NOX00 /= 255.0
        NOX01 /= 255.0

        NOX = torch.cat((NOX00, NOX01), 0)

        return Color00, Color01, NOX, Pose

    def convertData(self, RGBSet, TargetSets, isMaskNOX=False):
        SetSize = RGBSet.size(0)
        AllConvertedData = []
        for s in range(0, SetSize):
            ConvertedData = SNCD.convertData(RGBSet[s].squeeze(), (TargetSets[0][s].squeeze(), TargetSets[1][s]))
            AllConvertedData.append(ConvertedData)

        return AllConvertedData

    def convertItem(self, idx):
        RGBSet, TargetSets = self[idx]

        return self.convertData(RGBSet, TargetSets)

    def visualizeRandom(self, nSamples=2):
        nColsPerSample = 6 if self.LoadMask else 4
        nCols = nColsPerSample * self.SetSize
        nRows = min(nSamples, 10)
        nTot = nRows * nCols

        RandIdx = random.sample(range(1, len(self)), nRows)

        # fig = plt.figure(0, figsize=(16,2))
        ColCtr = 0
        for RowCtr, RandId in enumerate(RandIdx):
            AllConvertedData = self.convertItem(RandId)
            print('Sample Set Size:', len(AllConvertedData))
            for ViewCtr, ConvertedData in enumerate(AllConvertedData): # For each view
                if self.LoadMask:
                    Color00, Color01, NOX00, NOX01, Mask00, Mask01 = ConvertedData
                else:
                    Color00, Color01, NOX00, NOX01 = ConvertedData

                DivFact = 255 if (np.max(Color00) > 1) else 1
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
                ColCtr += nColsPerSample
            ColCtr = (RowCtr + 1) * (nCols) # Skip some cols in case there are variable number of views
        plt.show()

    @staticmethod
    def collate_fn(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        return (data, target)

    class L2Loss(SNCD.L2Loss):
        def __init__(self, HallucinatePeeledColor=True):
            super().__init__(HallucinatePeeledColor)

        def forward(self, output, target):
            return self.computeSetLoss(output, target)

        def computeSetLoss(self, output, target):
            NOXTarget = target[0]
            BatchSize = len(NOXTarget)
            L2Loss = 0
            for s in range(0, BatchSize):
                SetL2Loss = self.computeLoss(output[s].squeeze(), NOXTarget[s].squeeze()) # This loss is over the whole set (set replaces batch)
                L2Loss = L2Loss + SetL2Loss

            L2Loss = L2Loss / BatchSize
            return L2Loss

    class L2MaskLoss(SNCD.L2MaskLoss):
        def __init__(self, Thresh=0.7, HallucinatePeeledColor=True, UseChamfer=False):
            super().__init__(Thresh, HallucinatePeeledColor, UseChamfer)

        def forward(self, output, target):
            return self.computeSetLoss(output, target)

        def computeSetLoss(self, output, target):
            NOXTarget = target[0]
            BatchSize = len(NOXTarget)
            # print('BatchSize:', BatchSize)
            # print('target:', NOXTarget[0].size())
            # print('Output:', output[0].size())
            L2MaskLoss = 0.0
            for s in range(0, BatchSize):
                # print('output[s]:', output[s].size())
                # print('target[s]:', target[s].size())
                # Do not squeeze since computeLoss expects S x nChannels x W x H
                SetL2MaskLoss = self.computeLoss(output[s], NOXTarget[s]) # This loss is over the whole set (set replaces batch)
                L2MaskLoss = L2MaskLoss + SetL2MaskLoss

            L2MaskLoss = L2MaskLoss / BatchSize
            return L2MaskLoss

Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--DataDir', help='Specify the location of the directory to download and store MShapeNetCOCODataset.', required=True)
Parser.add_argument('-c', '--category', help='Name of category.', required=False, default='cars', choices=['cars', 'airplanes', 'chairs', 'all'])

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()

    Data = MShapeNetCOCODataset(root=Args.DataDir, train=True, download=True, loadMask=True,
                                setSize=5, isVariableSetSize=True, RandomizeViews=True, seed=0, category=Args.category, imgSize=(320, 240))
    Data.visualizeRandom(3)

    if Data.LoadMask:
        LossUnitTest = MShapeNetCOCODataset.L2MaskLoss(0.7)
    else:
        LossUnitTest = MShapeNetCOCODataset.L2Loss()
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=4, shuffle=True, num_workers=0, collate_fn=MShapeNetCOCODataset.collate_fn)
    for i, (Data, Targets) in enumerate(DataLoader, 0):  # Get each batch
        print('Data batch size:', len(Data))
        print('Data set size:', (Data[0].size(0)))

        Loss = LossUnitTest(Targets[0][0], Targets)
        print('Loss:', Loss.item())
