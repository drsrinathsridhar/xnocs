import sys, os, argparse, random, cv2, math, PIL
from torchvision import transforms

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))

from tk3dv.ptTools import ptUtils

class nxm_config():
    def __init__(self, InputArgs=None):
        self.Parser = argparse.ArgumentParser(description='NOXRay maps prediction.', fromfile_prefix_chars='@')
        InputGroup = self.Parser.add_mutually_exclusive_group(required=True)
        InputGroup.add_argument('--mode', help='Operation mode.', choices=['train', 'val', 'test'])
        ArgGroup = self.Parser.add_argument_group()

        # Experiment control
        ArgGroup.add_argument('--dataset', help='Choose dataset.', choices=['ShapeNetCOCODataset'], default='ShapeNetCOCODataset')
        ArgGroup.add_argument('--category', help='For ShapeNetCOCODataset choose one or all categories.', choices=['cars', 'airplanes', 'chairs', 'all'], default='cars')
        ArgGroup.add_argument('--arch', help='Choose architecture.', choices=['SegNet', 'SegNetSkip'], default='SegNet')
        ArgGroup.add_argument('--no-mask', help='Choose to train and test without mask.', action='store_true')
        self.Parser.set_defaults(no_mask=False)
        ArgGroup.add_argument('--no-color-hallucinate', help='Choose to not hallucinate peeled color.', action='store_true')
        self.Parser.set_defaults(no_color_hallucinate=False)

        # Machine control
        ArgGroup.add_argument('--gpu', help='GPU ID(s) to use for training. Default is cuda:0, or cpu if no GPU is available. -1 will use all available GPUs.', default=[0], type=int, choices=range(-1,8), nargs='+')
        ArgGroup.add_argument('--seed', help='Select random seed to initialize torch, np.random, and random. Default is auto seed.', type=int)

        # DEBUG purposes
        ArgGroup.add_argument('--data-limit', help='Select how many samples to load from data. Default is all.', type=int, required=False, default=-1)
        ArgGroup.add_argument('--val-data-limit', help='Select how many samples to load from data for validation. Default is all.', type=int, required=False, default=-1)
        ArgGroup.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true')
        self.Parser.set_defaults(force_test_on_train=False)
        ArgGroup.add_argument('--test-samples', help='Number of samples to test on.', default=30, type=int)

        # Loss function
        ArgGroup.add_argument('--loss', help='When using mask, what loss to use?', choices=['l2', 'chamfer'], default='l2')
        ArgGroup.add_argument('--use-small-dataset', help='Use a small dataset for testing purposes. CAUTION: Use this for debugging only.', action='store_true')
        self.Parser.set_defaults(use_small_dataset=False)

        self.InputSize = (320, 240, 3)
        self.ImageSize = (self.InputSize[0], self.InputSize[1])
        # self.DatasetInputTrans = transforms.Compose([
        #                                         transforms.ToPILImage(),
        #                                         transforms.Resize((self.InputSize[1], self.InputSize[0]), interpolation=PIL.Image.NEAREST),
        #                                         transforms.ToTensor(),
        #                                      ])
        # self.DatasetNOXTrans = transforms.Compose([
        #                                         transforms.ToPILImage(),
        #                                         transforms.Resize((self.InputSize[1], self.InputSize[0]), interpolation=PIL.Image.NEAREST),
        #                                         transforms.ToTensor(),
        #                                      ])

        self.Args, _ = self.Parser.parse_known_args(InputArgs)
        if len(sys.argv) <= 1:
            self.Parser.print_help()
            exit()

        self.WithMask = not self.Args.no_mask
        self.HallucinatePeeledColor = not self.Args.no_color_hallucinate
        if self.HallucinatePeeledColor == False:
            if self.WithMask:
                self.nOutChannels = 8 # (3 NOX + 1 mask) * 2
            else:
                self.nOutChannels = 6 # (3 NOX) * 2
        else:
            if self.WithMask:
                self.nOutChannels = 11 # (3 NOX + 1 mask) * 2 + 3 Peeled Color
            else:
                self.nOutChannels = 9 # (3 NOX) * 2 + 3 Peeled Color

        ptUtils.printArgs(self.Args)

    def serialize(self, FilePath, isAppend=True):
        ptUtils.configSerialize(self.Args, FilePath, isAppend)