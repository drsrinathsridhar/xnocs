import cv2, argparse, sys, os, random, glob, math
import numpy as np

Parser = argparse.ArgumentParser(description='Create collage from shapenetcoco dataset.')
Parser.add_argument('--input-dir', help='Specify the input base directory containing the dataset.', required=True, default=None)
Parser.add_argument('--output-dir', help='Specify the output base directory.', required=False, default='./collage')
Parser.add_argument('--num-views', help='Specify the number of views. Will create as many different collage images.', default=10, type=int, required=False)
Parser.add_argument('--num-models', help='Specify the number of models to select in a collage. Will automatically space the images within the image.', default=12, type=int, required=False)
Parser.add_argument('--category', help='Choose category. Default is car', default='*', required=False, choices=['car', 'airplane', 'chair', 'all'])

CatDict = {'car' : '02958343', 'airplane': '02691156', 'chair' : '03001627'}
ImTypes = ['Color_00', 'Color_01', 'NOXRayTL_00', 'NOXRayTL_01']

def makeCollage(ImageList, MaxWidth=800):
    if ImageList is None:
        return None

    nImages = len(ImageList)
    if nImages == 0:
        return None
    if nImages == 1:
        return ImageList[0]

    # Assuming images are all same size or we will resize to same size as the first image
    Shape = ImageList[0].shape
    for Image in ImageList:
        if Shape[0] != Image.shape[0] or Shape[1] != Image.shape[1]:
            Image = cv2.resize(Image, Shape)

    nCols = math.ceil(math.sqrt(nImages))
    nRows = math.ceil(nImages / nCols)
    # print('nImages, Cols, Rows:', nImages, ',' ,nCols, ',', nRows)

    Collage = np.zeros((Shape[0]*nRows, Shape[1]*nCols, Shape[2]), np.uint8)
    for i in range(0, nImages):
        Row = math.floor(i / nCols)
        Col = i % nCols
        Collage[Row*Shape[0]:Row*Shape[0]+Shape[0], Col*Shape[1]:Col*Shape[1]+Shape[1], :] = ImageList[i].copy()

    if Collage.shape[1] > MaxWidth:
        Fact = Collage.shape[1] / MaxWidth
        NewHeight = round(Collage.shape[0] / Fact)
        Collage = cv2.resize(Collage, (MaxWidth, NewHeight))

    return Collage


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    Args, _ = Parser.parse_known_args(sys.argv[1:])

    print('[ INFO ]: Making {} collages with {} models. Dataset base directory is {}, output is {}, category is {}'.format(Args.num_views, Args.num_models, Args.input_dir, Args.output_dir, Args.category))

    AllCats = ['car', 'airplane', 'chair']
    if Args.category is not '*' and Args.category is not 'all':
        AllCats = [Args.category]

    for Cat in AllCats:
        BasePath = os.path.join(Args.input_dir, 'train', CatDict[Cat])
        DirList = glob.glob(BasePath + '/*/', recursive=False)
        OutPath = os.path.join(Args.output_dir, Cat)
        if os.path.exists(OutPath) is False:
            os.makedirs(OutPath)
        # Randomize
        random.shuffle(DirList)

        for Type in ImTypes:
            for i in range(0, Args.num_views): # For each collage image
                ImageList = []
                for j in range(0, Args.num_models):
                    ImPath = os.path.join(DirList[j], 'frame_{}_{}.png'.format(str(i).zfill(8), Type))
                    ImageList.append(cv2.imread(ImPath, -1))

                Collage = makeCollage(ImageList, MaxWidth=1000)
                cv2.imwrite(os.path.join(OutPath, '{}_view_{}_{}.png'.format(Cat, i, Type)), Collage)
