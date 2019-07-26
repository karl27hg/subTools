import numpy as np

class ColorPixelManager:
    """픽셀 값들을 관리하는 덩어리
    """
    def __init__(self, srcImg, isRgb = True):
        """
        args
            srcImg
            roi : tuple(start row, start col, end row, end col)
            isRgb : True => RGB, False => BGR
        """
        assert isinstance(srcImg, np.ndarray)
        self.rows, self.cols = srcImg.shape[:2]
        self.pixelDict = {}
        # for itR in tqdm(range(self.rows)):
        for itR in range(self.rows):
            for itC in range(self.cols):
                colorValue = [str(srcImg.item(itR, itC, itCh)) for itCh in range(srcImg.shape[2])]
                colorValueStr = ",".join(colorValue)
                pos = (itR, itC)
                hasColorValue = False
                if colorValueStr not in self.pixelDict.keys():
                    self.pixelDict[colorValueStr] = []
                self.pixelDict[colorValueStr].append(pos)

    def getPixelPosList(self, color, isRgb = True):
        """
        """
        tmpColor = None
        colorKey = None
        if isinstance(color, tuple):
            tmpColor = list(color)
        elif isinstance(color, list):
            tmpColor = color
        elif isinstance(color, str):
            colorKey = color
        else:
            return None
        if colorKey is None:
            colorStrList = [str(itVal) for itVal in tmpColor]
            colorKey = ",".join(colorStrList)
        if colorKey not in self.pixelDict.keys():
            return None
        return self.pixelDict[colorKey]
    
    def getPixelCount(self, color, isRgb = True):
        """특정한 픽셀을 카운트함
        args
        return
        """
        return len(self.getPixelPos(color, isRgb))


    def getKeys(self):
        return self.pixelDict.keys()

    def getMinMaxPixel(self):
        """분류된 픽셀중 최대값과 최소 값에 대한 값
        return
            str min
            str max
        """
        return (
            min(self.pixelDict.keys(), key=self._callValue),
            max(self.pixelDict.keys(), key=self._callValue)
        )
    
    def _callValue(self, dictKey):
        """getMinMaxPixel을 위한 콜백 함수
        """
        return len(self.pixelDict[dictKey])

# =======================================================
# 이미지를 메모리에 load를 해봄으로 이미지가 영상인지 확인함
def checkImageFiles(imgFilePathList):
    """이미지 파일이 로드가 가능한 것인지 확인하는 메서드
    원리는 그냥 PIL로 메모리에 올려보면서 예최가 발생하는지 확인함
    """
    from PIL import Image
    tmpImg = None
    for itIndex, itPath in enumerate(imgFilePathList):
        try:
            tmpImg = Image.open(itPath)
        except Exception as ex:
            print("Error : [{}] : {}".format(itIndex, itPath))
            print("{} : {}".format(type(ex).__name__, ex.__str__()))

def showImageFiles(imgPathList):
    """본래 디버그 용도로 만듬
    """
    # %matplotlib inlie
    import matplotlib.pyplot as plt
    count = len(imgPathList)
    r, c, = 1, 1
    while r * c < count:
        if r == c:
            c += 1
        else:
            r += 1
    oldParams = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [r*10, c*10]
    fig = plt.figure()
    subImgList = []
    for itNum, itImg in enumerate(imgPathList):
        num = itNum + 1
        a = fig.add_subplot(r, c, num)
        a.imshow(itImg)
        a.axis("off")
        a.set_title(itNum)
        subImgList.append(a)
    plt.show()
    plt.rcParams["figure.figsize"] = oldParams


if __name__ == "__main__":
    import os, cv2
    dirPath, _ = os.path.split( os.path.dirname(__file__))
    imgFileName = "2019062000671_0.jpg"
    imgFilePath = os.path.join(dirPath, "images", imgFileName)
    srcImg = cv2.imread(imgFilePath)
    srcImgPixData = ColorPixelManager(srcImg)
    print(srcImgPixData.getMinMaxPixel())
    # cv2.imshow("src image", srcImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
