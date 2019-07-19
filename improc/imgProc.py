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
