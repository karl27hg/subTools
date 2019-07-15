import cv2
import os
import numpy as np

__version__ = 0.1

def maintainRateResize(srcImg, wantSize):
    """이미지의 비율을 유지한 크기변경
    """
    pass

def noCropRotateImg(srcImg, rotatedAngle):
    """삭제되는 영역 없이 회전시킴
    """
    assert 90 > rotatedAngle and 0 <= rotatedAngle
    tmpImg = np.copy(srcImg)
    imgFrameSize = tmpImg.shape[:2]
    rotatedRadians = np.radians(rotatedAngle)
    # 잘리지 않을 만한 크기 계산
    centerY, centerX = imgFrameSize
    centerY, centerX = centerY // 2, centerX // 2
    pass

if __name__ == "__main__":
    os.chdir("./sampleData")
    sampleFilePath = os.path.join(
        os.getcwd(), 
        "2D_miku_V4X.jpg"
        )
    import matplotlib.pyplot as plt
    srcImg = cv2.imread(sampleFilePath)
    noCropRotateImg(srcImg, 30)
    plt.imshow(cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB))
    # plt.show()