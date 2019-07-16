import cv2
import os
import numpy as np

__version__ = 0.1

def _getResizeFilter(isSrcImgGreater):
    """축소/확대에 따른 플러그인 값을 반환함
    args
        isSrcImgGreater : true => Zoom out
    return
        cv2.flag
    """
    assert type(isSrcImgGreater) is bool
    return cv2.INTER_AREA if isSrcImgGreater else cv2.INTER_LINEAR

def _getImgResizeOneLength(srcImg, wantSize, isRgb=True):
    """width, height 중 하나의 비율만 가지고 resize 해주는 hide 메서드
    args
        srcImg : np.ndarray
        wantSize = tuple(width, height) and ( (width, 0) or (0, height) )
        isRgb : bool True => rgb, False => bgr
    return
        dstImg : np.ndarray
    """
    dstImg = None
    if isinstance(srcImg, np.ndarray):
        srcHeight, srcWidth, _ = srcImg.shape
        tmpImg = cv2.cvtColor(srcImg, cv2.COLOR_RGB2BGR) if isRgb else srcImg.copy()
        if isinstance(wantSize, tuple) and 2 == len(wantSize):
            wantWidth, wantHeight = wantSize
            # height 기준으로 크기 변경
            if 0 == wantWidth:
                resizeFlag = _getResizeFilter(srcHeight > wantHeight)
                dstWidth = (srcWidth * wantHeight) // srcHeight
                dstSize = (dstWidth, wantHeight)
                dstImg = cv2.resize(tmpImg, dsize=dstSize, interpolation=resizeFlag)
            # width 기준으로 크기 변경
            elif 0 == wantHeight:
                resizeFlag = _getResizeFilter(srcWidth > wantWidth)
                dstHeight = (srcHeight * wantWidth) // srcWidth
                dstSize = (wantWidth, dstHeight)
                dstImg = cv2.resize(tmpImg, dsize=dstSize, interpolation=resizeFlag)
    return cv2.cvtColor(dstImg, cv2.COLOR_BGR2RGB) if isRgb else dstImg

def maintainRateResize(srcImg, wantSize = (300, 300), isRgb = True):
    """이미지의 비율을 유지한 크기변경
    args
        srcImg : np.ndarray
        wantSize : tuple(width, height)
    return
        dstImg : np.ndarray
    """
    dstImg = None
    if isinstance(srcImg, np.ndarray):
        srcHeight, srcWidth, _ = srcImg.shape
        if isinstance(wantSize, tuple):
            wantWidth, wantHeight = wantSize
            # 특정 길이를 기준으로 특정 길이를 늘이려고 할 경우
            if 0 == wantWidth or 0 == wantHeight:
                dstImg = _getImgResizeOneLength(srcImg, wantSize, isRgb)
            # 원본의 화면비율과 상관없이 변경하는 경우
            # padding을 하여 원하는 이미지 크기와 비율을 맞춰줌
            else:
                if srcWidth > srcHeight:
                    tmpHeight = (srcHeight * wantWidth) // srcWidth
                    # height에 padding이 필요로 하는 경우
                    if tmpHeight < wantHeight:
                        deltaHeight = wantHeight - tmpHeight
                        halfDeltaHeight = deltaHeight // 2
                        # rgb와 bgr 순서를 자주 바꾸지 않도록 하기 위해 RGB인 경우 BGR로 미리 바꿈
                        workImg = cv2.cvtColor(srcImg, cv2.COLOR_RGB2BGR) if isRgb else srcImg.copy()
                        tmpImg = _getImgResizeOneLength(workImg, (wantWidth, 0), not isRgb)
                        # 홀수일 경우 윗쪽에 1 pixel 더 padding 함
                        oddHeight = 1 if 1 == deltaHeight % 2 else 0
                        dstImg = cv2.copyMakeBorder(tmpImg, halfDeltaHeight + oddHeight, halfDeltaHeight, 0, 0, cv2.BORDER_REPLICATE)
                        dstImg = cv2.cvtColor(dstImg, cv2.COLOR_BGR2RGB) if isRgb else dstImg
                    # width에 padding이 필요한 경우
                    elif tmpHeight > wantHeight:
                        tmpWidth = (srcWidth * wantHeight) // srcHeight
                        deltaWidth = wantWidth - tmpWidth
                        halfDeltaWidth = deltaWidth // 2
                        # rgb와 bgr 순서를 자주 바꾸지 않도록 하기 위해 RGB인 경우 BGR로 미리 바꿈
                        workImg = cv2.cvtColor(srcImg, cv2.COLOR_RGB2BGR) if isRgb else srcImg.copy()
                        tmpImg = _getImgResizeOneLength(workImg, (0, wantHeight), not isRgb)
                        # 홀수일 경우 왼쪽에 1 pixel 더 padding 함
                        oddWidth = 1 if 1 == deltaWidth % 2 else 0
                        dstImg = cv2.copyMakeBorder(tmpImg, 0, 0, halfDeltaWidth + oddWidth, halfDeltaWidth, cv2.BORDER_REPLICATE)
                        dstImg = cv2.cvtColor(dstImg, cv2.COLOR_BGR2RGB) if isRgb else dstImg
                    # padding이 필요 없는 경우
                    else:
                        dstImg = _getImgResizeOneLength(srcImg, (wantHeight, 0), isRgb)
                # 원본의 height가 더 큰 경우 혹은 같은 경우
                else:
                    tmpWidth = (srcWidth * wantHeight) // srcHeight
                    # width에 padding이 필요한 경우
                    if tmpWidth < wantWidth:
                        deltaWidth = wantWidth - tmpWidth
                        halfDeltaWidth = deltaWidth // 2
                        # rgb와 bgr 순서를 자주 바꾸지 않도록 하기 위해 RGB인 경우 BGR로 미리 바꿈
                        workImg = cv2.cvtColor(srcImg, cv2.COLOR_RGB2BGR) if isRgb else srcImg.copy()
                        tmpImg = _getImgResizeOneLength(workImg, (0, wantHeight), not isRgb)
                        # 홀수일 경우 왼쪽에 1 pixel 더 padding 함
                        oddWidth = 1 if 1 == deltaWidth % 2 else 0
                        dstImg = cv2.copyMakeBorder(tmpImg, 0, 0, halfDeltaWidth + oddWidth, halfDeltaWidth, cv2.BORDER_REPLICATE)
                        dstImg = cv2.cvtColor(dstImg, cv2.COLOR_BGR2RGB) if isRgb else dstImg
                    # height에 padding이 필요한 경우
                    elif tmpWidth > wantWidth:
                        tmpHeight = (srcHeight * wantWidth) // srcWidth
                        deltaHeight = wantHeight - tmpHeight
                        halfDeltaHeight = deltaHeight // 2
                        # rgb와 bgr 순서를 자주 바꾸지 않도록 하기 위해 RGB인 경우 BGR로 미리 바꿈
                        workImg = cv2.cvtColor(srcImg, cv2.COLOR_RGB2BGR) if isRgb else srcImg.copy()
                        tmpImg = _getImgResizeOneLength(workImg, (wantWidth, 0), not isRgb)
                        # 홀수일 경우 윗쪽에 1 pixel 더 padding 함
                        oddHeight = 1 if 1 == deltaHeight % 2 else 0
                        dstImg = cv2.copyMakeBorder(tmpImg, halfDeltaHeight + oddHeight, halfDeltaHeight, 0, 0, cv2.BORDER_REPLICATE)
                        dstImg = cv2.cvtColor(dstImg, cv2.COLOR_BGR2RGB) if isRgb else dstImg
                    # padding이 필요 없는 경우
                    else:
                        dstImg = _getImgResizeOneLength(srcImg, (0, wantHeight), isRgb)
    return dstImg

def noCropRotateImg(srcImg, rotatedAngle):
    """삭제되는 영역 없이 회전시킴 현재는 예각만 제대로 유호하다.
    45도를 넘어가고 가로세로 비율이 1:1이 아니면 문제가 발생할 수 있다.
    args
        srcImg : np.ndarray
        rotatedAngle : number
    return
        cropImg : np.ndarray not has hall
    """
    # assert 90. > rotatedAngle and 0. <= rotatedAngle
    halfPiCount = rotatedAngle // 90
    print("Angles:", rotatedAngle)
    tmpImg = srcImg.copy()
    imgFrameSize = tmpImg.shape[:2]
    rotatedRadians = np.radians(rotatedAngle -45)
    # 잘리지 않을 만한 크기 계산
    srcHeight, srcWidth = imgFrameSize
    centerY, centerX = srcHeight // 2, srcWidth // 2
    print("src center", centerX, centerY)
    maxRadius = np.sqrt(centerX**2 + centerY**2)
    boundaryOnY = centerX * np.tan(rotatedRadians)
    minRadius = np.sqrt(centerX**2 + boundaryOnY**2)
    smallWidth = int(round(srcWidth * minRadius / maxRadius))
    smallHeight = int(round(srcHeight * minRadius / maxRadius))
    # smallSize = (smallHeight, smallWidth)
    # print(maxRadius, minRadius)
    smallImg = _getImgResizeOneLength(tmpImg, (smallWidth, 0))
    print("small image size:", smallImg.shape)
    deltaWidth = srcWidth - smallWidth
    deltaHeight = srcHeight - smallHeight
    halfDeltaWidth = deltaWidth // 2
    halfDeltaHeight = deltaHeight // 2
    # padding 용 아직은 예각만 해당됨
    betaWidthRadians = np.arcsin(srcHeight / (2 * maxRadius))
    halfBigDeltaWidth = abs(int(maxRadius * np.cos(betaWidthRadians - rotatedRadians)))
    betaHeightRadians = np.arcsin(srcWidth / (2 * maxRadius))
    halfBigDeltaHeight = abs(int(maxRadius * np.cos(betaHeightRadians - rotatedRadians)))
    # bigImg = cv2.copyMakeBorder(smallImg, halfDeltaHeight, halfDeltaHeight, halfDeltaWidth, halfDeltaWidth, cv2.BORDER_REPLICATE)
    bigImg = cv2.copyMakeBorder(smallImg, halfBigDeltaHeight, halfBigDeltaHeight, halfBigDeltaWidth, halfBigDeltaWidth, cv2.BORDER_REPLICATE)
    print("big iamge size:", bigImg.shape)
    bigImgHeight, bigImgWidth = bigImg.shape[:2]
    bigCenterY, bigCenterX = bigImgHeight // 2, bigImgWidth // 2
    rotMatrix = cv2.getRotationMatrix2D((bigCenterX, bigCenterY), rotatedAngle, 1)
    rotatedImg = cv2.warpAffine(bigImg, rotMatrix, (bigImgWidth, bigImgHeight))
    # 자르기
    deltaCropY = (bigImgHeight - srcHeight) // 2
    deltaCropX = (bigImgWidth - srcWidth) // 2
    cropImg = rotatedImg[deltaCropY : bigImgHeight - deltaCropY, deltaCropX : bigImgWidth - deltaCropX]
    print("crop iamge size:", cropImg.shape)
    # return rotatedImg
    return cropImg


if __name__ == "__main__":
    # os.chdir("./sampleData")
    # sampleFilePath = os.path.join(
    #     os.getcwd(), 
    #     "2D_miku_V4X.jpg"
    #     )
    import matplotlib.pyplot as plt
    # srcImg = cv2.imread(sampleFilePath)
    srcImg = np.zeros((300, 300, 3), np.uint8)
    cv2.rectangle(srcImg, (0, 0), (300, 300), (255, 0, 0), thickness=-1)
    cv2.rectangle(srcImg, (3, 3), (297, 297), (255, 255, 255), thickness=-1)
    resultImg = noCropRotateImg(srcImg, 0)
    # plt.imshow(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB))
    # plt.show()
    cv2.imshow("rect", resultImg)
    import time
    for itAngle in range(0, 360, 10):
        # time.sleep(1)
        resultImg = noCropRotateImg(srcImg, itAngle)
        cv2.imshow("rect", resultImg)
        cv2.waitKey(1000)
    cv2.waitKey(0)
    cv2.destroyAllWindows()