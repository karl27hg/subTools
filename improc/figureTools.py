import cv2, math
import numpy as np
import random as rd
import matplotlib.pyplot as plt

__version__ = 0.2

def createRandomCircle(imgSize = (300, 300), shapeColor = (255, 255, 255), thickness = 1):
    """절단 방지, 이미지 경계 겹침 방지, 랜덤으로 원을 생성
    args
        imgSize : tuple (width, height)
        shapeColor : tuple (R, G, B)
        thickness : int
    return
        circle img : cv2.np.ndarray
    """
    # 바깥 원 정보 받음
    centerPt, radius = _getRandomCircleInfo(imgSize, thickness)
    centerX, centerY = centerPt
    # 원을 그림
    imgShape = imgSize + (3,)
    circleImg = np.zeros(imgShape, np.uint8)
    cvShapeColor = shapeColor[::-1]
    cv2.circle(circleImg, centerPt, radius, cvShapeColor, thickness)
    return cv2.cvtColor(circleImg, cv2.COLOR_BGR2RGB)

def createRandomRotatedRect(imgSize = (300, 300), rectColor = (255, 255, 255), thickness = 1):
    """짤림 방지된 회전된 사각형 생성 원을 기반으로 그림
    args
        imgSize : tuple (width, height)
        rectColor : tuple (R, G, B)
        thickness : int
    return
        rotated rectange img : cv2.np.ndarray
    """
    # 바깥 원 정보 받음
    centerPt, radius = _getRandomCircleInfo(imgSize, thickness)
    centerX, centerY = centerPt
    # 회전 시킬 각도
    rotDeg = rd.randrange(0, 90, 15)
    # 각도로 사각형의 start, end point를 결정
    rectDeg = rd.randrange(10, 90, 10)
    endX = int(round(centerX + radius * math.cos(rectDeg)))
    endY = int(round(centerY + radius * math.sin(rectDeg)))
    endPt = (endX, endY)
    startX = int(round(centerX - radius * math.cos(rectDeg)))
    startY = int(round(centerY - radius * math.sin(rectDeg)))
    startPt = (startX, startY)
    # 사각형 그리기
    imgShape = imgSize + (3,)
    rectImg = np.zeros(imgShape, np.uint8)
    cvRectColor = rectColor[::-1]
    cv2.rectangle(rectImg, startPt, endPt, cvRectColor, thickness)
    # 사각형 회전
    rotMatrix = cv2.getRotationMatrix2D(centerPt, rotDeg, 1)
    rotRectImg = cv2.warpAffine(rectImg, rotMatrix, imgSize)
    return cv2.cvtColor(rotRectImg, cv2.COLOR_BGR2RGB)

def createRandomRotatedSharpRect(imgSize = (300, 300), rectColor = (255, 255, 255), thickness = 1):
    """createRandomRotatedRect의 경우 경계가 굵어 질 수 록 꼭지점 부분이 라운드가 발생하는 것을 방지하는 사각형
    args
        imgSize : tuple (width, height)
        rectColor : drawing color (R, G, B)
        thickness : int
    return
        rotated Sharp Rectangle image : cv2.np.ndarray
    """
    if -1 == thickness:
        return createRandomRotatedRect(imgSize, rectColor, thickness)
    # 바깥 원 정보 받음
    centerPt, radius = _getRandomCircleInfo(imgSize, thickness)
    centerX, centerY = centerPt
    # 사각형 그리기
    rectDeg = rd.randrange(10, 90, 10)
    endX = int(round(centerX + radius * math.cos(rectDeg)))
    endY = int(round(centerY + radius * math.sin(rectDeg)))
    endPt = (endX, endY)
    startX = int(round(centerX - radius * math.cos(rectDeg)))
    startY = int(round(centerY - radius * math.sin(rectDeg)))
    startPt = (startX, startY)
    imgShape = imgSize + (3,)
    rectImg = np.zeros(imgShape, np.uint8)
    cvRectColor = rectColor[::-1]
    cv2.rectangle(rectImg, startPt, endPt, cvRectColor, -1)
    # 속 사각형 그리기
    startInPt = (startX + thickness, startY + thickness)
    endInPt = (endX - thickness, endY - thickness)
    # 속 사각형을 그릴 수 없는 경우 재귀호출함
    if startInPt[0] <= startPt[0] or startInPt[1] <= startPt[1]:
        createRandomRotatedSharpRect(imgSize, rectColor, thickness)
    if endInPt[0] >= endInPt[0] or endInPt[1] >= endPt[1]:
        createRandomRotatedSharpRect(imgSize, rectColor, thickness)
    cv2.rectangle(rectImg, startInPt, endInPt, (0,0,0), -1)
    return cv2.cvtColor(rectImg, cv2.COLOR_BGR2RGB)

def _getRandomCircleInfo(imgSize, thickness, inPadding = None):
    """
    args
        imgSize : tuple (width, height)
        thickness : int
        inPadding : custom inPadding : int
    return
        centerPoint : tuple (center x, center y)
        radius : int
    """
    padding = (thickness // 2 + 1)
    shortLength = min(imgSize)
    maxRadius = shortLength // 2 - 2 * padding
    minPadding = inPadding if inPadding is not None and inPadding >= padding else padding
    radius = rd.randrange(minPadding, maxRadius)
    # 회전 중심점 좌표의 범위 구하기
    minCenterPos = radius + padding
    maxCenterPosX = imgSize[0] - radius - padding
    maxCenterPosY = imgSize[1] - radius - padding
    if minCenterPos >= maxCenterPosX or minCenterPos >= maxCenterPosY:
        return _getRandomCircleInfo(imgSize, thickness)
    centerX = rd.randrange(minCenterPos, maxCenterPosX)
    centerY = rd.randrange(minCenterPos, maxCenterPosY)
    centerPt = (centerX, centerY)
    return centerPt, radius

def overDrawImg(bgImg, fgImg, imgFrame, thresholdRange = (50, 255)):
    """bgImg 위에 fgImg를 imgFrame 모양에 맞게 붙임
    args
        bgImg : back ground image : np.ndarray
        fgImg : front image : np.ndarray
        imgFrame : mask image : np.ndarray
        thresholdRange : threshold range : tuple (min, max)
    return
        retImg : result image : np.ndarray
    """
    grayImg = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2GRAY)
    thresholdMin, thresholdMax = thresholdRange
    _, shapeMask = cv2.threshold(grayImg, thresholdMin, thresholdMax, cv2.THRESH_BINARY)
    shapeMask_inv = cv2.bitwise_not(shapeMask)
    tmpImg = bgImg.copy()
    tmpBgImg = cv2.bitwise_and(tmpImg, tmpImg, mask = shapeMask_inv)
    tmpImg = fgImg.copy()
    tmpFgImg = cv2.bitwise_and(tmpImg, tmpImg, mask = shapeMask)
    retImg = cv2.add(tmpBgImg, tmpFgImg)
    return retImg

def createRandomTriangle(imgSize = (300, 300), figColor = (255, 255, 255), thickness = 1):
    """
    args
        imgSize : tuple (width, height)
        figColor : tuple (R, G, B)
        thickness : int
    return
        triangle image : np.ndarray (R, G, B)
    """
    padding = (thickness // 2 + 1) if thickness > 0 else 1
    shortLength = min(imgSize)
    minPos = padding
    maxPosX = imgSize[0] - padding
    maxPosY = imgSize[1] - padding
    ptList = []
    for itPt in range(3):
        ptList.append([rd.randrange(minPos, maxPosX), rd.randrange(minPos, maxPosY)])
    ptNp = np.array(ptList, np.int32)
    # 폴리곤 그리기로 삼각형 그리기
    imgShape = imgSize + (3,)
    triangleImg = np.zeros(imgShape, np.uint8)
    cvFigColor = figColor[::-1]
    if 0 > thickness:
        cv2.fillPoly(triangleImg, [ptNp], cvFigColor)
    else:
        cv2.polylines(triangleImg, [ptNp], True, cvFigColor, thickness)
    return cv2.cvtColor(triangleImg, cv2.COLOR_BGR2RGB)

def getRandomColor(inten = 51):
    """랜덤 RGB 값을 반환함
    args
        inten : int [0, 255]
    return 
        color : tuple (R, G, B) or (B, G, R)
    """
    return (rd.randrange(0, 256, inten),
            rd.randrange(0, 256, inten),
            rd.randrange(0, 256, inten))

if __name__ == "__main__":
    pass