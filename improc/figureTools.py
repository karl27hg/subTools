import cv2, math
import numpy as np
import random as rd
import matplotlib.pyplot as plt

__version__ = 0.2

def createRandomCircle(imgSize = (299, 299), shapeColor = (255, 255, 255), thickness = 1):
    """절단 방지, 이미지 경계 겹침 방지, 랜덤으로 원을 생성
    args
        imgSize : tuple (width, height)
        shapeColor : tuple (B, G, R)
        thickness : int
    return
        circle img : cv2.np.ndarray
    """
    padding = (thickness // 2 + 1) if thickness > 0 else 1
    shortLength = min(imgSize)
    longLength = max(imgSize)
    maxRadius = shortLength // 2 - 2*padding
    radius = rd.randrange(padding, maxRadius)
    minCenterPos = radius + padding
    maxCenterPos = shortLength - radius - padding
    # 입장이 곤란한 경우 재귀호출 결과 반환
    if minCenterPos >= maxCenterPos:
        return createRandomCircle(imgSize, shapeColor, thickness)
    centerX = rd.randrange(minCenterPos, maxCenterPos)
    centerY = rd.randrange(minCenterPos, maxCenterPos)
    centerPt = (centerX, centerY)
#     print("center point:", centerPt)
    imgShape = imgSize + (3,)
    circleImg = np.zeros(imgShape, np.uint8)
    cv2.circle(circleImg, centerPt, radius, shapeColor, thickness)
    return circleImg

def createRandomRotatedRect(imgSize = (299, 299), rectColor = (255, 255, 255), thickness = 1):
    """짤림 방지된 회전된 사각형 생성 원을 기반으로 그림
    args
        imgSize : tuple (width, height)
        rectColor : tuple (B, G, R)
        thickness : int
    return
        rotated rectange img : cv2.np.ndarray
    """
    padding = (thickness // 2 + 1) if thickness > 0 else 1
    shortLength = min(imgSize)
    longLength = max(imgSize)
    maxRadius = shortLength // 2 - 2*padding
    radius = rd.randrange(padding, maxRadius)
    minCenterPos = radius + padding
    maxCenterPos = shortLength - radius - padding
    # 입장이 곤란한 경우 재귀호출 결과 반환
    if minCenterPos >= maxCenterPos:
        return createRandomRotatedRect(imgSize, rectColor, thickness)
    centerX = rd.randrange(minCenterPos, maxCenterPos)
    centerY = rd.randrange(minCenterPos, maxCenterPos)
    centerPt = (centerX, centerY)
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
#     print(startPt, endPt)
    # 사각형 그리기
    imgShape = imgSize + (3,)
    rectImg = np.zeros(imgShape, np.uint8)
    cv2.rectangle(rectImg, startPt, endPt, rectColor, thickness)
    # 사각형 회전
    rotMatrix = cv2.getRotationMatrix2D(centerPt, rotDeg, 1)
    rotRectImg = cv2.warpAffine(rectImg, rotMatrix, imgSize)
    return rotRectImg

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
    tmpBgImg = cv2.bitwise_and(tmpImg, tmpImg, mask=shapeMask_inv)
    tmpImg = fgImg.copy()
    tmpFgImg = cv2.bitwise_and(tmpImg, tmpImg, mask = shapeMask)
    retImg = cv2.add(tmpBgImg, tmpFgImg)
    return retImg

def getRandomColor(inten = 51):
    """랜덤 RGB 값을 반환함
    """
    return (rd.randrange(0, 256, inten),
            rd.randrange(0, 256, inten),
            rd.randrange(0, 256, inten))

if __name__ == "__main__":
    pass