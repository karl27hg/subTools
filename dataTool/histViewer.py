import json
import os
import matplotlib.pyplot as plt
import copy

def checkEpochCount(histJson, refresh = False):
    """history json의 모든 배열를 비교하여 epochCount 항목을 추가한다.
    """
    resultHistJson = None
    # 파일 경로를 받은 경우
    if isinstance(histJson, str):
        pass
    # 로드된 것을 받은 경우
    elif isinstance(histJson, dict):
        # 우선 json obj로 받았다고 전제
        # resultHistJson = copy.deepcopy(histJson)
        resultHistJson = histJson
    # 정체 불문을 받은 경우
    else:
        return resultHistJson
    epCountLb = "epCount"
    if epCountLb not in resultHistJson.keys() or refresh:
        dataLenList = []
        for itKey in resultHistJson.keys():
            selectedObj = resultHistJson[itKey]
            if isinstance(selectedObj, list):
                dataLenList.append(len(selectedObj))
        epCount = dataLenList[0]
        if len(dataLenList) > 0 and len(dataLenList) == dataLenList.count(epCount):
            resultHistJson[epCountLb] = epCount
    return resultHistJson

# def getHistPlot(jsonData, valueLabels = ("loss")):
#     """json 리스트를 읽어서 그래프로 그려주는 함수
#     valueLabels : ["loss", "val_loss", "acc", "val_acc"]
#     """
#     assert isinstance(jsonData, dict)
#     assert isinstance(valueLabels, tuple)
#     assert isinstance(valueLabels[0], str)
#     lossLabels = ["loss", "val_loss"]
#     accLabels = ["acc", "val_acc"]
#     # 비트 플래그
#     graphBitFlag = 0
#     for itValueLabel in valueLabels:
#         if itValueLabel in lossLabels:
#             graphBitFlag |= 1
#         elif itValueLabel in accLabels:
#             graphBitFlag |= 2
#     # object 생성
#     fig, ax = plt.subplots()
#     valueList = []
#     for itValue in valueLabels:
#         valueList += jsonData[itValue]
#     ax_max = max(valueList)
#     ax_min = min(valueList)
#     ax.set_ylim([ax_min - .1, ax_max + .1])
#     ax.set_xlabel("epoch")
#     if valueLabels[0] in lossLabels:
#         ax.set_ylabel("loss")

#     elif valueLabels[0] in accLabels:
#         ax.set_ylabel("acc")
#     return fig

def getHistLossPlot(jsonData, isShowValLoss = True):
    """
    """
    lossLabel = "loss"
    lossValLabel = "val_loss"
    # 데이터가 없는 경우
    if lossLabel not in jsonData.keys():
        return None
    hasValLoss = lossValLabel in jsonData.keys() and isShowValLoss
    # valueList = jsonData[lossLabel] + (jsonData[lossValLabel] if hasValLoss else [])
    # loss_max = max(valueList)
    # loss_min = min(valueList)

    fig, loss_ax = plt.subplots()
    # loss_ax.set_ylim([loss_min - .1, loss_max + .1])
    if hasValLoss:
        # loss_ax.plot(jsonData[lossValLabel], "r", label = "val loss")
        loss_ax.plot(jsonData[lossValLabel], label = "val loss")
    # loss_ax.plot(jsonData[lossLabel], "b", label = "train loss")
    loss_ax.plot(jsonData[lossLabel], label = "train loss")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.legend(loc = "upper right")
    return fig

def getHistAccPlot(jsonData, isShowValAcc = True):
    """
    """
    accLabel = "acc"
    accValLabel = "val_acc"
    # 데이터가 없는 경우
    if accLabel not in jsonData.keys():
        return None
    hasValAcc = accValLabel in jsonData.keys() and isShowValAcc
    # valueList = jsonData[accLabel] + (jsonData[accValLabel] if hasValAcc else [])
    # acc_max = max(valueList)
    # acc_min = min(valueList)
    fig, acc_ax = plt.subplots()
    # acc_ax.set_ylim([acc_min - .1, acc_max + .1])
    if hasValAcc:
        acc_ax.plot(jsonData[accValLabel], label = "val acc")
    acc_ax.plot(jsonData[accLabel], label = "train acc")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("acc")
    acc_ax.legend(loc = "lower right")
    return fig

def _tmpGetFigPlot(jsonDataList, dataLabel = "loss"):
    """모델 비교를 위해서 임시로 만든 그래프
    """
    assert isinstance(jsonDataList, list)
    labelDict = {
        "loss" : "train loss",
        "val_loss" : "val loss",
        "acc" : "train acc",
        "val_acc" : "val acc"
    }
    fig, ax = plt.subplots()
    for itJsonData in jsonDataList:
        ax.plot(itJsonData[dataLabel], label = itJsonData["modelName"])
    ax.set_xlabel("epoch")
    ax.set_ylabel(labelDict[dataLabel])
    ax.legend(loc = "upper left")
    return fig

if __name__ == "__main__":
    # # jsonDataList = []
    # # fileName = "VGG16_rd_ep500.json"
    # # fileName = "inceptionV4_rd_ep500.json"
    # fileName = "resnet50_rd_ep500.json"
    # jsonFilePath = os.path.join("./dataTool", fileName)
    # with open(jsonFilePath, "r") as f:
    #     jsonData = json.load(f)
    # checkEpochCount(jsonData)
    # # getHistPlot(jsonData)
    # fig = getHistLossPlot(jsonData)
    # fig.savefig("tmpFig.png", format="png")
    # # plt.show(fig)
    # fig1 = getHistAccPlot(jsonData)
    # fig1.savefig("tmpFig1.png", format="png")
    # # plt.show(fig1)
    # 
    # 
    fileNames = ["VGG16_rd_ep500.json", "inceptionV4_rd_ep500.json", "resnet50_rd_ep500.json"]
    jsonDataList = []
    for itFileName in fileNames:
        jsonName, _ = os.path.splitext(itFileName)
        jsonFilePath = os.path.join("./dataTool", itFileName)
        with open(jsonFilePath) as f:
            jsonData = json.load(f)
            jsonData["modelName"] = jsonName
            jsonDataList.append(jsonData)
    for itlabel in ["loss", "val_loss", "acc", "val_acc"]:
        fig = _tmpGetFigPlot(jsonDataList, itlabel)
        fig.savefig("modelComp_" + itlabel +".png", format="png")