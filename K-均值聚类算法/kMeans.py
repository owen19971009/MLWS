#K-均值聚类算法

#导入库
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

#K-均值聚类支持函数
#导入数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

#计算两个向量的欧氏距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB,2)))


#为给定的数据集构建一个包含K个随机质心得集合
#创建簇中心矩阵，初始化为k个在数据集的边界内随机分布的簇中心
def randCent(dataSet,k):
    n = shape(dataSet)[1]          #得到数据集的列数
    centroids = mat(zeros((k,n)))  #创建一个（k，n）的存放簇的矩阵
    #构建簇质心
    for j in range(n):             #在每个纬度的范围内创建随机的簇中心
        minJ = min(dataSet[:,j])   #求出数据集中第J列的最小值（即第J个特征）
        #用第J个特征最大值减去最小值得出特征j的范围
        rangeJ = float(max(dataSet[:,j]) - minJ)
        #可以这样理解,每个centroid矩阵每列的值都在数据集对应特征的范围内,
        #那么k个簇中心自然也都在数据集范围内;
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids

#K-均值聚类算法
def kMeans(dataSet,k,distMeas = distEclud,createCent = randCent):
    m = shape(dataSet)[0]
    # 创建一个(m,2)维矩阵，第一列存储每个样本对应的簇心，第二列存储样本到簇心的距离
    clusterAssment = mat(zeros((m,2)))
    #用createCent()函数初始化簇心矩阵
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            #遍历簇心，找出离i样本最近的簇心
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果clusterAssment更新，表示对应样本的簇心发生变化，那么继续迭代
            if clusterAssment[i,0] != minIndex:clusterChanged = True
            # 更新clusterAssment,样本到簇心的距离
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            # 利用数组过滤找出簇心对应的簇。获取某个簇类的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            # 对簇求均值,赋给对应的centroids簇心
            centroids[cent,:] = mean(ptsInClust,axis = 0)
    return centroids,clusterAssment

#二分K-均值聚类算法
def biKmeans(dataSet,k,distMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis = 0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            #尝试划分每一簇
            ptsInCurrCluster =dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroidMat,splitClustAss =kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit =\
                    sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit,and notSplit:",sseSplit,sseNotSplit)
            if((sseSplit + sseNotSplit) < lowestSSE):
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print("the bestCentToSplit is:",bestCentToSplit)
        print("the len of bestClustAss is :",len(bestClustAss))
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == \
                               bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment
































