# 对地理坐标进行聚类
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

#函数功能：计算两个数据集之间的欧式距离
#输入：两个数据集。
#返回：两个数据集之间的欧式距离（此处用距离平方和代替距离）
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

#创建簇中心矩阵,初始化为k个在数据集的边界内随机分布的簇中心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        #求出数据集中第j列的最小值(即第j个特征)
        minJ = min(dataSet[:,j])
        #用第j个特征最大值减去最小值得出特征值范围
        rangeJ = float(max(dataSet[:,j]) - minJ)
        #创建簇矩阵的第J列,random.rand(k,1)表示产生(10,1)维的矩阵，其中每行值都为0-1中的随机值
        #可以这样理解,每个centroid矩阵每列的值都在数据集对应特征的范围内,那么k个簇中心自然也都在数据集范围内
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    #创建一个(m,2)维矩阵，第一列存储每个样本对应的簇心，第二列存储样本到簇心的距离
    clusterAssment = mat(zeros((m,2)))
    #用createCent()函数初始化簇心矩阵
    centroids = createCent(dataSet, k)
    #保存迭代中clusterAssment是否更新的状态,如果未更新，那么退出迭代，表示收敛
    #如果更新，那么继续迭代，直到收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        #程序中可以创建一个标志变量clusterChanged，如果该值为True，则继续迭代。
        #上述迭代使用while循环来实现。接下来遍历所有数据找到距离每个点最近的质心，
        #这可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        #对每个样本找出离样本最近的簇心
        for i in range(m):
            #minDist保存最小距离
            #minIndex保存最小距离对应的簇心
            minDist = inf
            minIndex = -1
            #遍历簇心，找出离i样本最近的簇心
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            #如果clusterAssment更新，表示对应样本的簇心发生变化，那么继续迭代
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            #更新clusterAssment,样本到簇心的距离
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        #遍历簇心,更新簇心为对应簇中所有样本的均值
        for cent in range(k):
            #利用数组过滤找出簇心对应的簇。获取某个簇类的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            #对簇求均值,赋给对应的centroids簇心
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    #取数据集特征均值作为初始簇中心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    #centList保存簇中心数组,初始化为一个簇中心
    #create a list with one centroidfrfc 
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    #迭代，直到簇中心集合长度达到k
    while (len(centList) < k):
        #初始化最小误差
        lowestSSE = inf
        #迭代簇中心集合，找出找出分簇后总误差最小的那个簇进行分解
        for i in range(len(centList)):
            #获取属于i簇的数据集样本
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #对该簇进行k均值聚类
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #获取该簇分类后的误差和
            sseSplit = sum(splitClustAss[:,1])
            #获取不属于该簇的样本集合的误差和，注意矩阵过滤中用的是!=i
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            #打印该簇分类后的误差和和不属于该簇的样本集合的误差和
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            #两误差和相加即为分簇后整个样本集合的误差和,找出簇中心集合中能让分簇后误差和最小的簇中心，保存最佳簇中心(bestCentToSplit),
            #最佳分簇中心集合(bestNewCents),以及分簇数据集中样本对应簇中心及距离集合(bestClustAss),最小误差(lowestSSE)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #更新用K-means获取的簇中心集合，将簇中心换为len(centList)和bestCentToSplit,
        #以便之后调整clusterAssment(总样本集对应簇中心与和簇中心距离的矩阵)时一一对应
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        #更新簇中心集合,注意与bestClustAss矩阵是一一对应的
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

def distSLC(vecA, vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 

def loadDataSet(fileName):      
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

print(clusterClubs())
































