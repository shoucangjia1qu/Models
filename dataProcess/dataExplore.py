"""
*********************************************************************
*                          Funtion Table                            *
*********************************************************************
*     ydescribe()               *      描述y值                       *
*********************************************************************
*     xdescribe()               *      描述x特征                      *
*********************************************************************
*     continuousDscrb()         *      数值型特征的描述性统计           *
*********************************************************************
*     categoryDscrb()           *      单个分类特征的描述统计           *
*********************************************************************

"""

__all__ = ['ExploreMethod']

class ExploreMethod(object):

    # 对标签y的描述统计
    def ydescribe(self, y):
        ySet = np.unique(y)
        n_samples = np.shape(y)[0]
        print('***********************************')
        print('*            y描述统计              *')
        print('***********************************')
        print('y值的类别：{}'.format(ySet))
        print('===================================')
        for i in ySet:
            print('*y为{}的数量{}, \n*占比{}\n*'.format(i, sum(y==i), sum(y==i)/n_samples))
        print('***********************************')
        pass


    # 对特征X的描述统计
    def xdescribe(self, X, dropColumns:list):
        X2 = X.drop(dropColumns, axis=1)
        n_samples, n_features = np.shape(X2)
        feature_lost_ratio = X2.isnull().sum(axis=0)/n_samples
        print('***********************************')
        print('*            X描述统计              *')
        print('***********************************')
        print('*X特征的样本数量：{}'.format(n_samples))
        print('*X特征的特征数量：{}'.format(n_features))
        print('===================================')
        for i in [1,5,10,20,30,50,60,70,80,90,95,99,100]:
            lostthres = 1.-i/100.
            count = sum(feature_lost_ratio >= lostthres)
            print('*缺失率大于等于{}的特征个数有{}个，\n*占比{}\n*'.format(lostthres,count,count/n_features))
        print('***********************************')
        pass


    # 单个数值型特征的描述统计
    def continuousDscrb(self, X, y, colName):
        """
        单个数值型特征的描述统计，包括值的分布图
        Parameters
        ----------
        X: DataFrame
           数据集
        y: Series
            y值
        colName: str
            特征名称

        """
        Xarray = X[colName]
        yValue = np.unique(y)
        #基础统计值，并保存到字典中
        Xmean = Xarray.mean()
        yMean = [Xarray[y==i].mean() for i in yValue]
        #实际画图
        #maxP = sum(Xarray==Xarray.mode()[0])/(Xarray.size*1)
        #画bar\plot图
        fig2 = plt.figure(figsize=(8,6))
        ax0 = fig2.add_subplot(1,1,1)
        ax0.bar(['ymean_'+str(j) for j in yValue],[yMean[j] for j in yValue])
        ax0.plot(['ymean_'+str(j) for j in yValue],[Xmean]*len(yValue),'r--')
        ax0.set_xlabel('label_sort')
        ax0.set_ylabel('feature_mean')
        ax0.set_title(colName)
        #画distplot图，画全量的分布，以及每个y值对应的X的分布
        fig1 = plt.figure(figsize=(8,6))
        ax = fig1.add_subplot(2,np.ceil((len(yValue)+1)/2),1)
        try:
            sns.distplot(Xarray)
        except:
            ax.hist(Xarray, bins=100)
        ax.set_xlabel('feature_value')
        ax.set_ylabel('feature_value_ratio')
        for i in range(len(yValue)):
            ax = fig1.add_subplot(2,np.ceil((len(yValue)+1)/2),i+2)
            try:
                sns.distplot(Xarray[y==yValue[i]])
            except:
                ax.hist(Xarray[y==yValue[i]], bins=100)
            ax.set_xlabel('feature_value:y={}'.format(yValue[i]))
            ax.set_ylabel('feature_value_ratio')
        #画箱型图
        fig3 = plt.figure(figsize=(8,6))
        ax1 = fig3.add_subplot(1,1,1)
        box = []
        for v in yValue:
            #Xarray.dropna(inplace=True)
            #Xarray.reset_index(drop=True, inplace=True)
            Xarray2 = Xarray.dropna().reset_index(drop=True) #必须重新赋值或者分步写
            box.append(Xarray2[y==v])
        ax1.boxplot(box, patch_artist=True)
        plt.xticks(range(1,len(yValue)+1), yValue)
        ax1.set_xlabel('yValue')
        ax1.set_ylabel('feature_value')
        ax1.set_title(colName)
        plt.show()
        pass


    # 单个分类特征的描述统计
    def categoryDscrb(self, X, y, colName):
        """
        单个有序分类特征的描述统计
        Parameters
        ----------
        X: DataFrame
           数据集
        y: Series
            y值
        colName: str
            特征名称
        """
        Xarray = X[colName].fillna(-99)  #需要先填缺，不然Xarray[y==v].value_counts().to_dict()会少一类
        yValue = np.unique(y)
        #基础统计值，并保存到字典中
        Xmean = Xarray.mean()
        yMean = [Xarray[y==i].mean() for i in yValue]
        Xkinds = Xarray.value_counts().to_dict()                #各特征值对应的数量
        yXkinds = {}
        for v in yValue:
            XkindyRatio = {}
            Xkindy = {}
            for i in Xkinds.keys():
                try:
                    icount = Xarray[y==v].value_counts().to_dict()[i]
                except:
                    icount = 0
                Xkindy[i] = icount                              #y=v时，特征值对应的数量
                XkindyRatio[i] = icount/Xkinds[i]               #y=v时，特征值对应的占比
            yXkinds[v] = (Xkindy,XkindyRatio)
        response = y.values.mean()                              #实际响应率
        Xlim = [str(i) for i in Xkinds.keys()]                  #将坐标轴转化为字符串类型
        #画图1，各特征类别中y各值的数量对比
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(2,2,1)
        bottom = np.zeros(len(Xkinds))
        for idx, i in enumerate(yValue):
            if idx == 0:
                ax1.bar(Xlim,list(yXkinds[i][0].values()))
            else:
                bottom += np.array(list(yXkinds[yValue[idx-1]][0].values()))
                ax1.bar(Xlim,list(yXkinds[i][0].values()),bottom=bottom)
        ax1.set_xlabel(f'feature_{colName}')
        ax1.set_ylabel('Quantity')
        ax1.set_title('数量对比：{}'.format(colName))
        plt.legend(['y={}'.format(i) for i in yValue])
        #画图2，各特征值类别中y各值的占比对比
        ax2 = fig.add_subplot(2,2,2)
        bottom = np.zeros(len(Xkinds))
        for idx, i in enumerate(yValue):
            if idx == 0:
                ax2.bar(Xlim,list(yXkinds[i][1].values()))
            else:
                bottom += np.array(list(yXkinds[yValue[idx-1]][1].values()))
                ax2.bar(Xlim,list(yXkinds[i][1].values()),bottom=bottom)
        ax2.set_xlabel(f'feature_{colName}')
        ax2.set_ylabel('Ratio')
        ax2.set_title('占比对比：{}'.format(colName))
        plt.legend(['y={}'.format(i) for i in yValue])
        #plt.legend(['y=1,response','y=0,response','response'])
        #画图3，各0-1分类类别的均值和整体均值的对比
    #     ax3 = fig.add_subplot(2,2,3)
    #     ax3.bar(['ymean_'+str(j) for j in yValue],[yMean[j] for j in yValue])
    #     ax3.plot(['ymean_'+str(j) for j in yValue],[Xmean]*len(yValue),'r--')
    #     ax3.set_xlabel('label_sort')
    #     ax3.set_ylabel('feature_mean')
    #     ax3.set_title('均值对比：{}'.format(colName))
    #     plt.legend(['y={}'.format(i) for i in yValue])
        plt.show()
        pass

