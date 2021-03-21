# -*- coding: utf-8 -*-
"""
*********************************************************************
*                          Funtion Table                            *
*********************************************************************
*     binning_continuous0()     *      连续值特征分箱方法               *
*********************************************************************
*     binning_category0()       *      离散值特征分箱方法               *
*********************************************************************
*     __orderCategory_bins()    *      有序离散特征分箱                *
*********************************************************************
*     __merging_counts()        *      选择最小的箱子向前或者向后合并     *
*********************************************************************
*     __merging_chi2()          *      卡方分箱办法                    *
*********************************************************************
*     __judge_monotonic()       *      判断单调性                     *
*********************************************************************
*     __noorderCategory_bins()  *      无序分类变量整合长尾部分          *
*********************************************************************
*     __LargeCategory_bins()    *      大型分类变量进行分箱计数          *
*********************************************************************
*     __bins_dict_replace()     *      根据bins字典替换原值            *
*********************************************************************

"""

import numpy as np
import pandas as pd
from scipy import stats
from utils.utils import is_monotonic

__all__ = ['BinningMethod']


class BinningMethod(object):

    # 1、连续值等频分箱，有缺失的单独一项
    def binning_continuous0(self, X, colName, labelName, n, box_thres=5, fillna=-99):
        # 有缺失的单独分为一箱
        if sum(X[colName] == fillna) > 0:
            box_thres -= 1
            # 接着等频分箱，并获得分箱后的结果
            ss, bins0 = pd.qcut(X[colName].iloc[np.nonzero(X[colName].values != fillna)[0]], n, retbins=True, duplicates='drop')
            bins0[0] = bins0[0] - 0.01
            Xarray = pd.cut(X[colName].iloc[np.nonzero(X[colName].values != fillna)[0]], bins0, labels=list(range(len(bins0) - 1))).astype(int)
            # 等频分箱后，继续按照有序离散变量的分箱方法进行操作
            bins1 = self.__orderCategory_bins(Xarray, X[labelName], box_thres)
            bins = [bins0[0]] + [bins0[i+1] for i in bins1[1:]]
            bins = [fillna - 0.01] + bins
        else:
            ss, bins0 = pd.qcut(X[colName], n, retbins=True, duplicates='drop')
            bins0[0] = bins0[0] - 0.01
            Xarray = pd.cut(X[colName].iloc[np.nonzero(X[colName].values != fillna)[0]], bins0, labels=list(range(len(bins0) - 1))).astype(int)
            # 等频分箱后，继续按照有序离散变量的分箱方法进行操作
            bins1 = self.__orderCategory_bins(Xarray, X[labelName], box_thres)
            bins = [bins0[0]] + [bins0[i + 1] for i in bins1[1:]]
        X_box = pd.cut(X[colName], bins, labels=list(range(len(bins) - 1)))
        X_box = X_box.astype(int)
        return X_box, bins

    # 2、分类值分箱
    def binning_category0(self, X, colName, labelName, order=False, box_thres=5, fillna=-99):
        """
        # 离散型分箱:
        # 对于原始特征不超过5类的，不再细分
        # 对于无序离散特征，则正序排列累加，当数量已经超过95%时，即将后面的长尾样本归置为一类，若最终分箱数量超过指定阈值，则进行分箱计数，转为连续值。
        # 对于有序离散特征，先刨除缺失的箱子，再进行箱子的合并，合并方法有很多：
            可以先最小箱子向前或者向后合并，如果合并出来的分箱是单调的，则停止，
            否则尝试卡方分箱方法。
        """
        Xarray = X[colName]
        label = X[labelName]
        if Xarray.unique().__len__() <= box_thres:
            # 箱子个数少的不需要再分箱
            X_box = Xarray
            bins = None
        elif order == False:
            # 无序离散特征
            ## 0、合并箱子，得到bins，字典格式
            bins = self.__noorderCategory_bins(Xarray, ratioThres=0.95)
            Xarraytemp = self.__bins_dict_replace(Xarray, bins)
            print(f"第一次合并后的bins是{bins}")
            ## 1、分箱后箱子数量太多，就再次进行分箱计数，替换掉bins
            if bins.__len__() > box_thres+1:
                bins1 = self.__LargeCategory_bins(Xarraytemp, label)
                print(f"第二次分箱计数后的bins是{bins1}")
                for k, v in bins.items():
                    bins[k] = bins1[v]
            ## 2、根据bins替换原值
            X_box = self.__bins_dict_replace(Xarray, bins)
        elif order == True:
            # 有序离散特征
            ## 0、合并箱子，先判断是否有缺失一类的箱子，有的话，单独拿出来
            if sum(Xarray == fillna) > 0:
                print("有缺失值的分箱，单独作为一类！")
                ### 目标箱数减1
                box_thres -= 1
                ### 取出非Nan的特征和标签
                notNanIdx = np.nonzero(Xarray.values != fillna)[0]
                Xarraytemp = Xarray.iloc[notNanIdx]
                labeltemp = label.iloc[notNanIdx]
                ### 有序离散特征的分箱
                bins = self.__orderCategory_bins(Xarraytemp, labeltemp, box_thres)
                ### 把缺失的箱子值加进bins里去
                bins = [fillna - 0.01] + bins
            else:
                bins = self.__orderCategory_bins(Xarray, label, box_thres)
            print(f"最后分箱结果是{bins}")
            ## 1、根据bins替换原值
            X_box = pd.cut(Xarray, bins, labels=list(range(len(bins) - 1)))
            X_box = X_box.astype(int)
        else:
            pass
        return X_box, bins


    # 有序离散特征的分箱
    def __orderCategory_bins(self, Xarray, ylabel, n):
        """
        判断单调性，在utils中的is_monotonic的基础上进行修改

        Parameters
        ----------
        Xarray : 1D array-like
            特征数据.
        ylabel : 1D array-like
            标签数据.
        n : Int
            分箱的目标个数.

        Returns
        -------
        bins : List
            分箱的切分点.

        """
        ### 最小的箱子向前或者向后合并
        Xset = list(np.unique(Xarray))
        bins0 = [Xset[0] - 0.01] + Xset
        bins = self.__merging_counts(Xarray, bins0, n)
        print(f">>>最小箱子合并结束，bins是{bins}")
        ### 判断单调性
        monotonic = self.__judge_monotonic(Xarray, ylabel, bins)
        ### 判断是否进行其他分箱，没有单调性则重新分箱
        if monotonic:
            print(f"单调性是{monotonic}")
            print("分箱结束！")
        else:
            print(">>>开始卡方分箱")
            bins = self.__merging_chi2(Xarray, ylabel, bins0, n)
            print(f"卡方分箱结束，bins是{bins}")
        return bins


    # 选择最小的箱子，向前或者向后合并箱子
    def __merging_counts(self, Xarray, bins, n):
        """
        最小箱子向前或者向后合并。

        Parameters
        ----------
        Xarray : 1D array-like
            特征数据.
        bins : List
            分箱切分点.
        n : Int
            分箱的目标个数.

        Returns
        -------
        bins : List
            分箱的切分点.

        """
        while len(bins) > n+1:
            minCount = np.inf
            minIdx = 0
            CountDict = {}
            # 找到最小的箱子数量、箱子下标
            for idx, bin in enumerate(bins):
                if idx==0:
                    continue
                bin0 = bins[idx-1]
                bin1 = bin
                XiCount = sum((Xarray>bin0)&(Xarray<=bin1))
                if XiCount < minCount:
                    minIdx = idx
                    minCount = XiCount
                CountDict[idx] = XiCount
            # 根据最小箱子判断向前合并还是向后合并
            if minIdx == 1:
                bins.pop(minIdx)
            elif minIdx == len(bins)-1:
                bins.pop(minIdx-1)
            elif CountDict[minIdx-1] <= CountDict[minIdx+1]:
                bins.pop(minIdx - 1)
            else:
                bins.pop(minIdx)
        return bins


    # 按照卡方进行分箱
    def __merging_chi2(self, Xarray, ylabel, bins, n):
        """
        卡方分箱，个数小于指定箱数，并且要符合单调性，才停止分箱。

        Parameters
        ----------
        Xarray : 1D array-like
            特征数据.
        ylabel : 1D array-like
            标签数据.
        bins : List
            分箱的节点.
        n : Int
            分箱的目标个数

        Returns
        -------
        bins : List
            分箱的切分点.

        """
        flag = True
        while (len(bins) > n + 1) or flag:
            minChi2 = np.inf
            minIdx = 0
            # 找到最小的临近卡方值
            for idx, bin in enumerate(bins):
                if (idx == 0) or (idx == len(bins)-1):
                    continue
                usedIdx = np.nonzero((Xarray.values>bins[idx-1])&(Xarray.values<=bins[idx+1]))[0]
                cross_table = pd.crosstab(columns=ylabel.iloc[usedIdx], index=Xarray.iloc[usedIdx])
                chiq, pValue, df, expected_freq = stats.chi2_contingency(cross_table)
                if chiq < minChi2:
                    minIdx = idx
                    minChi2 = chiq
            # 合并最小的临近卡方值
            bins.pop(minIdx)
            # 检验单调性
            monotonic = self.__judge_monotonic(Xarray, ylabel, bins)
            if monotonic:
                flag = False
            else:
                flag = True
        print(f"单调性是{monotonic}")
        return bins


    def __judge_monotonic(self, Xarray, ylabel, bins):
        """
        判断单调性，在utils中的is_monotonic的基础上进行修改

        Parameters
        ----------
        Xarray : 1D array-like
            特征数据.
        ylabel : 1D array-like
            标签数据.
        bins : List
            分箱的节点.

        Returns
        -------
        flag : String Or None
            单调性判断，有“递增”、“递减”、None

        """
        # 按照分箱节点切分后的数据
        Xarraybins = pd.cut(Xarray, bins, labels=list(range(len(bins) - 1)))
        # 调用is_monotonic函数进行判断单调性
        flag = is_monotonic(pd.DataFrame({'feature': Xarraybins.astype(int), 'label': ylabel}), 'feature', 'label')
        return flag


    # 无序分类小型分类变量的分箱
    def __noorderCategory_bins(self, Xarray, ratioThres):
        ratio = 0
        size = Xarray.size
        bins = {}
        box_values_dict = Xarray.value_counts().to_dict()
        for k, v in box_values_dict.items():
            # 当占比大于阈值时，剩余的特征全部用当前特征代替
            if ratio >= ratioThres:
                bins['other'] = k
                break
            ratio += v / size
            bins[k] = k
        return bins

    # 无序大型分类变量进行分箱计数
    def __LargeCategory_bins(self, Xarray, y):
        exp = 1.0e-5
        ySet = np.unique(y)
        xSet = np.unique(Xarray)
        bins = {}
        for j in xSet:
            bins[j] = np.log((sum((y == 1) & (Xarray == j)) + exp) / (sum((y != 1) & (Xarray == j)) + exp))
        return bins


    # 根据bins字典替换掉原值
    def __bins_dict_replace(self, Xarray, bins):
        X_box = Xarray.copy()
        for i in Xarray.unique():
            if i in bins.keys():
                X_box.replace({i: bins[i]}, inplace=True)
            else:
                X_box.replace({i: bins['other']}, inplace=True)
        return X_box



