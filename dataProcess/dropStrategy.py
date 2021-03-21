# -*- encoding: utf-8 -*

"""
@File: dropStrategy.py.py
@Author: Wang Muxin
@Date: 2020/11/22

"""
"""
*********************************************************************
*                          Funtion Table                           *
*********************************************************************
*     dropCol_saturation()        *      对饱和度低的特征进行删除       *
*********************************************************************
*     dropCol_highratio()         *      对个值占比特高的特征进行删除    *
*********************************************************************
*     dropCol_completelable()     *     和y值完全相关的特征进行删除      *
*********************************************************************

"""

import pandas as pd
from utils.utils import calSaturation

__all__ = ['DropcolMethod']


class DropcolMethod(object):

    #删除低饱和度特征
    def dropCol_saturation(self, X, colName, saturationThres=0.05):
        """
        删除饱和度低的特征

        Parameters
        ---------------
        X: DataFrame
            数据集
        colName: Str
            特征的名称
        saturationThres: Float
            饱和度的阈值,默认0.05,即删除饱和度不足5%的特征
        """
        flag = False
        saturation = calSaturation(X, colName)
        if saturation < saturationThres:
            flag = True
        return flag


    #删除个值占比特高的特征
    def dropCol_highratio(self, X, colName, valueThres=0.99):
        '''
        删除个值占比特高的特征

        Parameters
        ------------------
        X: DataFrame
            数据集
        colName: Str
            特征的名称
        valueThres: Float
            单个值占比的阈值，默认0.99，即删除99%为同值的特征

        Returns
        -----------------
        flag: Boolean
            是否需要删除
        '''
        flag = False
        mode = X[colName].mode()[0]
        mode_ratio = sum(X[colName].values == mode)/X.shape[0]
        if  mode_ratio > valueThres:
            flag = True
        return flag


    #删除和标签完全相关的特征，即特征中的同一值的样本标签为一类
    def dropCol_completelable(self, X, colName, labelName):
        '''
            删除和标签完全相关的特征，即特征中的同一值的样本标签为一类

            Parameters
            ------------------
            X: DataFrame
                数据集
            colName: Str
                特征的名称
            labelName: Str
                标签的名称

            Returns
            -----------------
            flag: Boolean
                是否需要删除
            '''
        flag = False
        target = set(X[labelName].values)
        table = X.groupby(colName)[labelName].mean()
        ratioSet = set(table.values)
        if ratioSet == target:
            flag = True
        return flag


