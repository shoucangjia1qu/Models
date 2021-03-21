# -*- coding: utf-8 -*-
# @Time:  2020/11/27 21:38
# @Author: wangmuxin

"""
*********************************************************************
*                          FillNan Table                           *
*********************************************************************
*     fillna_specialValue()     *      一值进行填缺，指定的特殊值       *
*********************************************************************
*     fillna_modeValue()        *      单一值进行填缺，众数            *
*********************************************************************
*     fillna_meanValue()        *      单一值进行填缺，平均数           *
*********************************************************************
*     fillna_boxing()           *      将缺失值归为一类                *
*********************************************************************
*     fillna_model()            *      模型方法填缺                   *
*********************************************************************

"""

from utils.utils import calSaturation
import pandas  as pd

__all__ = ['FillnanMethod']


class FillnanMethod(object):

    #单一值进行填缺，指定的特殊值
    def fillna_specialValue(self, X,colName,value):
        '''
        单一数值填缺的策略，返回最终用来填补好的列
        1、指定填缺值

        Parameters
        -------------------
        X: DataFrame
            数据集
        colName: Str
            指定填缺的列
        value: Str/int/float
            用来填缺的指定值

        Return
        -------------------
        Xarray : Series
            填缺好的数列
        value : Str/int/float
            用来填缺的值

        '''
        Xarray = X[colName].copy()
        Xarray.fillna(value, inplace=True)
        return Xarray, value

    # 单一值进行填缺，众数
    def fillna_modeValue(self, X, colName):
        '''
        单一数值填缺的策略，返回最终用来填补好的列
        2、指定填缺值,众数

        Parameters
        -------------------
        X: DataFrame
            数据集
        colName: Str
            指定填缺的列

        Return
        -------------------
        Xarray : Series
            填缺好的数列
        value : Str/int/float
            用来填缺的值
        '''
        Xarray = X[colName].copy()
        value = Xarray.mode()[0]
        Xarray.fillna(value, inplace=True)
        return Xarray, value

    # 单一值进行填缺，均值
    def fillna_meanValue(self, X, colName):
        '''
        单一数值填缺的策略，返回最终用来填补好的列
        3、指定填缺值,均值

        Parameters
        -------------------
        X: DataFrame
            数据集
        colName: Float
            指定填缺的列

        Return
        -------------------
        Xarray : Series
            填缺好的数列
        value : Float
            用来填缺的值
        '''
        Xarray = X[colName].copy()
        value = Xarray.mean()
        Xarray.fillna(value, inplace=True)
        return Xarray, value

    # 将缺失值归为一类
    def fillna_boxing(self, X, colName, value=-99):
        '''
        缺失值归为一类的策略，返回最终用来填补好的列
        1、离散无序特征中缺失过高的特征可单独作为一类，默认用-99来表示

        Parameters
        -------------------
        X: DataFrame
            数据集
        colName: Str
            指定填缺的列
        value: Int, default -99
            用来归为一类的缺失值

        Return
        -------------------
        Xarray : Series
            填缺好的数列
        value : Int
            用来填缺的值
        '''
        Xarray = X[colName].copy()
        value = value
        Xarray.fillna(value, inplace = True)
        return Xarray, value

    #模型填缺策略，K近邻、决策树等
    def fillna_model(self, X,column,model='KNN'):
        '''
        模型填缺策略
        '''
        pass









