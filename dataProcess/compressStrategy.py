# -*- coding: utf-8 -*-
# @Time:  2020/11/28 23:44
# @Author: wangmuxin

"""
*********************************************************************
*                          Funtion Table                           *
*********************************************************************
*     compress_nstd()           *      对均值+-N倍的标准差进行压缩      *
*********************************************************************

"""

import pandas as pd
import numpy as np

__all__ = ['compress_nstd']

def compress_nstd(X,colName,Nstd=5,style='Float'):
    """
    压缩特征的极值策略，N倍标准差进行压缩
    Parameter
    ---------------
    X: DataFrame
        数据集
    Nstd: Float, default is 5.0
        标准差倍数，默认5
    style: Str, default is 'Float'
        极值类型是float还是int

    Returns
    ----------------
    Xarray
        压缩后的数组
    """
    Xarray = X[colName].copy()
    mean = Xarray.mean()
    std = Xarray.std()
    if style == 'Float':
        maxinum = mean + Nstd * std
        mininum = mean - Nstd * std
    elif style == 'Int':
        maxinum = int(mean + Nstd * std)
        mininum = int(mean - Nstd * std)
    maxinum_idx = (Xarray>maxinum).to_numpy().nonzero()[0]
    mininum_idx = (Xarray<mininum).to_numpy().nonzero()[0]
    Xarray[maxinum_idx] = maxinum
    print('设置的极大值为{}，有{}个样本进行了压缩'.format(maxinum,len(maxinum_idx)))
    Xarray[mininum_idx] = mininum
    print('设置的极小值为{}，有{}个样本进行了压缩'.format(mininum,len(mininum_idx)))
    return Xarray
