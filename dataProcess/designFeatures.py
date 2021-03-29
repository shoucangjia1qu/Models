# -*- coding: utf-8 -*-
"""
*********************************************************************
*                          Funtion Table                            *
*********************************************************************
*     monotonicFeatures()       *      生成单调性特征                  *
*********************************************************************
*     totalFeatures()           *      汇总特征值                     *
*********************************************************************
*     mathFeatures()            *      特征统计量特征                  *
*********************************************************************
*     dateFeatures()            *      日期转天数特征                  *
*********************************************************************
*     whetherFeatures()         *      转为0-1是否特征                 *
*********************************************************************
*     crossFeatures()           *      交叉特征                       *
*********************************************************************
*     logratioFeatures()        *      转为对数占比特征                 *
*********************************************************************
*     percentFeatures()         *      转为百分比的特征                 *
*********************************************************************
*     timezoneFeatures()        *      判断时间区间特征                 *
*********************************************************************
*     dayzoneFeatures()         *      判断日期区间特征                 *
*********************************************************************

"""


import numpy as np
import pandas as pd
import datetime

__all__ = ['DesignMethod']


class DesignMethod(object):

    # 生成单调性特征
    def monotonicFeatures(self, X, colName):
        """
        自动生成判断单调性的特征

        Parameters
        ----------
        X : DataFrame
            数据集
        colName : String
            需要计算单调性的特征名称.

        Returns
        -------
        monotonic_series : Series
            生成单调性特征的数据.

        """

        # 判断单调性的函数
        def monotonicFlag(series):
            """
            {0:持平, 1:递增, 2:递减, 3:无单调性}
            """
            if series.isnull().sum() > 0:
                series.fillna(0, inplace=True)
            if series.is_monotonic_decreasing and series.is_monotonic_increasing:
                return 0
            elif series.is_monotonic_increasing:
                return 1
            elif series.is_monotonic_decreasing:
                return 2
            else:
                return 3

        monotonic_series = X[colName].apply(monotonicFlag, axis=1)
        return monotonic_series


    # 汇总指定特征的值
    def totalFeatures(self, X, colList):
        """
        汇总指定的特征，形成新的特征

        Parameters
        ----------
        X : DataFrame
            数据集.
        colList : List[String]
            需要汇总的特征.

        Returns
        -------
        total_series : Series
            汇总之后的特征

        """
        total_series = X[colList].sum(axis=1)
        return total_series


    # 计算指定特征的统计量
    def mathFeatures(self, X, colName):
        """
        自动生成数值型的特征

        Parameters
        ----------
        X : DataFrame
            数据集
        colName : String
            需要计算数值型的特征名称.

        Returns
        -------
        math_df : DataFrame
            生成数值型特征组合的数据.

        """
        series = X[colName]
        math_df = pd.DataFrame()
        math_df[colName + "_" + "max"] = series.max(axis=1)
        math_df[colName + "_" + "min"] = series.min(axis=1)
        math_df[colName + "_" + "mean"] = series.mean(axis=1)
        math_df[colName + "_" + "std"] = series.std(axis=1)
        return math_df

    # 生成日期特征
    def dateFeatures(self, X, colName, eddt, format='%Y-%m-%d'):
        """
        根据日期特征，求解日期距离当前的天数，形成新的特征
        若有缺失值，则以当前的日期填缺

        Parameters
        ----------
        X : DataFrame
            数据集.
        colName : String
            需要转换的特征.
        eddt : String
            结束日期，也就是当前日期.
        format : String
            日期格式，字符串转换为日期格式的格式，默认为'%Y-%m-%d'

        Returns
        -------
        date_series : Series
            转换之后的特征

        """
        series = X[colName]
        if series.isnull().sum() > 0:
            series.fillna(eddt, inplace=True)
        dataStamp = datetime.datetime.strptime(eddt, format)
        date_series = (dataStamp - pd.to_datetime(series, format="%Y-%m-%d")).apply(lambda x: x.days)
        return date_series

    # 生成0-1标签
    def whetherFeatures(self, X, colName):
        """
        根据特征，判断是否有过该特征的性质，形成新的特征，以是否大于0来判断是否有过该特征

        Parameters
        ----------
        X : DataFrame
            数据集.
        colName : String
            需要判断的特征.

        Returns
        -------
        whether_series : Series
            判断之后的特征

        """
        series = X[colName]
        whether_series = (series>0).astype(int)
        return whether_series


    # 特征交叉
    def crossFeatures(self, X, colNameA, colNameB):
        """
        两两生成，交叉特征。

        Parameters
        ----------
        X: DataFrame
            数据集.
        colNameA : String
            第一个特征名称.
        colNameB : String
            第二个特征名称.

        Returns
        -------
        cross_series : Series
            生成交叉特征之后的新特征

        """
        seriesA = X[colNameA]
        seriesB = X[colNameB]
        cross_series = (seriesA.astype(str) + seriesB.astype(str)).astype(int)
        return cross_series


    # 求两个特征的比值取对数
    def logratioFeatures(self, X, colNameA, colNameB):
        """
        对两个特征求比值，并取对数

        Parameters
        ----------
        X : DataFrame
            数据集.
        colNameA : String
            第一个特征名称.
        colNameB : String
            第二个特征名称.

        Returns
        -------
        logratio_series : Series
            比值取对数后的生成的特征

        """
        exp = 1.0e-4
        logratio_series = np.log(X[colNameA]/(X[colNameB] + exp) + exp)
        return logratio_series


    # 求两个特征的百分比特征
    def percentFeatures(self, X, colNameA, colNameB):
        """
        求特征A在特征B中的百分比占比，生成特征

        Parameters
        ----------
        X : DataFrame
            数据集.
        colNameA : String
            第一个特征名称.
        colNameB : String
            第二个特征名称.

        Returns
        -------
        percent_series : Series
            求百分比后的生成的特征

        """
        exp = 1.0e-4
        percent_series = X[colNameA]/(X[colNameB] + exp)
        return percent_series


    # 求时间区间特征
    def timezoneFeatures(self, X, colName, format):
        """

        Parameters
        ----------
        X : DataFrame
            数据集.
        colName : String
            需要判断的特征.
        format : String
            字符换转换成日期格式的日期.

        Returns
        -------
        timezone_series : Series
            时区的分段特征.

        """
        def judgeTimeZone(timeStamp):
            """
            {0:0~6, 1:6~12, 2:12~18, 3:18~24, 99:NaT}

            """

            if pd.isnull(timeStamp):
                return 99
            elif 0 <= timeStamp.hour < 6:
                return 0
            elif 6 <= timeStamp.hour < 12:
                return 1
            elif 12 <= timeStamp.hour < 18:
                return 2
            else:
                return 3
        timess = pd.to_datetime(X[colName], format=format)
        timezone_series = timess.apply(judgeTimeZone)
        return timezone_series


    # 求日期戳区间特征
    def dayzoneFeatures(self, X, colName, format):
        """

        Parameters
        ----------
        X : DataFrame
            数据集.
        colName : String
            需要判断的特征.
        format : String
            字符换转换成日期格式的日期.

        Returns
        -------
        dayzone_series : Series
            上中下旬的分段特征.

        """
        def judgeDayZone(dateStamp):
            """
            {0:1~10, 1:11~20, 2:21~31, 99:NaT}

            """
            if pd.isnull(dateStamp):
                return 99
            elif 1 <= dateStamp.day <= 10:
                return 0
            elif 11 <= dateStamp.day <= 20:
                return 1
            else:
                return 2
        dayss = pd.to_datetime(X[colName], format=format)
        dayzone_series = dayss.apply(judgeDayZone)
        return dayzone_series






