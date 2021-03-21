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
*     whetherFeatures()         *      转为是否特征                    *
*********************************************************************
*     crossFeatures()           *      交叉特征                       *
*********************************************************************
*     ratioFeatures()           *      转为对数占比特征                 *
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
            {0:缺失, 1:持平, 2:递增, 3:递减, 4:无单调性}
            """
            if series.is_monotonic_decreasing and series.is_monotonic_increasing:
                return 1
            elif series.is_monotonic_increasing:
                return 2
            elif series.is_monotonic_decreasing:
                return 3
            elif series.isnull().sum() > 0:
                return 0
            else:
                return 4

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
    def mathFeatures(self, X, colList, keyName):
        """
        获取指定特征的一些数学统计值，如：最大值、最小值等，形成新的特征

        Parameters
        ----------
        X : DataFrame
            数据集.
        colList : List[String]
            需要统计的特征名列表.
        keyName : String
            特征命名的关键字.

        Returns
        -------
        math_df : DataFrame
            统计之后的特征

        """
        math_df = pd.DataFrame()
        math_df[keyName + "_" + "max"] = X[colList].max(axis=1)
        math_df[keyName + "_" + "min"] = X[colList].min(axis=1)
        math_df[keyName + "_" + "mean"] = X[colList].mean(axis=1)
        math_df[keyName + "_" + "std"] = X[colList].std(axis=1)
        return math_df


    def dateFeatures(X, colName, eddt):
        """
        根据日期特征，求解日期距离当前的天数，形成新的特征

        Parameters
        ----------
        X : DataFrame
            数据集.
        colName : String
            需要转换的特征.
        eddt : %Y-%m-%d
            结束的日期.

        Returns
        -------
        date_series : Series
            转换之后的特征

        """
        dataStamp = datetime.datetime.strptime(eddt, "%Y-%m-%d")
        date_series = (dataStamp - pd.to_datetime(df[colName], format="%Y-%m-%d")).apply(lambda x: x.days)
        return date_series


    def whetherFeatures(self, X, colName):
        """
        根据特征，判断是否有过该特征的性质，形成新的特征

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
        whether_series = X[colName].notnull().astype(int)
        return whether_series


    # 特征交叉
    def crossFeatures(self, featureA, featureB):
        crossfeature = (featureA.astype(str) + featureB.astype(str)).astype(int)
        return crossfeature


    # 求对数比值
    def ratioFeatures(self, X, col1, col2):
        exp = 1.0e-4
        log_series = np.log(X[col1] + exp) - np.log(X[col2] + exp)
        return log_series


    # 求时间区间特征
    def timezoneFeatures(self, X, colName, format):
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
        time_series = timess.apply(judgeTimeZone)
        return time_series


    # 求日期戳区间特征
    def dayzoneFeatures(self, X, colName, format):
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
        day_series = dayss.apply(judgeDayZone)
        return day_series






