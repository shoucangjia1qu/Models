# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:46:16 2020

@author: ecupl
"""

import numpy as np
import pandas as pd
import os, re




class InputBaseData(object):
    def __init__(self, featurePath, labelPath, dateStamp):
        self.feature_path = featurePath
        self.label_path = labelPath
        self.dateStamp = dateStamp
        self.n_samples = 0
        self.n_features = 0
        self.labelCount = {}
        self.featureFiles = os.listdir(featurePath)

    def inputLabel(self):
        """
        输入标签，
        包含"cust_no", "label"两个字段

        return:
        ---------
        label_df : DataFrame
            包含标签的DataFrame
        """
        label_df = pd.read_csv(self.label_path)
        self.labelCount = label_df['label'].value_counts().to_dict()
        return label_df

    def inputVildInfo(self, df, pattern):
        """
        输入客户有效客户数据，
        parameter:
        ----------
        df : DataFrame
            已有的table
        pattern : Str
            符合有效客户的模块

        return:
        ---------
        custvild_df : DataFrame
            追加有效客户数据的df
        """
        for file in self.featureFiles:
            if re.search(pattern, file):
                custdf = pd.read_csv(self.feature_path + "/" + file)
        custdf['vaild_cust'] = True
        custvild_df = df.merge(custdf, how='left', on='cust_no')
        return custvild_df

    def inputQuarterInfo(self, df, pattern):
        """
        输入每季度的信息，不用按月切片
        parameter:
        ----------
        df : DataFrame
            已有的table
        pattern : Str
            符合客户信息的模块

        return:
        ----------
        custinfo_df : DataFrame
            追加信息后的df
        """
        for file in self.featureFiles:
            if re.search(pattern, file):
                custdf = pd.read_csv(self.feature_path + "/" + file)
        custinfo_df = df.merge(custdf, how='left', on='cust_no')
        return custinfo_df

    def inputMonthInfo(self, df, pattern):
        """
        输入每月的数据，按月切片输入数据
        parameter:
        ----------
        df : DataFrame
            已有的table
        pattern : Str
            符合AUM数据的模块

        return:
        ----------
        custother_df : DataFrame
            追加客户AUM数据的df
        """
        #判断列表中的字符串是否在目标字符串中
        #def isinList(targetList, targetString):
        #    for i in targetList:
        #        if targetString.__contains__(i):
        #            return True
        #    else:
        #        return False

        custother_df = df.copy()
        for file in self.featureFiles:
            if re.search(pattern, file):
                fileName, ext = file.split(".")
                otherdf = pd.read_csv(self.feature_path + "/" + file)
                #print(self.feature_path + "/" + file)
                colsName = list(otherdf.columns)
                colsName.remove("cust_no")
                if fileName.split("_")[1] in ['m1', 'm4', 'm7', 'm10']:
                    fileExt = 'm2'
                elif fileName.split("_")[1] in ['m2', 'm5', 'm8', 'm11']:
                    fileExt = 'm1'
                elif fileName.split("_")[1] in ['m3', 'm6', 'm9', 'm12']:
                    fileExt = 'm0'
                else:
                    pass
                #if isinList(['m1', 'm4', 'm7', 'm10'], fileName):
                #    fileExt = 'm2'
                #elif isinList(['m2', 'm5', 'm8', 'm11'], fileName):
                #    fileExt = 'm1'
                #elif isinList(['m3', 'm6', 'm9', 'm12'], fileName):
                #    fileExt = 'm0'
                #else:
                #    pass
                newColsName = [i+"_"+fileExt for i in colsName]
                renameDict = dict([(i, j) for i, j in zip(colsName, newColsName)])
                otherdf.rename(renameDict, axis=1, inplace=True)
                custother_df = custother_df.merge(otherdf, how='left', on='cust_no')
        return custother_df

    def run(self):
        # 0 导入标签数据
        baseData = self.inputLabel()
        # 1 导入客户信息数据
        baseData = self.inputQuarterInfo(baseData, "cust_info")
        # 2 导入客户重大事件数据
        baseData = self.inputQuarterInfo(baseData, "big_event")
        # 3 导入有效客户数据
        baseData = self.inputVildInfo(baseData, "cust_avli")
        # 4 导入aum数据
        baseData = self.inputMonthInfo(baseData, "aum")
        # 5 导入behavior数据
        baseData = self.inputMonthInfo(baseData, "behavior")
        # 6 导入cunkuan数据
        baseData = self.inputMonthInfo(baseData, "cunkuan")
        # 7导入数据回溯时间戳
        baseData['dataStamp'] = pd.to_datetime(pd.Series([self.dateStamp+" 23:59:59"]*baseData.shape[0]), format="%Y-%m-%d %H:%M:%S")
        self.baseData = baseData
        return

#%%
        
if __name__ == "__main__":
    label_path = r"C:\Users\ecupl\Desktop\比赛\data\y_train\y_Q4_3.csv"
    features_path = r"C:\Users\ecupl\Desktop\比赛\data\x_train\Q4"
    dateStamp = "2019-12-31"
    InputBD = InputBaseData(features_path, label_path, dateStamp)
    InputBD.run()
    df = InputBD.baseData  
    
            
    
    
    
    
    
    
    
    

