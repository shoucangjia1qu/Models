# Models
完整的通用的数据挖掘建模代码

## 一、dataProcess说明

### （一）设计衍生变量DesignFeatures.py说明

- 写了一个DesignMethod的类，其中包含了各种设计方法。

	- 生成单调性的特征**monotonicFeatures()**

	- 生成汇总特征值的特征**totalFeatures()**
	
	- 生成各种数值型统计量的特征**mathFeatures()**

	- 生成日期转为距离某天的天数特征**dateFeatures()**

	- 生成0-1是否的特征**whetherFeatures()**

	- 生成交叉特征**crossFeatures()**

	- 生成对数占比特征**logratioFeatures()**

	- 生成百分比的特征**percentFeatures()**

	- 生成判断时间区间的特征**timezoneFeatures()**

	- 生成判断日期区间的特征**dayzoneFeatures()**







