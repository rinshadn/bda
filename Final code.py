#!/usr/bin/env python
# coding: utf-8

# In[487]:


get_ipython().system('pip install pyspark')


# In[488]:


import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator


# In[ ]:


df = spark.read.format("csv").option("header", "true").load('gs://arsattrition_forecasting/ars attrition/MFG10YearTerminationData.csv')


# In[379]:


df.columns


# In[380]:


df=df.drop('termtype_desc','gender_full','store_name','recorddate_key','terminationdate_key','orighiredate_key')


# In[381]:


df.dtypes


# In[382]:


from pyspark.sql.types import IntegerType


# In[383]:


df = df.withColumn("age",df["age"].cast(IntegerType()))
df = df.withColumn("length_of_service",df["length_of_service"].cast(IntegerType()))
df = df.withColumn("STATUS_YEAR",df["STATUS_YEAR"].cast(IntegerType()))


# In[384]:


df.dtypes


# In[385]:


df.show(5)


# In[55]:


from pyspark.sql.functions import to_date


# In[386]:


df.dtypes


# In[387]:


df.show(2)


# In[388]:


from pyspark.sql.functions import when
from pyspark.sql.functions import lit
df=df.withColumn("Final_year",                  when((df.gender_short=='M'), lit((60-df.age)+df.STATUS_YEAR))                  .otherwise(lit(65-df.age)+df.STATUS_YEAR)                 )


# In[389]:


df.show(5)


# In[390]:


df.createOrReplaceTempView("df_sql")


# In[391]:


df_clean = spark.sql('''Select * from df_sql where age is not null and length_of_service is not null 
                        and STATUS_YEAR is not null and STATUS is not null and gender_short is not null''')


# In[392]:


df.show(2)


# In[393]:


df_clean.createOrReplaceTempView("df_sql")


# In[332]:


df_clean.show(2)


# In[394]:


AGE = spark.sql("Select age from df_sql")
AGE = AGE.rdd.map(lambda row : row.age).collect()


# In[396]:


from pyspark.sql.functions import expr, col


# In[397]:


print (df)


# In[398]:


df.createOrReplaceTempView('filter_view') 


# In[399]:


df.select("age").show()


# In[400]:


df1=df.filter(df['termreason_desc']=='Retirement') #Terminated data due to retirement is stored to df1 frame so training data set


# In[401]:


df2=df1.filter(df1['gender_short']=='M') #To find retirement age of male who have been terminated as a result of retirement


# In[402]:


df2.show(5)


# In[403]:


df3=df1.filter(df1['gender_short']=='F')#To find retirement age of male who have been terminated as a result of retirement


# In[407]:


df1.select("age","gender_short","termreason_desc").show()  #df2 for male=60 and #df3 for female=65


# In[408]:


df.show(5)


# In[486]:


df_active=df.filter(df['STATUS']=='ACTIVE')


# In[485]:


df_active.show(3) #Test data set with active employees for whom we need to predict the retirement year.


# In[411]:


result_df = df.groupBy("gender_short").count().sort("gender_short", ascending=False)


# In[412]:


result_df.toPandas().plot.bar(x='gender_short',figsize=(14, 6))


# In[349]:


result1_df = df1.groupBy("gender_short").count().sort("gender_short", ascending=False)


# In[413]:


result1_df.toPandas().plot.bar(x='gender_short',figsize=(5, 6))


# In[414]:


result2_df = df1.groupBy("age").count().sort("age", ascending=True)


# In[415]:


result2_df.toPandas().plot.bar(x='age',figsize=(5, 6))


# In[177]:


result2_df.show(5)


# In[182]:


result1_df.show(5)


# In[ ]:





# In[416]:


dfob=df_clean.toPandas()
dfob.dtypes


# In[417]:


spark_df = sqlContext.createDataFrame(dfob)


# In[418]:


spark_df.show(3)


# In[419]:


spark_df.createOrReplaceTempView("dfob_sql")


# In[420]:


df.dtypes


# In[430]:


df_final=df.select("age","length_of_service","STATUS_YEAR","Final_year")


# In[456]:


featureCols = ["age","STATUS_YEAR"]


# In[457]:


from pyspark.ml.feature import VectorAssembler


# In[458]:


assembler = VectorAssembler(inputCols=featureCols, outputCol="features")


# In[459]:


assembled_df = assembler.transform(df_final)


# In[460]:


assembled_df.show(10, truncate=False)


# In[461]:


standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")


# In[462]:


scaled_df = standardScaler.fit(assembled_df).transform(assembled_df)


# In[463]:


scaled_df.select("features", "features_scaled").show(10, truncate=False)


# In[464]:


train_data, test_data = scaled_df.randomSplit([0.8,0.2])


# In[465]:


train_data.columns


# In[466]:


from pyspark.ml.regression import LinearRegression


# In[467]:


lr = LinearRegression(featuresCol="features_scaled",labelCol="Final_year",maxIter=10,regParam=0.3,elasticNetParam=0.2)


# In[468]:


linearModel = lr.fit(train_data)


# In[469]:


predictions = linearModel.transform(test_data)


# In[471]:


predictions.select("Final_year","prediction").show()


# In[472]:


linearModel.coefficients


# In[473]:


print("RMSE: {0}".format(linearModel.summary.rootMeanSquaredError))


# In[474]:


print("MAE: {0}".format(linearModel.summary.meanAbsoluteError))


# In[475]:


print("R2: {0}".format(linearModel.summary.r2))


# In[ ]:




