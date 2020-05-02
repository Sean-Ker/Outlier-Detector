# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: 'Python 3.7.6 64-bit (''ptf'': conda)'
#     name: python37664bitptfcondab6eeea374e5945a4a5dfaf7ce892e423
# ---

# # 6.1 Data Homework
# Sean Kernitsman 
#
# April 27, 2020
#
# Page 168 #1-3, 5, 6, 10.
#

# 1.
#
#     a. Positive moderate
#     b. Positive moderate
#     c. Negative weak
#     d. None
#     e. Positive weak

# 2.
#
#     a. Independent: Cholosterol level. Dependent: Heart disease.
#     b. Independent: Hours of basketball practice. Dependent: Free throw success rate.
#     c. Independent: Amount of fertilizer used. Dependent: Plant height.
#     d. Independent: Level of education. Dependent: Income level.
#     e. Independent: Running speed. Dependent: Pulse rate.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('ggplot')

# 3.
data3 = np.array([[10,11,15,14,8,5],[8,7,4,3,9,10],[72,67,81,93,54,66]])
df3 = pd.DataFrame(data3.T,columns=['Hours Studied','Hours Watching TV','Examination Score'])
df3


plt.scatter(df3['Hours Studied'],df3['Examination Score'],Title = 'Hours Studied vs. Examination Score')
pd.DataFrame([3,2,5])
# plt.legend()
