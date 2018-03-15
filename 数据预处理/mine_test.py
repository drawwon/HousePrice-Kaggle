#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 19:21

import pandas as pf
from pandas import DataFrame as df

a = df(data=[1,2,3,0,1,0,1,0,0],index=None,columns=['data'])
print(a)
a[a['data'] == 0] = 1
print(a)