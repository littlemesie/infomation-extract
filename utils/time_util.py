# -*- coding:utf-8 -*-

"""
@date: 2023/8/3 下午7:26
@summary:
"""
import time
from datetime import timedelta

def get_time_diff(start_time):
    # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))