# 載入必要套件
import requests, datetime, os, time
import numpy as np
import matplotlib.dates as mdates
# from talib.abstract import *

# 算K棒
class KBar():
    # 設定初始化變數
    def __init__(self, date, cycle=1):
        # K棒的頻率 (分鐘)
        self.TAKBar = {}
        self.TAKBar['time'] = np.array([])
        self.TAKBar['open'] = np.array([])
        self.TAKBar['high'] = np.array([])
        self.TAKBar['low'] = np.array([])
        self.TAKBar['close'] = np.array([])
        self.TAKBar['volume'] = np.array([])
        self.current = datetime.datetime.strptime(date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.cycle = datetime.timedelta(minutes=cycle)

    # 更新最新報價
    def AddPrice(self, time, open_price, close_price, low_price, high_price, volume):
        # 若是第一筆資料，直接新增 K 棒
        if self.TAKBar['time'].size == 0:
            self.TAKBar['time'] = np.append(self.TAKBar['time'], self.current)
            self.TAKBar['open'] = np.append(self.TAKBar['open'], open_price)
            self.TAKBar['high'] = np.append(self.TAKBar['high'], high_price)
            self.TAKBar['low'] = np.append(self.TAKBar['low'], low_price)
            self.TAKBar['close'] = np.append(self.TAKBar['close'], close_price)
            self.TAKBar['volume'] = np.append(self.TAKBar['volume'], volume)
            return 1

        # 同一根K棒
        if time <= self.current:
            self.TAKBar['close'][-1] = close_price
            self.TAKBar['volume'][-1] += volume  
            self.TAKBar['high'][-1] = max(self.TAKBar['high'][-1], high_price)
            self.TAKBar['low'][-1] = min(self.TAKBar['low'][-1], low_price)
            return 0
        # 不同根K棒
        else:
            while time > self.current:
                self.current += self.cycle
            self.TAKBar['time'] = np.append(self.TAKBar['time'], self.current)
            self.TAKBar['open'] = np.append(self.TAKBar['open'], open_price)
            self.TAKBar['high'] = np.append(self.TAKBar['high'], high_price)
            self.TAKBar['low'] = np.append(self.TAKBar['low'], low_price)
            self.TAKBar['close'] = np.append(self.TAKBar['close'], close_price)
            self.TAKBar['volume'] = np.append(self.TAKBar['volume'], volume)
            return 1

    # 取時間
    def GetTime(self):
        return self.TAKBar['time']

    # 取開盤價
    def GetOpen(self):
        return self.TAKBar['open']

    # 取最高價
    def GetHigh(self):
        return self.TAKBar['high']

    # 取最低價
    def GetLow(self):
        return self.TAKBar['low']

    # 取收盤價
    def GetClose(self):
        return self.TAKBar['close']

    # 取成交量
    def GetVolume(self):
        return self.TAKBar['volume']
