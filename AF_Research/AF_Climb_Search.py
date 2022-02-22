import numpy as np
import matplotlib.pyplot as plt  # 绘图模块
from pylab import *
from sympy import true  # 绘图辅助模块

#函数
def func(x):
    result = 4*sin(x*pi/180)*(1+cos(x*pi/180))
    return result

#全局搜索
def Climb_search(left,right,e,step):
    x_values = []
    y_values = []

    max_fv = 0
    iter = 0
    for  i in range(left,right,step):
        temp_fv = func(i)
        y_values.append(temp_fv)
        if (temp_fv < max_fv):
            iter = iter + 1
        else:
            max_fv = temp_fv
        back_fv = func(i)
        temp_step = i
        x_values.append(i)
        while (iter>3):
            y_values.append(back_fv)
            back_fv = func(temp_step)

            print(temp_step,back_fv,max_fv)
            if (abs(back_fv-max_fv)<0.001):
                x_values.append(temp_step)
                draw(x_values,y_values)
                return
            else:
                temp_step = temp_step - 1
                x_values.append(temp_step)

#绘图
def draw(x_values,y_values):
    #设置绘图风格
    # plt.style.use('ggplot')
    #处理中文乱码
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #坐标轴负号的处理
    plt.rcParams['axes.unicode_minus']=False
    #横坐标是区间
    #纵坐标是函数值
    # x_values.sort()  #默认列表中的元素从小到大排列
    # for x in x_values:
        # y_values.append(func(x))
    #绘制折线图
    plt.plot(x_values[len(x_values)-10:],
             y_values[len(x_values)-10:],
             color = 'b', # 折线颜色
             marker = '*', # 折线图中添加圆点
             markerfacecolor='b', # 点的y颜色
             markersize = 5, # 点的大小
             )
    plt.plot(x_values[:len(x_values)-10],
            y_values[:len(x_values)-10],
            color = 'r', # 折线颜色
            marker = 'o', # 折线图中添加圆点
            markerfacecolor='r', # 点的y颜色
            markersize = 5, # 点的大小
            )
    # 修改x轴和y轴标签
    plt.xlabel('区间')
    plt.ylabel('函数值')
    # 添加图形标题
    plt.title('Golden Section Search Method求函数最大值')
    # 显示图形
    plt.show()

if __name__ == '__main__':
    a = 0  # 区间下限
    b = 90  # 区间上限
    dx = 0.05  # 迭代精度
    step = 1 # 全局搜索步长
    Climb_search(a, b, dx, step)
