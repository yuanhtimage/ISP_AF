
import numpy as np
import matplotlib.pyplot as plt  # 绘图模块
from pylab import *  # 绘图辅助模块

#函数
def func(x):
    result = 4*sin(x*pi/180)*(1+cos(x*pi/180))
    return result

#全局搜索
def Global_search(left,right,e,step):
    x_values = []
    while True:
        a = func(step)
        b = func(step + 1)
        step += 2

        x_values.append(a)
        x_values.append(b)

        # DX = abs(b-a)
        if  step>=90:
            # print(f'迭代第{i}次,迭代精度小于{e}, 最终的搜索区间为: {min(left, right), max(left, right)}, A的最大值: {func((left + right) / 2)}')
            max_value = max(x_values)
            max_index = x_values.index(max_value)
            print(max_value,max_index)
            print('确定最大值为: ', max_value, ' 最大值索引: ',max_index)
            break
        else:
            pass
    draw(x_values)

#绘图
def draw(x_values):
    #设置绘图风格
    # plt.style.use('ggplot')
    #处理中文乱码
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #坐标轴负号的处理
    plt.rcParams['axes.unicode_minus']=False
    #横坐标是区间
    #纵坐标是函数值
    y_values = []
    y_values = list(range(0,90))
    # x_values.sort()  #默认列表中的元素从小到大排列
    # for x in x_values:
        # y_values.append(func(x))
    #绘制折线图
    plt.plot(y_values,
             x_values,
             color = 'blue', # 折线颜色
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
    Global_search(a, b, dx, step)
