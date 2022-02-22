# 求函数A = 4*sinx *(1+cosx) 的最大值（x是角度）
# x的范围是0~90°
# 收敛精度ε = 0.05

import numpy as np
import matplotlib.pyplot as plt  # 绘图模块
from pylab import *  # 绘图辅助模块
#处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

def f(x):
    result = 4*sin(x*pi/180)*(1+cos(x*pi/180))
    return result

# 绘图f(x)函数图像：给定闭区间（绘图间隔），绘图间隔默认为0.05，若区间较小，请自行修改
def drawf(left,right,e):
    x = [left + ele * e for ele in range(0, int((right-left) / e))]
    y = [f(ele) for ele in x]
    plt.figure(1)
    plt.plot(x, y,
             color = 'blue', # 折线颜色
             marker = 'o', # 折线图中添加圆点
             markerfacecolor='r', # 点的y颜色
             markersize = 0.01, # 点的大小
             )
    xlim(left, right)
    # 添加图形标题
    plt.title('f(x)函数图像')
    plt.show()

#斐波那契搜索
def Fabonaci_search(left,right,e):
    N = (right-left)/e
    #获得斐波那契数列
    F = [1, 1]
    while F[-1] < N:  #获得小于10000的斐波那契数列
        F.append(F[-2] + F[-1])
    print(F)

    n = len(F) - 1  # 获取斐波那契数列的长度(计数从0开始）
    if n < 3:
        print("精度过低, 无法进行斐波那契搜索")
    else:
        pass

    x_values = []
    x_values.append(left)
    x_values.append(right)
    x1 = left + F[n - 2] / F[n] * (right - left)
    x2 = left + F[n - 1] / F[n] * (right - left)
    t = n
    i = 0
    while (t > 3):
        i += 1
        if f(x1) < f(x2):  # 如果f(x1)<f(x2)，则在区间(x1,right)内搜索
            left = x1
            x1 = x2
            x2 = left + F[t - 1] / F[t] * (right - left)
        elif f(x1) > f(x2):  # 如果f(x1)>f(x2),则在区间(left,x2)内搜索
            right = x2
            x2 = x1
            x1 = left + F[t - 2] / F[t] * (right - left)
        else:  # 如果f(x1)=f(x2)，则在区间(x1,x2)内搜索
            left = x1
            right = x2
            x1 = left + F[t - 2] / F[t] * (right - left)
            x2 = left + F[t - 1] / F[t] * (right - left)
        t -= 1
        x_values.append(left)
        x_values.append(x1)
        x_values.append(x2)
        x_values.append(right)
    #当t<3时
    x1 = left + 0.5 * (right - left)  # 斐波那契数列第3项和第2项的比
    x2 = x1 + 0.1 * (right - left)  # 偏离一定距离，人工构造的点
    if f(x1) < f(x2):  # 如果f(x1)<f(x2)，则在区间(x1,right)内搜索
        left = x1
    elif f(x1) > f(x2):  # 如果f(x1)>f(x2),则在区间(left,x2)内搜索
        right = x2
    else:  # 如果f(x1)=f(x2)，则在区间(x1,x2)内搜索
        left = x1
        right = x2
    x_values.append(left)
    x_values.append(x1)
    x_values.append(x2)
    x_values.append(right)
    #打印结果
    print(f'迭代第{i}次,迭代精度小于等于{e}, 最终的搜索区间为: {min(left, right), max(left, right)}')
    print(f'A的最小值: {f((left + right) / 2)}')
    print('确定最大值的两端值为: ', f(left), f(right))
    draw(x_values)

#绘迭代图
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
    x_values.sort()  #默认列表中的元素从小到大排列
    for x in x_values:
        y_values.append(f(x))
    #绘制折线图
    plt.plot(x_values,
             y_values,
             color = 'blue', # 折线颜色
             marker = 'o', # 折线图中添加圆点
             markerfacecolor='r',
             markersize = 5, # 点的大小
             )
    # 修改x轴和y轴标签
    plt.xlabel('区间')
    plt.ylabel('函数值')
    # 添加图形标题
    plt.title('Faboncci Method求函数最大值')
    # 显示图形
    plt.show()

if __name__ == '__main__':
    a = 0  # 区间下限
    b = 90  # 区间上限
    dx = 0.05  # 迭代精度
    drawf(a,b,dx)  #绘制f(x)函数图像
    Fabonaci_search(a, b, dx)


