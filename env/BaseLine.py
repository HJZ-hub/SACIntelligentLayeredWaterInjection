from math import pi,acos,sqrt
# from sympy import *
import numpy as np


class BaseLine():
    '''根据deta与Q关系得到的水嘴开度数据'''
    def __init__(self, a, b,d_o,h,p,p_0,ro,gamma_lambda,nu,ta,eps,g):
        '''
        a-水嘴纵向距离
        b-水嘴横向距离
        zeta-局部损失系数
        d_0-注水管柱直径
        h-注水深度
        p-嘴后压力(地层压力)
        p_0-井口压力
        ro-密度
        gamma_lambda - 沿程损失的计算误差
        nu-流体粘度
        ta-目标流量
        eps-误差
        alpha-学习率
        g-重力加速度
        '''
        self.a = a
        self.b = b
        # self.zeta = zeta
        self.d_o =d_o
        self.h = h 
        self.p = p
        self.p_0 = p_0
        self.ro = ro # 密度
        self.gamma_lambda = gamma_lambda # 沿程损失系数误差
        self.nu = nu # 流体运动粘度
        self.ta = ta 
        self.eps = eps # 误差
        # self.alpha = alpha # 学习率
        self.g = g
        varepsilon=0.2
        self.epsilon = varepsilon/d_o*1e3 #相对粗糙度
    def open_x(self,):
        d_list = np.empty(len(self.h))
        A_list = np.empty(len(self.h))
        v1_list  = np.empty(len(self.h))
        a = self.a*0.001
        b = self.b * 0.001
        d_0 = self.d_o 
        hf = 0

        for i in np.arange(len(self.h)):
            v1_list[i] = (self.ta[i]/86400) / (np.pi * d_0**2/4)
            Re = v1_list[i]*self.d_o/self.nu
            lamd_temp = self.friction_factor(Re=Re)
            #计算沿程损失 
            if i==0:
                h = self.h[i]
            else:
                h = self.h[i] - self.h[i-1]
            hf += (lamd_temp*h*v1_list[i]**2 * self.ro)/(2*self.d_o)
            
            bp = self.p_0*1e6 + self.ro * self.g * self.h[i] - hf  # 单位Pa
            deta_p = bp*1e-6 - self.p[i] #单位MPa

            d = ((self.ta[i] + 3.1425)/(14.0774* deta_p**(0.4836)))**(1.00986)    #这里的 ta是³米每天**重要的公式
            # print(d)
            d_list[i] = (d /(self.a + self.b))*100
        return d_list
              
    def friction_factor(self,Re):
        if Re < 2300:
            return 64 / Re
        else:
            return (-1.8 * np.log10((self.epsilon / 3.7) ** 1.11 + 6.9 / Re)) ** (-2)       

#测试

# x = BaseLine(a =5,b=15,
#              d_o=0.04,h=[1100],
#              p=[24.57],p_0=14.3,
#              ro=980,gamma_lambda=0.0001,
#              nu=1.00e-6,ta=[74.46926276],
#              eps=0.0001,g=9.8).open_x()
# print(f'x:{x}%')