import numpy as np
import math
import warnings
from decimal import Decimal
class FlowCount():
    def __init__(self, 
                 method,             
                 a, 
                 b, 
                 nu, 
                 levecount, 
                 gamma_lambda,
                 gamma_flow,
                 p_0, 
                 p, h, x
                 ,zeta,
                 ro,
                 g,
                 d_o,
                 varepsilon,
                 totQ,
                 alpha,
                 epoch_number
                 ):
        '''
        - method-> True-恒流方法 False-恒压方法
        - U型水嘴a-> 水嘴宽m
        - U型水嘴b -> 水嘴长m
        - nu -> 流体运动粘度
        - levelcount -> 层段数
        - gamma_lambda -> 沿程损失误差（恒压情况下）
        - gamma_flow -> 计算的流量误差（恒流情况下）
        - p_0 -> 井口压力 MPa
        - p -> 地层压力 MPa
        - h -> 层段深度 m
        - x -> 水嘴开度 30%
        - zeta -> 局部损失系数
        - ro -> 流体密度 kg/m^3
        - g ->重力加速度 m/s^2
        -  d_o ->管柱直径 m 
        - varepsilon 粗糙度

        #测试
        Q = FlowCount(method=True,a=0.01,b=0.02,nu=1.003e-6,
                    levecount=3,gamma_lambda=0.00001,gamma_flow=0.00001,p_0=15,p=[10,10,10],h=[100,250,370],x=[30,5,40],
                    zeta=0.5,ro=1,g=9.8,d_o=0.1,varepsilon=0.015,totQ=197.294,alpha=0.3,epoch_number=5000).justifyModel()
        print('Q:',Q)
        返回Q,p-井口压力,bp-嘴前压力,lambdas-沿程损失系数,v1-水嘴速度,v2-井筒速度,hf-沿程损失,hi-局部损失
        '''
        self.method = method
        self.a = a #毫米
        self.b = b #毫米
        self.nu = nu
        self.levecount = levecount
        self.gamma_lambda = gamma_lambda
        self.gamma_flow = gamma_flow
        self. p_0 = p_0
        self.p = p
        self.h = h
        self.x = x       
        self.zeta = zeta
        self.ro = ro 
        self.g = g
        self.d_o = d_o 
        self.varepsilon = varepsilon
        self.totQ = totQ
        self.alpha = alpha
        self.epoch_number = epoch_number 
        self.epsilon = varepsilon/d_o*1e3 #相对粗糙度
    def countQ(self,):
        '''恒压方法''' 
        hf = np.empty(self.levecount) #沿程损失
        hi = np.empty(self.levecount) #局部损失
        v1_list = np.empty(self.levecount) #管柱流速
        v2_list = np.empty(self.levecount) #出口流速
        lambda_list = np.empty(self.levecount)
        press_Q = np.empty(self.levecount)
        beforP = np.empty(self.levecount)
        rA = np.empty(self.levecount)
        hfadd = 0 #累计的沿程损失
        temp_hf =0
        certen_hf = 0
        for i in np.arange(self.levecount):
            A, X, d_e = self.eqDiameter(self.x[i])      
            # print(f'A:{A}, X:{X}, d_e:{d_e}')
            zeta_ac = self.zeta
            s1 = (np.pi * self.d_o**2) / 4 #井筒面积
            s2 = (np.pi * d_e**2) / 4 #水嘴面积      
            # print(f's1:{s1},s2:{s2},aa:{aa}')
            temp_lamba = 0.01
            cir_count = 0  
            while True:
                cir_count+=1
                lamba = temp_lamba         
                #计算速度v
                #压差
                if(((self.p_0*1e6 + self.ro*self.g*self.h[i]-self.p[i]*1e6) <=0 )or (A == 0)):
                    v1 = 0
                    v2 = 0   
                else:
                    count_x = self.x[i]*(float(self.a) +float(self.b))*0.01
                    zeta_ac = (1.6614e-6 * count_x**6) - (1.0456e-4 * count_x**5) +(2.5830e-3*count_x**4) -(3.1816e-2*count_x**3) + (2.0192e-1*count_x**2) - (5.8107e-1*count_x) +1.2991 #损失函数
                    # print(f'zeta:{zeta_ac}')
                    # print(f'x:{self.x[i]}, zeta:{zeta_ac}')
                    #沿程损失系数的计算
                    if i == 0:
                        # befor_hf = 0
                        certen_hf = (lamba*self.h[i]/self.d_o*self.ro)* (s2**2/s1**2)
                    else :
                        # befor_hf += certen_hf
                        certen_hf += ((lamba*(self.h[i]-self.h[i-1]))/self.d_o*self.ro)* (s2**2/s1**2)
                    v2 = np.sqrt((2*self.p_0*1e6 + 2*self.ro*self.g*self.h[i]-2*self.p[i]*1e6 - 2*certen_hf)/(self.ro + certen_hf + zeta_ac*self.ro))
                    #只算第一段损失
                    # v2 = np.sqrt((2*self.p_0*1e6 + 2*self.ro*self.g*self.h[i]-2*self.p[i]*1e6)/(self.ro + zeta_ac*self.ro))  
                    v1 = v2*A/s1 #井筒流量          
                Re = v1*self.d_o/self.nu
                temp_lamba = self.friction_factor(Re)
                if cir_count > 50:
                    break
                else:
                    if (np.abs(temp_lamba-lamba) <= self.gamma_lambda):
                        break       
            # v2 = s1 * v / s2 # 水嘴入口
            v1_list[i] = v1
            v2_list[i] = v2
            lambda_list[i] = lamba  
            
            # press_Q[i] = v2 * 0.25*np.pi*d_e**2*86400 #转化为m^3/d 
            press_Q[i] = v2 * A *86400 #转化为m^3/d 
            # beforP[i] = (self.p_0*1e6 + self.ro*self.g*self.h[i] - ((0.5*lamba*self.ro*v**2)*(self.h[i]/self.d_o)) - 0.5*zeta_ac*self.ro*v**2)*1e-6        
            if i == 0:
                h = self.h[i]
            else:
                h = self.h[i] - self.h[i-1]
            temp_hf += (self.ro*lamba * h *v1**2)/(self.d_o*2)
            bp = self.p_0*1e6 + self.ro*self.g*self.h[i] - temp_hf/(self.d_o*2)  #- self.ro*v1**2/2
            beforP[i] = bp*1e-6  
            hf[i] = temp_hf
            hi[i] = (self.ro*zeta_ac*v2**2/2)
        return press_Q, self.p_0, beforP,lambda_list,v1_list,v2_list,hf,hi
    def eqDiameter(self, x, ):
        '''面积'''
        b =self.b*1e-3 
        a =self.a*1e-3
        count_x = x * 0.01 * (b + a) #单位是m
        if count_x>=0 and count_x<(a/2):
            if count_x == 0:
                A = 0
                X = 0
            else:
                A =  a**2/4 *np.arccos(1-(2*count_x/a)) - (a/2-count_x) * np.sqrt(a*count_x-count_x**2)
                X =  a * np.arccos(1-2*count_x/a) + 2*np.sqrt(a*count_x-count_x**2)
        elif count_x>=(a/2) and count_x<= np.around((b+a/2),6):
            A = (np.pi * a**2 )/ 8 + a*(count_x-a/2)     
            X = np.pi * a / 2 + 2 * count_x       
        else:
            if count_x == (a + b):
                A = a * b + np.pi * a**2 /4
                X = 2*b + np.pi * a 
            else:
                A = (np.pi * a**2) / 4 + a * b - a**2 /4 * np.arccos(2*(count_x-b)/a - 1) + (count_x - a/2 - b) * np.sqrt(-b*a+2*count_x*b+count_x*a-count_x**2-b**2)
                X = np.pi * a + 2*b - a * np.arccos(2*(count_x-b)/a-1) + 2*np.sqrt(-b*a+2*count_x*b+count_x*a-count_x**2-b**2)
        if count_x == 0:
            d_e = 0
        else:
            d_e = 4*A/X  # 当量直径
        return A, X, d_e # 单位为m^2、m
    def friction_factor(self,Re):
        if Re == 0:
            return 0
        if Re < 2300 :
            return 64 / Re
        else:
            return (-1.8 * math.log10((self.epsilon / 3.7) ** 1.11 + 6.9 / Re)) ** (-2)

#测试代码
# Q,p,bp,lambdas,v1,v2,hf,hi = FlowCount(method=False,a=5,b=15,nu=1.01e-6,
#                 levecount=1,gamma_lambda=0.0001,
#                 gamma_flow=0.00001,p_0=14.3,
#                 p=[24.57],h=[1100],x=[59.95],
#                 zeta=0.7,ro=980.2,g=9.8,d_o=0.04,
#                 varepsilon=0.2,totQ=197.294,
#                 alpha=0.5,epoch_number=5000).countQ()
# print(f'Q:{Q}m³/d,bp:{bp}pa,v1:{v1}m/s,v2:{v2}m/s,hf:{hf}pa,hi:{hi}pa,lambda:{lambdas}')

# x = [10,20,30,40,50,60,70,80,90,100]
# p = [11.9,12,12.1,12.2,12.3,12.4,12.5]
# temp_Q = np.empty(len(x))

# for i in np.arange(len(x)):
#     Q,p,bp,lambdas,v1,v2,hf,hi = FlowCount(method=False,a=5,b=15,nu=1.01e-6,
#                 levecount=1,gamma_lambda=0.0001,
#                 gamma_flow=0.00001,p_0=12.5,
#                 p=[21.38],h=[1026. ],x=[x[i]],
#                 zeta=0.7,ro=980.2,g=9.8,d_o=0.04,
#                 varepsilon=0.2,totQ=197.294,
#                 alpha=0.5,epoch_number=5000).countQ()
#     temp_Q[i]=Q 
# print(f',Q={temp_Q}')      