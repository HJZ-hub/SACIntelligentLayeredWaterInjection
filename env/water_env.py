"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""
import logging
import math
import gym
import torch
from gym import spaces
from gym.utils import seeding
from env.flow_count_ranzeta import FlowCount as Fl 
from env.flow_count import FlowCount as F2
import numpy as np
from env.BaseLine import BaseLine as BL
import openpyxl
import pandas as pd 

logger = logging.getLogger(__name__)
#恒压环境下的 环境
class FlowEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self,
                levelcount , # 层段数
                turn_time= 200, # turn_time转动时间s 最大转动时间 ->60s
                p0_low = 5,# p0 最小压力
                p0_high = 15,
                a_low = 5, #水嘴宽最小值mm   
                a_high =10, #m
                b_low =15, #m水嘴长
                b_high = 30, #mm
                open_high = 100, #完全打开100%
                pa_low = 10, # 嘴后压力   
                pa_high = 30, 
                volume_low = 10, #目标注入量最小值
                volume_high = 100, #目标注入量最大值
                firstdeep = 1000, # 第一层深度
                interval_step = 100, # 井段取值范围
                def_deep = False, #是否随机深度
                start_press_low = 5, #开始最小压力 5
                start_press_high = 25, #开始最大压力 25
                zeta = 0.5, #局部损失系数按照0.5进行计算
                rotation_rate =0.1, #点击转速水嘴移动距离 mm/s        
                nu=1.003e-6, #流体运动粘度(1.003e-6 水的)
                gamma_lambda = 0.0001, #沿程损失误差（恒压情况下）
                gamma_flow = 0.00001, #计算的流量误差（恒流情况下）
                ro = 980, # 流体密度 kg/m^3(1kg/m^3 水的密度)
                g = 9.8, #重力加速度 m/s^2
                d_o=0.1, #注水管柱直径 m
                varepsilon = 0.015, #粗糙度              
                alpha = 0.3, #学习率
                epoch_number = 400,#训练次数 梯度下降法
                flow_erro = 0.05, #误差5%
                adjust_max_count = 50,
                meted = ''
                ):
        #清理数据文件
        self.meted = meted
        self.ad_count = 0
        self.Ea_rate_list = np.array([])
        self.count = 0
        self.level_count = levelcount #层段数
        #self.p_0 = p_0 #井口压力
        #self.totQ = totQ #总流量
        self.turn_time = turn_time # turn_time转动时间s
        self.p0_low = p0_low # 井口压力  
        self.p0_high = p0_high # 
        self.a_low = a_low # 水嘴宽最小值mm   
        self.a_high = a_high # mm
        self.b_low = b_low #水嘴长
        self.b_high = b_high 
        self.open_high = open_high # 水嘴最大开度
        self.pa_low = pa_low# 嘴后压力 MPa
        self.pa_high = pa_high #
        self.volume_low = volume_low #目标注入量 
        self.volume_high = volume_high #
        self.firstdeep = firstdeep # 第一层深度 m
        self.interval_step = interval_step # 井段取值范围
        self.def_deep = def_deep #是否随机深度 bool
        self.start_press_low = start_press_low #开始最小压力 MPa
        self.start_press_high = start_press_high #开始最大压力 
        self.flow_low = 0
        self.flow_high = 300
        self.zeta = zeta #局部损失系数按照0.5进行计算
        self.rotation_rate = rotation_rate #电机转动的速度 mm/s
        self.nu = nu #运动粘度
        self.gamma_lambda = gamma_lambda #沿程损失误差（恒压情况下）
        self.gamma_flow = gamma_flow #计算的流量误差（恒流情况下）
        self.ro = ro #流体密度 kg/m^3(1kg/m^3 水的密度)
        self.g = g #重力加速度
        self.d_o = d_o #注水管柱
        self.varepsilon = varepsilon #粗糙度
        self.alpha = alpha #学习率
        self.epoch_number =epoch_number #训练次数
        self.Ea_tadd1 = 1 #奖励1计算的误差 
        self.flow_erro = flow_erro # 注入误差
        self.Q = np.empty(self.level_count) #定义一个全局的Q
        self.fig = None #定义一个全局画布
        self.file_name = ''
        #a、b
        self.a = self.a_low #5
        self.b = self.b_low #15
        # Angle at which to fail the episode
        self.adjust_max_count = adjust_max_count #最大迭代次数
        self.reward = 0
        self.done = None
        self.success_list = np.full((levelcount),False)
        Q_low = 0
        Q_high = 200

        '''初始化'''
        #连续动作空间
        action_low = np.array([])
        action_high = np.array([])
        action2_low = -turn_time #旋转旋转时间
        action2_high = turn_time
                
        with open(self.meted +'_'+ str(self.level_count)+'.txt', 'w') as file:
            file.truncate(0)
        # 打开Excel文件
        self.file_name = f'{self.level_count}_{self.meted}_RL_done.xlsx'
        #初始化excle
        try:
            # 尝试打开已存在的 Excel 文件
            workbook = openpyxl.load_workbook(self.file_name)
        except FileNotFoundError:
            workbook = openpyxl.Workbook()
            sheet = workbook.create_sheet()
            sheet.title = "Sheet1"
            workbook.save(self.file_name)
        # 选择第一个工作表
        sheet = workbook.active
        # 删除所有行
        sheet.delete_rows(1, sheet.max_row)
        # 删除所有列
        sheet.delete_cols(1, sheet.max_column)
        # 在单元格中添加数据
        sheet['A1'] = f'{self.level_count}level_done_count' 
        # 保存文件
        workbook.save(self.file_name)
        # 关闭Excel文件
        workbook.close() 

        self.errolist = np.empty((0,self.level_count))
        self.openx = np.empty((0,self.level_count))

        for i in np.arange(self.level_count):    
            action_low = np.append(action_low,action2_low)
            action_high = np.append(action_high,action2_high)
        self.action_space = spaces.Box(action_low,action_high)
        '''动作空间  结构:[转动时间1,转动时间2]''' 
        level_deep_low = np.array([])  # 深度最小的集合
        level_deep_high = np.array([]) # 深度最大的集合
        temp_deep = self.firstdeep # 第一层深度
        for i in range(self.level_count):
            level_deep_low = np.append(level_deep_low, temp_deep)
            temp_deep = temp_deep + self.interval_step
            level_deep_high = np.append(level_deep_high, temp_deep)
        # volume_low = 0 #层段的目标注入量最小
        # volume_high = 100 # 
        #这里记得改
        self.state_low = np.zeros([1+ 5* self.level_count])
        self.state_high = np.zeros([1+ 5* self.level_count])
        self.state_low[0] = self.p0_low # 井口压力范围
        self.state_high[0] = self.p0_high # 
        for i in np.arange(self.level_count):    
            self.state_low[1 + i * 5] =  0 # 水嘴开度
            self.state_high[1 + i * 5] = self.open_high
            self.state_low[2 + i * 5] = level_deep_low[i]  # 深度 
            self.state_high[2 + i * 5] = level_deep_high[i]
            self.state_low[3 + i * 5] = self.pa_low # 嘴后压力
            self.state_high[3 + i * 5] = self.pa_high    
            self.state_low[4 + i * 5] =  Q_low # 流量
            self.state_high[4 + i * 5] = Q_high
            self.state_low[5 + i*5] = self.volume_low #层段目标注入量
            self.state_high[5 + i*5] = self.volume_high
        observation_low = np.full((1+self.level_count*5),0)
        observation_high = np.full((1+self.level_count*5),1)
        self.observation_space = spaces.Box(observation_low,observation_high)  # 状态空间 # type: ignore      
        ''' 状态空间  [井口压力,(x_1(水嘴开度),h_1(层深度),p_1(),ta_1(目标流量),f1,x_2,h_2,ta_2,f1]'''
        self._seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        
     

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action,):
        #判断action是否在动作空间里 
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action)) 
        '''
        ## 传入动作并返回状态和奖励
        - action[转动方向1,转动时间1,转动方向2,转动时间2];
        - state 返回的状态空间[水嘴开度,深度,地层压力(嘴后),井口压力,水嘴尺寸a,水嘴尺寸b]
        - reward 返回的回报
        - done False没有结束 True完成
        - info {}调试项目
        '''
        #action动作换算[转动方向1[-1,0,1]，转动时间1，转动方向2，转动时间2]
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        else:
            action = action
           
        if not self.done:
            temp_x = np.empty(self.level_count) #水嘴开度的集合
            temp_h = np.empty(self.level_count) #地层深度的集合 
            temp_bp = np.empty(self.level_count) #嘴前压力 
            temp_p = np.empty(self.level_count) #地层压力的集合
            temp_Q = np.empty(self.level_count) #流量Q
            temp_ta = np.empty(self.level_count) #目标流量  
            count_state = self.trans_state(self.state)
            for i in np.arange(self.level_count): 
                temp_x[i] = count_state[1 + i*5] #水嘴开度
                temp_h[i] = count_state[2 + i * 5] # 地层深度的集合
                temp_p[i] = count_state[3 + i * 5] # 地层压力的集合
                temp_Q[i] = count_state[4 + i*5] # 流量Q
                temp_ta[i] = count_state[5 + i * 5] # 目标流量的集合
            p_0= count_state[0] #提取井口压力
            a = self.a  # 提取a
            b = self.b # 提取b   
            Ea_t = np.abs(np.subtract(temp_Q,temp_ta)) #误差流量m^3/d 
            Ea_t0 = np.subtract(temp_Q,temp_ta) #带有正负号
                            #奖励的定义
            reward1 = 0
            reward2 = 0
            reward3 = 0
            reward4 = 0  
            reward5 = 0
            reward6 = 0
            if np.all((Ea_t/temp_ta)< self.flow_erro):
                self.done = True
                self.reward = 100
                filename =  self.meted +'_'+ str(self.level_count)+'.txt' 
                with open(filename, 'a') as f:
                    f.write(f'Ea_rate:{np.around(1-(Ea_t/temp_ta))}, Q:{np.around(self.Q)},x:{temp_x},p_0:{p_0},pi:{temp_p},h:{temp_h},ta:{temp_ta}done \n')

                #打开excle
                workbook = openpyxl.load_workbook(self.file_name )
                # 选择第一个工作表
                worksheet = workbook.worksheets[0]
                # 在最后一行的下方添加一行数据
                worksheet.append([self.count])
                # 保存文件
                workbook.save(self.file_name )
                # 关闭Excel文件
                workbook.close()

                erro =np.array((np.subtract(self.Q,temp_ta))/temp_ta) 
                self.errolist = np.vstack((self.errolist,erro))
                self.openx = np.vstack((self.openx,temp_x)) 
            else:
                action_count = action #创建动作的全0数组
                #状态空间  [井口压力，x_1（水嘴开度），h_1（地层深度），p_1（）ta_1(目标流量)，x_2,h_2,ta_2]
                methoed = False # 恒压0
                #更新sate 并取值
                re_time = 0
                for i in np.arange(self.level_count):   
                    if self.level_count == 1:      
                        re_time = action_count
                    else:
                        re_time = action_count[i] #时间为s
                    open_remove =   self.rotation_rate * re_time  #mm
                    if (open_remove>0 and Ea_t0[i]<0) or (open_remove<0 and Ea_t0[i]>0):
                        reward1 += 1
                    # else:
                        # reward1 += -1
                        #  reward1 += -1
                    temp_open =np.around(count_state[1 + i * 5]*(a+b)*0.01 + open_remove,2)  #计算值mm
                    if temp_open <= 0 : #计算值小于0
                        temp_open = 0
                    elif temp_open >= (b + a): #计算值大于最大开度时
                        temp_open = (b + a)
                    rate_open = temp_open/(a+b) * 100 #百分数
                    # 更新了水嘴的开度 应该是百分数0~0.1 #算的有问题
                    self.state[1 + i * 5] = (rate_open-self.state_low[1+i*5])/(self.state_high[1+i*5]- self.state_low[1+i*5]) # 百分数转化为0-1区间
                    temp_x[i] = rate_open #百分比   
                self.Q,countp_0,bP,lamda_list,v_list,v2_list,hf_list,hi_list=Fl(method=methoed,
                        a = a,b = b,
                        nu = self.nu,
                        levecount = self.level_count,
                        gamma_flow = self.gamma_flow,
                        gamma_lambda = self.gamma_lambda,
                        p_0=p_0, 
                        p=temp_p,h=temp_h,x=temp_x,
                        zeta = self.zeta,
                        ro = self.ro,
                        g = self.g,
                        d_o = self.d_o,
                        varepsilon = self.varepsilon,
                        totQ = np.sum(temp_ta),
                        alpha = self.alpha,
                        epoch_number=self.epoch_number
                        ).countQ() #计算
                for i in np.arange(self.level_count):
                    self.state[4+i*5] = (self.Q[i] - self.state_low[4+i*5]) / (self.state_high[4+i*5]-self.state_low[4+i*5]) #更新流量
                Ea_t =np.abs(np.subtract(self.Q ,temp_ta))#误差流量m^3/d
                self.reward = 0 
                if  self.count < self.adjust_max_count:
                    #目标奖励
                    if np.all((Ea_t/temp_ta)< self.flow_erro):
                        self.reward = 100
                        self.done = True # 满足误差要求
                        filename =  self.meted +'_'+ str(self.level_count)+'.txt' 
                        with open(filename, 'a') as f:
                            f.write(f'Ea_rate:{np.around(1-(Ea_t/temp_ta))}, Q:{np.around(self.Q)},x:{temp_x},p_0:{p_0},pi:{temp_p},h:{temp_h},ta:{temp_ta}done \n')
                        #打开excle
                        workbook = openpyxl.load_workbook(self.file_name )
                        # 选择第一个工作表
                        worksheet = workbook.worksheets[0]
                        # 在最后一行的下方添加一行数据
                        worksheet.append([self.count])
                        # 保存文件
                        workbook.save(self.file_name )
                        # 关闭Excel文件
                        workbook.close()

                        erro =np.array((np.subtract(self.Q,temp_ta))/temp_ta) 
                        self.errolist = np.vstack((self.errolist,erro))
                        self.openx = np.vstack((self.openx,temp_x)) 

                        #保存到表格
                        df = pd.DataFrame(self.errolist)
                        df.to_excel(f"{self.level_count}_{self.meted}_erro.xlsx", index=False) 
                        df = pd.DataFrame(self.openx)
                        df.to_excel(f"{self.level_count}_{self.meted}_openx.xlsx", index=False)
                    else:
                        #检查单个是否完成
                        for i in np.arange(self.level_count):
                            # 单个层段完成情况
                            if (Ea_t[i]/temp_ta[i]) < self.flow_erro:
                                self.success_list[i] = True
                                reward5 += 1
                            else:
                                self.success_list[i] = False
                                # reward5 += -1
                            # 单个层段 减小情况   
                            if self.count>0:
                                if (self.Ea_tadd1[i]-Ea_t[i])>=0:
                                    reward6 += 1
                                # else:
                                #     reward6 += -1                       
                else :  # 超过了最大次数
                    self.reward=0
                    self.done = True
                    #打开excle
                    workbook = openpyxl.load_workbook(self.file_name )
                    # 选择第一个工作表
                    worksheet = workbook.worksheets[0]
                    # 在最后一行的下方添加一行数据
                    worksheet.append([self.count])
                    # 保存文件
                    workbook.save(self.file_name )
                    # 关闭Excel文件
                    workbook.close()

                    erro =np.array((np.subtract(self.Q,temp_ta))/temp_ta) 
                    self.errolist = np.vstack((self.errolist,erro))
                    self.openx = np.vstack((self.openx,temp_x)) 

                    #保存到表格
                    df = pd.DataFrame(self.errolist)
                    df.to_excel(f"{self.level_count}_{self.meted}_erro.xlsx", index=False) 
                    df = pd.DataFrame(self.openx)
                    df.to_excel(f"{self.level_count}_{self.meted}_openx.xlsx", index=False)
                    # 稀疏奖励
                    if (temp_x[i]<=0) or (temp_x[i]>=100):
                        reward2 = reward2 -2
                #连续奖励
                ##误差
                reward3 =-(sum(Ea_t/temp_ta)/self.level_count)*10
                ##次数
                reward4 = -2
                self.Ea_tadd1 = Ea_t  
            self.reward = self.reward + reward1 + reward2 + reward3 + reward4 + reward5 + reward6
                # self.reward = -self.Ea_tadd1
            self.count +=1  #计数+1   
            filename =  self.meted +'_'+ str(self.level_count)+'.txt'  
            with open(filename, 'a') as f:
                f.write(f'Ea_rate:{Ea_t/temp_ta}, Q:{np.around(self.Q,2)},x:{temp_x},p_0:{p_0},pi:{temp_p},h:{temp_h},ta:{temp_ta},reward:{np.around(self.reward,2)},count:{self.count}\n')
            erro =np.array((np.subtract(self.Q,temp_ta))/temp_ta) 
            self.errolist = np.vstack((self.errolist,erro))
            self.openx = np.vstack((self.openx,temp_x)) 

        return np.array(self.state.astype(np.float32)),np.float32(self.reward),self.done, {}

    def random_reset(self):
        # self.ran_zeta = np.random.uniform(-0.1,0.4)
        self.success_list = np.full((self.level_count),False) #层段完成状态
        self.ad_count = 0
        self.done = False
        self.reward = 0
        self.count = 0 #重置计数
        while(True):
            # a b open method d都要分开显示
            self.state = np.array([])       
            ''' 状态空间  [井口压力,a(水嘴宽),b(水嘴长)(x_1(水嘴开度),h_1(层深度),p_1()ta_1(目标流量),x_2,h_2,ta_2]'''   
            methoed = False # 恒压  
            self.state =np.random.random(1 + self.level_count * 5)
            count_state =self. trans_state(self.state) #算完为什么是小数
            p_0= count_state[0] #提取井口压力
            a = self.a  #提取a
            b = self.b #提取b 
            temp_x = np.empty(self.level_count) #水嘴开度百分数 *100
            temp_h = np.empty(self.level_count) #地层深度的集合
            temp_bp = np.empty(self.level_count) #嘴前压力
            temp_p = np.empty(self.level_count) #嘴后压力
            temp_Q =np.empty(self.level_count) #流量Q
            temp_ta = np.empty(self.level_count) #目标流量 
            for i in np.arange(self.level_count):
                temp_x[i] = count_state[1 + i * 5] # 水嘴位置x的集合 转化为实际距离 m    
                temp_h[i] = count_state[2 + i * 5] # 地层深度的集合
                temp_p[i] = count_state[3 + i * 5] # 地层压力的集合
                temp_Q[i] = count_state[4 + i*5] # 流量Q
                temp_ta[i] = count_state[5 + i*5] # 目标流量的集合
            #更改嘴后压力，永远只比之前的大0.1Mpa~0.5MPa
            temp_p = (self.ro * self.g * temp_h + p_0*1e6)*1e-6 - round(np.random.uniform(0.1,0.5),2) 
            if np.all((temp_p) > 0):
                if np.all(((self.ro * self.g * temp_h + p_0*1e6)*1e-6 - temp_p)>0):
            #计算当前压力环境下是否能满足注水过程
                    one_hundredx = np.full(self.level_count,[100])
                    self.Q,countp_0,bP,lamda_list,v_list,v2_list,hf_list,hi_list=F2(method=methoed,
                        a = a,b = b,
                        nu = self.nu,
                        levecount = self.level_count,
                        gamma_flow = self.gamma_flow,
                        gamma_lambda = self.gamma_lambda,
                        p_0=p_0, 
                        p=temp_p,h=temp_h,x=one_hundredx,
                        zeta = self.zeta,
                        ro = self.ro,
                        g = self.g,
                        d_o = self.d_o,
                        varepsilon = self.varepsilon,
                        totQ = np.sum(temp_ta),
                        alpha = self.alpha,
                        epoch_number=self.epoch_number
                        ).countQ() 
                    # ba_state = self.baseline_open(self.state).
                    if ((np.all(self.Q >= 1.2*temp_ta)) ):
                    # if ((np.all(self.Q >= 1.2*temp_ta)) and ba_state):
                        self.done = False
                        break        
        #计算当前开度下的误差
        self.Q,countp_0,bP,lamda_list,v_list,v2_list,hf_list,hi_list= F2(method=methoed,
                    a = a,b = b,
                    nu = self.nu,
                    levecount = self.level_count,
                    gamma_flow = self.gamma_flow,
                    gamma_lambda = self.gamma_lambda,
                    p_0=p_0, 
                    p=temp_p,h=temp_h,x=temp_x,
                    zeta = self.zeta,
                    ro = self.ro,
                    g = self.g,
                    d_o = self.d_o,
                    varepsilon = self.varepsilon,
                    totQ = np.sum(temp_ta),
                    alpha = self.alpha,
                    epoch_number=self.epoch_number
                    ).countQ()
        # 更新嘴前压力、流量数据
        for i in np.arange(self.level_count):
            self.state[3+i*5] =(temp_p[i] - self.state_low[3+i*5]) / (self.state_high[3+i*5]-self.state_low[3+i*5])  #将嘴后压力存储到状态矩阵里
            self.state[4+i*5] =(self.Q[i] - self.state_low[4+i*5]) / (self.state_high[4+i*5]-self.state_low[4+i*5])  #更新流量
        self.Ea_tadd1 = np.sum(np.abs(np.subtract(self.Q ,temp_ta))) #误差流量m^3/d         
        return np.array(self.state).astype(np.float32), bP-temp_p
    
    def trans_state(self,state):
        #每个位置的取值范围：array(p_0:[0~20],a:[5~10],b:[15~30],x:[0~100],h:[0~500],p:[5~20],ta:[0~100])
        result = np.array([])
        p_0 = self.state_low[0] + state[0] * (self.state_high[0]- self.state_low[0])
        result = np.append(result,p_0)
        for i in np.arange(self.level_count):
            x = self.state_low[1+i*5] + state[1+i*5] * (self.state_high[1+i*5]-self.state_low[1+i*5]) #水嘴开度
            h = self.state_low[2+i*5] + state[2+i*5] * (self.state_high[2+i*5]-self.state_low[2+i*5]) #深度
            p = self.state_low[3+i*5] + state[3+i*5] * (self.state_high[3+i*5]-self.state_low[3+i*5]) #嘴后压力
            Q = self.state_low[4+i*5] + state[4+i*5] * (self.state_high[4+i*5]-self.state_low[4+i*5]) #流量Q
            ta = self.state_low[5+i*5] + state[5+i*5] * (self.state_high[5+i*5]-self.state_low[5+i*5]) #目标流量
            result = np.concatenate((result,[x],[h],[p],[Q],[ta]))
        return np.around(result,2) 
    def _reset(self):
        self.state,dP = self.random_reset()
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return       
        
    def install(self,state):
        self.state = state  
        self.count = 0
        self.done = False
        return np.array(self.state).astype(np.float32) 

    def baseLine(self,state,train =False):
        self.done=False
        if train:
            self.state = state  
            self.count = 0
        temp_x = np.empty(self.level_count) #水嘴开度的集合
        temp_h = np.empty(self.level_count) #地层深度的集合 
        temp_bp = np.empty(self.level_count) #嘴前压力 
        temp_p = np.empty(self.level_count) #地层压力的集合
        temp_Q = np.empty(self.level_count) #流量Q
        temp_ta = np.empty(self.level_count) #目标流量  
        count_state = self.trans_state(self.state)
        for i in np.arange(self.level_count): 
            temp_x[i] = count_state[1 + i*5] #水嘴开度
            temp_h[i] = count_state[2 + i * 5] # 地层深度的集合
            temp_p[i] = count_state[3 + i * 5] # 地层压力的集合
            temp_Q[i] = count_state[4 + i*5] # 流量Q
            temp_ta[i] = count_state[5 + i * 5] # 目标流量的集合
        p_0= count_state[0] #提取井口压力
        a = self.a  # 提取a
        b = self.b # 提取b   
        Ea_t = np.abs(np.subtract(temp_Q,temp_ta)) #误差流量m^3/d 
        Ea_t0 = np.subtract(temp_Q,temp_ta) #带有正负号
        #奖励的定义 
        if np.all((Ea_t/temp_ta)> self.flow_erro):
            #计算开度 百分比(所有的)
            rate_open = BL( a =self.a,b=self.b,
                            d_o=self.d_o,h=temp_h,
                            p=temp_p,p_0=p_0,
                            ro=self.ro,gamma_lambda=0.0001,
                            nu=self.nu,ta=temp_ta,
                            eps=0.0001,g=self.g).open_x()
            for i  in np.arange(len(rate_open)):
                if math.isnan(rate_open[i] ):
                    return np.array(self.state).astype(np.float32)
                else :
                    if rate_open[i]>100:
                        rate_open[i] = 99
                #更新水嘴开度参数
                for i in np.arange(self.level_count):               
                    self.state[1 + i * 5] = (rate_open[i]-self.state_low[1+i*5])/(self.state_high[1+i*5]- self.state_low[1+i*5]) # 百分数转化为0-1区间
                temp_x = rate_open #百分比
                #计算流量Q
                self.Q,countp_0,bP,lamda_list,v_list,v2_list,hf_list,hi_list=Fl(method=False,
                            a = a,b = b,
                            nu = self.nu,
                            levecount = self.level_count,
                            gamma_flow = self.gamma_flow,
                            gamma_lambda = self.gamma_lambda,
                            p_0=p_0, 
                            p=temp_p,h=temp_h,x=temp_x,
                            zeta = self.zeta,
                            ro = self.ro,
                            g = self.g,
                            d_o = self.d_o,
                            varepsilon = self.varepsilon,
                            totQ = np.sum(temp_ta),
                            alpha = self.alpha,
                            epoch_number=self.epoch_number
                            ).countQ() #计算
            for i in np.arange(self.level_count):      
                        self.state[4+i*5] = (self.Q[i] - self.state_low[4+i*5]) / (self.state_high[4+i*5]-self.state_low[4+i*5])  #更新流量
                        if self.state[4+i*5]>1 :
                            print(self.state[4+i*5])
            filename =  'filename'+ str(self.level_count) +'.txt'  
            with open(filename, 'a') as f:
                f.write(f'Ea_rate:{np.around(np.abs(np.subtract(self.Q,temp_ta))/temp_ta,2)}, Q:{np.around(self.Q,2)},x:{temp_x},p_0:{p_0},pi:{temp_p},h:{temp_h},ta:{temp_ta},reward:{np.around(self.reward,2)},count:0,\n')
            erro =np.array((np.subtract(self.Q,temp_ta))/temp_ta) 
            self.errolist = np.vstack((self.errolist,erro))
            self.openx = np.vstack((self.openx,temp_x))
        return np.array(self.state).astype(np.float32)           

    def ins(data,state):
        return  np.array(self.state).astype(np.float32)

