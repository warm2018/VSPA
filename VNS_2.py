# 将变化了的需求插入到新的解中，再利用变邻域搜索提高插入解的质量。
#第一步应生成动态需求，航班延误的需求(规定)，即时间窗变化规则，全部随机变化。
# Required Libraries
import pandas as pd
import random
import copy
import numpy as np
import math 
import collections
from VSPA_1 import Gene


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation 
import time


DEBUG = 0
## SOME base parameters for this model
X_coordinate = [120,56, 66, 56, 88, 88, 24, 40, 32, 16, 88, 48, 32, 80, 48, 23, 48, 16, 8, 32, 24, 72, 72, 72, 88, 34]
Y_coordinate = [60,56, 78, 27, 72, 32, 48, 48, 80, 69, 96, 96, 45, 56, 40, 16, 8, 32, 48, 64, 96, 65, 32, 16, 8, 56]
demand       = [0,1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1]
expect_low   = [0,4, 4, 4, 4, 6, 6, 6, 6, 10, 10, 10, 10, 15, 15, 15, 15, 18, 18, 18, 18, 22, 22, 22, 22, 22]
expect_upper = [100,8, 8, 8, 8, 10, 10, 10, 10, 14, 14, 14, 14, 19, 19, 19, 19, 22, 22, 22, 22, 25, 25, 25, 25, 25]
service      = [0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

### the smaple solution got by GA algorithm
sampleSolution1 = [[3, 5, 1, 8, 2, 4, 25, 23, 24, 21, 6, 7, 9, 10, 15, 14, 12, 11, 17, 16, 18, 13, 19, 22, 20], [2.0, 2.8, 3.8, 4.6, 5.5, 6.1, 18.9, 20.299999999999997, 20.699999999999996, 23.099999999999994, 5.8, 6.2, 7.0, 8.9, 9.8, 10.700000000000001, 12.3, 12.700000000000001, 12.000000000000002, 13.000000000000002, 14.400000000000002, 16.200000000000003, 16.0, 17.3, 19.3]]
sampleSolution2 = [[6, 3, 7, 1, 5, 2, 8, 11, 12, 4, 15, 14, 9, 10, 18, 16, 20, 13, 17, 25, 23, 19, 22, 24, 21], [0.2, 1.2, 1.9, 2.3, 3.3, 4.6, 4.8, 5.3999999999999995, 5.8, 7.4, 8.700000000000001, 9.600000000000001, 10.700000000000001, 12.600000000000001, 10.7, 12.1, 14.399999999999999, 16.099999999999998, 14.899999999999999, 15.7, 17.099999999999998, 18.7, 17.3, 18.0, 20.4]]
Popuation = 200
CustNumber = 25
## The number of orders which the online platform provide
StartStation = 0
## The id of Start deport 
Terminal = 0
## The terminal id of terminal station (airport)
VehicleCpacity = 7
## capacity for per CAR
MaxVehicle = int(CustNumber / VehicleCpacity) + 2
delay_cost = 4000
waitcost = 2000
VehStartCost = 200
# Penaty for early arrival and late arrival
speed = 40
# The average speed for car
costunitdist = 10
## 绕行距离惩罚
penaty_detour = 300

Prob_delay = 0.2

Ratio_new =  0.2
## 新顾客产生的比率

MAX_TW = 36
MIN_TW = 4
MAX_X = 120
MIN_X = -10
MAX_Y = 120
MIN_Y = -10



class GA:
	def __init__(self):
		'''
		best: the best gene which obtained in last GA algorithm step
		'''
		self.solution = [[19, 25, 21, 24, 18, 16, 23, 22, 20, 15, 17, 10, 13, 6, 8, 11, 2, 4, 9, 12, 14, 5, 7, 1, 3], [16.3, 16.5, 18.0, 20.4, 17.5, 18.9, 19.5, 19.9, 8.2, 10.2, 10.6, 13.0, 14.0, 4.6, 5.3999999999999995, 5.999999999999999, 6.599999999999999, 7.199999999999998, 5.300000000000001, 6.300000000000001, 7.9, 8.9, 3.8, 4.2, 4.9]]
		## 上一阶段得到的染色体
		self.subroutes = [[19, 25, 21, 24], [18, 16, 23, 22], [20, 15, 17, 10, 13], [6, 8, 11, 2, 4], [9, 12, 14, 5], [7, 1, 3]]
		self.subtime = [[16.3, 16.5, 18.0, 20.4], [17.5, 18.9, 19.5, 19.9], [8.2, 10.2, 10.6, 13.0, 14.0], [4.6, 5.3999999999999995, 5.999999999999999, 6.599999999999999, 7.199999999999998], [5.300000000000001, 6.300000000000001, 7.9, 8.9], [3.8, 4.2, 4.9]]
		self.demand = initial_demand


class insertion:
	def __init__(self,ga_solution):
		self.current = 0
		self.solution = ga_solution.solution[0]
		self.subroutes = ga_solution.subroutes ## 子路径 [[],[]]
		self.subtime = ga_solution.subtime## 子路径对应的到达时刻 [[],[]]
		self.remain_subroutes = []
		self.remain_subtime =[]
		self.subload = [] ## 从、
		self.subloaded = []
		self.sub_remain_capacity = []
		self.substart = []
		self.demand = ga_solution.demand #订单需求,包括需求人数、时间窗期望，
		## [[demand]，[expect_low]，[expect_upper],[corX][corY]]
		## [[coorx].[coory]]
		###将demand改成字典格式。id为key,值为四个属性、
		self.subterminal = []
		self.visited_order = []
		self.visited_ordertime = []
		self.get_initload()

	def get_initload(self):
		## 得到原始路线各条子路径的载客量
		for subroute in self.subroutes:
			load = 0
			for order in subroute:
				load += self.demand[0][order]
			self.subload.append(load)


	def visited(self):
		## 让目前的路径随着时间更新,并返回未遍历
		## + 路径 + 路径的预计出发时 + 接待下一个顾客点之后
		sub_start = []
		temp_subtime =copy.deepcopy(self.subtime)
		temp_subroutes = copy.deepcopy(self.subroutes)
		for i in range(len(temp_subtime)):
			load = 0
			for j in range(len(temp_subtime[i])):
				load += self.demand[0][self.subroutes[i][j]]
				if temp_subtime[i][j] > self.current:
					#self.subroutes[i].insert(j,-1)
					#self.subtime[i].insert(j,-1)
					#标记现在路径的状态
					self.subloaded.append(load)
					self.substart.append(temp_subtime[i][j])
					if j+1 <= len(temp_subtime[i]):
						##防止超出索引
					   self.remain_subroutes.append(temp_subroutes[i][j:])
					   self.visited_order.append(temp_subroutes[i][:j])
					   self.remain_subtime.append(temp_subtime[i][j:])
					   self.visited_ordertime.append(temp_subtime[i][:j])
					assert(len(self.remain_subroutes) == len(self.remain_subtime))
					   #更新remain,断言
					break ##跳出第二个循环
		## 此函数将已访问的订单与还未访问的订单区别开来，方便下一步插入优化操作


	def dynamic_delay(self):
		#延误，随机选取还未遍历的顾客点，延后其时间窗
		calculateDist = lambda x1, y1, x2, y2: math.sqrt(((x1-x2) ** 2) + ((y1 - y2) ** 2 ))
		record_delay = []

		current_time = self.current
		for subroute,subtime in zip(self.remain_subroutes,self.remain_subtime):
			for i in subroute[1:]:
				if random.random() < Prob_delay:
					delaytime_low =  random.randint(6, 10)
					delaytime_upper =  random.randint(8, 10)
					self.demand[1][i] = min(self.demand[1][i] + delaytime_low,MAX_TW)
					self.demand[2][i] = min(self.demand[2][i] + delaytime_upper,MAX_TW)
					print("******************************\n订单%d出现延误,变化前的时间窗：[%d,%d].变化后的时间窗：[%d,%d]"%(i,self.demand[1][i]-delaytime_low,\
						self.demand[2][i]-delaytime_upper,self.demand[1][i],self.demand[2][i]),"变化前所在路径:",subroute)
					record_delay.append(i) ##还需记录延误的订单

		sub_count =0
		for subroute,subtime in zip(self.remain_subroutes,self.remain_subtime):
			load = 0
			for i in record_delay:
				if i in subroute:
					time_index = subroute.index(i)
					subroute.remove(i)##如果改点延误，则将其从子路径中删除
					subtime.pop(time_index)##子路径的时间也随之删除
					load += self.demand[0][i]
			self.sub_remain_capacity.append(VehicleCpacity - (self.subload[sub_count] - load))
			sub_count += 1

			## 重新更新时间 子路径的时间
		#返回delay 对应的I
		return record_delay

	def dynamic_new(self):
		#生成随机动态需求
		start = len(self.demand[0])-1
		max_number = int(CustNumber * Ratio_new)  
		real_number = random.randint(0, max_number)       
		new_corx =[]; new_cory=[];  new_number=[]; new_low=[]; new_upper =[]                                                                                                                                                                                                                                                                                                                                                              
		for i in range(real_number):
			temp_corx = random.randint(MIN_X, MAX_X)
			temp_cory = random.randint(MIN_Y, MAX_Y)
			temp_number = random.randint(1,3)
			temp_expect_low = self.current + random.randint(5,10)
			temp_expect_upper = temp_expect_low + random.randint(4, 6)
			##随机生成
			demand =[temp_number,temp_expect_low,temp_expect_upper,temp_corx,temp_cory]
			for i in range(len(demand)):
				self.demand[i].append(demand[i])
				#将新产生的需求加入demand列表中去

		end = len(self.demand[0]) - 1
		order_number = end - start 
		order = [i for i in range(start,end)]
		## 返回新生成的需求
		print("******************************\n新生成订单%d个，订单号分别是"%real_number,order)
		return [start,end]


	def normalize(self,coorX,coorY,time):
		assert(len(coorX) == len(coorY) == len(time))
		for i in range(len(coorX)):
			coorX[i] = (coorX[i] - MIN_X) / (MAX_X - MIN_X)
			coorY[i] = (coorY[i] - MIN_Y) / (MAX_Y - MIN_Y)
			time[i] = (time[i]- MIN_TW) / (MAX_TW - MIN_TW)
		##归一化


	def merge(self):
		##合并延误和动态需求，生成新的需求
		##生成变邻域搜索的初始解
		#思路：将延误的站点和新生成的订单在时空上表现出来，再将其插入到已有线路，如果运力不够，则开新的路线
		record_delay = self.dynamic_delay()
		new_index = self.dynamic_new()
		old_label =[]
		old_id = []
		label = 0
		for subroute in self.remain_subroutes:
			for i in subroute:
				old_label.append(label)
				###标记点的分组
				old_id.append(i)
				## 记录ID
			label += 1


		coorX_old = [self.demand[3][index] for index in old_id]
		coorY_old = [self.demand[4][index] for index in old_id]
		time_old =  [(self.demand[1][index] + self.demand[2][index])/2  for index in old_id]
		## old id
		new_id_new = [i for i in range(new_index[0],new_index[1])]
		new_id_delay = record_delay
		##合并两个ID

		new_label_new = [-1 for i in range(new_index[0],new_index[1])]
		new_label_delay = [-2 for i in range(len(record_delay))]

		new_id =new_id_delay + new_id_new 
		new_label = new_label_delay + new_label_new 
		##保证延误的订单优先级比新加入的订单的优先级高
		##合并
		coorX_new = [self.demand[3][index] for index in new_id]
		coorY_new = [self.demand[4][index] for index in new_id]
		time_new =  [(self.demand[1][index] + self.demand[2][index])/2  for index in new_id]
		## new id
		##X下面是KNN算法
		#还要将record_delay 里的加到新的里面
		##首先将得到的数据归一化
		self.normalize(coorX_old,coorY_old,time_old)
		self.normalize(coorX_new,coorY_new,time_new) 
   #生成散点.利用c控制颜色序列,s控制大小

		fig = plt.figure()
		 
		# 将二维转化为三维
		axes3d = Axes3D(fig)
		 
		# axes3d.scatter3D(x,y,z)
		# 效果相同
		axes3d.scatter(coorX_old,coorY_old,time_old,color = 'g',s = 30,marker='o')
		## 原有线路
		axes3d.scatter(coorX_new[:len(new_label_new)],coorY_new[:len(new_label_new)],time_new[:len(new_label_new)],color = 'r',s = 30,marker='o')

		axes3d.scatter(coorX_new[len(new_label_new):],coorY_new[len(new_label_new):],time_new[len(new_label_new):],color = 'b',s = 30,marker='o')


		plot_mark = 0
		plot_subroute_x = []
		plot_subroutes_x = []
		for i,label in enumerate(old_label):
			if label == plot_mark:
				plot_subroute_x.append(coorX_old[i])
			else:
				plot_subroutes_x.append(plot_subroute_x)
				plot_subroute_x = [coorX_old[i]]
				plot_mark = label
		if plot_subroute_x != []:
			plot_subroutes_x.append(plot_subroute_x)		

		plot_mark = 0
		plot_subroute_y = []
		plot_subroutes_y = []
		for i,label in enumerate(old_label):
			if label == plot_mark:
				plot_subroute_y.append(coorY_old[i])
			else:
				plot_subroutes_y.append(plot_subroute_y)
				plot_subroute_y = [coorY_old[i]]
				plot_mark = label
		if plot_subroute_y != []:
			plot_subroutes_y.append(plot_subroute_y)		

		plot_mark = 0
		plot_subroute_time = []
		plot_subroutes_time = []
		for i,label in enumerate(old_label):
			if label == plot_mark:
				plot_subroute_time.append(time_old[i])
			else:
				plot_subroutes_time.append(plot_subroute_time)
				plot_subroute_time = [time_old[i]]
				plot_mark = label
		if plot_subroute_time != []:
			plot_subroutes_time.append(plot_subroute_time)

		for X,Y,Z in zip(plot_subroutes_x,plot_subroutes_y,plot_subroutes_time):
			axes3d.plot(X,Y,Z)

		plt.xlabel('X',color = 'b',size = 10)
		plt.ylabel('Y',color = 'b',size = 10)
		axes3d.set_zlabel('Z',color = 'b',size=10)
		axes3d.set_xlim(0,1)
		axes3d.set_ylim(0,1)
		axes3d.set_zlim(0,1)
		plt.show()

		calculateDist = lambda x1, y1, z1, x2, y2, z2: math.sqrt(((x1-x2) ** 2) + \
			((y1 - y2) ** 2 ) + (z1-z2) ** 2)
		k = int(len(coorX_old) * 1) ## KNN算法的 K值
		


		for i in range(len(new_id)):
			Distance = [ calculateDist(coorX_new[i], coorY_new[i], time_new[i], \
				coorX_old[j], coorY_old[j], time_old[j]) for j in range(len(coorX_old))]
			Dis_matrix  = np.array(Distance)
			K_index = [index for index in Dis_matrix.argsort()[0:k]]
			K_mark =[old_label[index] for index in K_index]
			## argsort 函数
			##计算出的K_mark是距离排序后其值所对应的在Dis_matrix所对应的索引，选取其中最好的k个。
			##再找出label出现最多的第一个点，再将新的点插入至这个点的后续第一个节点，行成动态优化的初始方案
			###获取列表中出现次数最多的label,按照从多到少的顺序排序，得到一个元组列表[(,),(,)]

			labels = collections.Counter(K_mark).most_common(len(self.remain_subroutes))
			##再找到这个label在K_mark里面第一次出现的索引值
			success = False #判断此点是否插入成功
			for label,count in labels:
				## 对于每一一条候选的线路，必须在其容量满足的情况下才能插入
				if self.sub_remain_capacity[label] >= self.demand[0][new_id[i]]:
					new_label[i] = label               
					first_index = K_mark.index(label)
					##在根据索引值从K_index中找到其在Dis_matrix中的索引 
					Nearsest_index = K_index [first_index]
					##根据在Dis_matrix中的索引找到其在old_idz中的ID
					Nearsest_ID = old_id[Nearsest_index]  
					##将这个新的点插入到旧的线路中去
					##更新self.remain_subroutes：
					for subroute,subtime in zip(self.remain_subroutes,self.remain_subtime):
						for sub_id_index in range(len(subroute)):
							if subroute[sub_id_index] == Nearsest_ID:
								##将符合条件的订单插入至子路径中
								print("******************************\n订单%d插入到了最邻近点%d后面"%(new_id[i],Nearsest_ID))
								subroute.insert(sub_id_index + 1,new_id[i])
								subtime.insert(sub_id_index + 1,new_id[i])
					self.sub_remain_capacity[label] -= self.demand[0][new_id[i]]
					## 更新剩余路径的车辆容量
					success = True 
					break  
			if not success:
				print("订单%d未能插入成功，因为原始路径的容量不允许"%new_id[i])   

			## new_label中找出还未被替换的车辆，重新未其分配车辆路线。
			##加入新的路线至subroutes 新路线的条件是满载率超过50%， 如果不超过50%，将剩余的订单随机剔除
			##得到还未插入到旧路线的id与label(判断其是延误订单还是新订单)
		new_route = [[],[]] #[[id],[label]]
		for i in range(len(new_label)):
			if new_label[i] <= -1:
				new_route[0].append(new_id[i])
				new_route[1].append(new_label[i])
				new_label[i] = -3
		

		if new_route[0] != []:
			##如果真实存在还未插入的订单
			##则将其重新加入到一条路线中去
			print("生成预备新路径", new_route[0])
			sub_newroute = []
			load = 0		
			for order in new_route[0]: 
				if load <= VehicleCpacity:
					order_demand = self.demand[0][order]
					load += order_demand
					sub_newroute.append(order)
					continue
				else:
					self.remain_subroutes.append(sub_newroute)
					sub_newroute = []
					load = 0
			if sub_newroute !=[]:
				self.remain_subroutes.append(sub_newroute)

			 #检查前面的步骤已对new_label全部进行操作


	def update(self):
		## 每一个时间间隔随机生成动态的需求并更新现阶段的状态
		self.visited()
		## 当前车辆行驶状态

		self.merge()
		##KNN算法将动态需求插入到原有行驶路线中
		##通过前面两个步骤已经得到更新后的subroutes,以及旧路径的出发时刻，
		##接下来需要重新运用变邻域算法来优化插入方案。


		self.VNS()
		##VNS的输入有self.remain_subroutes, self.subtime, self.substart 
		##对于VNS算法，它的输入是insetion类的属性值，但在VNS算法中不能
		##insertion类的属性值改变，只能在最后算出结果来之后才将属性值更新。
		## 因此对于VNS下属的functions,除了全局变量以外，
		## 其余一律不得使用insertion类的属性值



	def get_init_fit(self):

		subterminal = self.updatetime(self.remain_subtime, self.remain_subroutes)
		fit = self.obj_value(self.remain_subroutes,subterminal)
		self.remain_subroutes.append(fit)

	def VNS(self):
		## 变邻域算法，通过邻域之间的跳动搜索、抖动算子，来寻找较优解\
		VNS_iterations = 100
		neighbourhood_size = 10
		count = 0
		self.get_init_fit()
		initial_route = copy.deepcopy(self.remain_subroutes)
		initial_time = copy.deepcopy(self.remain_subtime)
		print("******************************\n生成初始路径（列表尾值为其目标值））：",initial_route)

		best_solution = self.remain_subroutes
		initial_route.pop(-1)## 去掉初始方案尾部的目标函数值
		while (count < VNS_iterations):## 不断地在邻域内来搜索
			for i in range(0, neighbourhood_size):##搜索某个邻域里的解空间
				for j in range(0, neighbourhood_size):
					solution = self.stochastic_2_opt(initial_route)
				solution = self.local_search(solution)
				if (solution[-1] < best_solution[-1]):
					best_solution = copy.deepcopy(solution) 
					break
					##如果找到了一个比当前最优解还好的解，则将最优解更新，并跳出循环，重新从第一个邻域开始搜索。
			count = count + 1
			print("Iteration = ", count, "->", best_solution)
		return best_solution


	# Function: Stochastic 2_opt
	def stochastic_2_opt(self,route):
		####随机选择两条路径交换节点
		##规则：每次随机选取两个在空间上距离最近的点，交换
		##目前可以用随机选取两个点交换
		remaining_subroutes = copy.deepcopy(route) 
		remaining_subtime = self.remain_subtime
		##得到余下的子路径  
		##将余下的子路径按照一定规则交换
		##[[],[]...]
		#随机选取两条子线路
		subroute1,subroute2 = random.sample(range(len(remaining_subroutes)), 2)
		#在第一条子路径中随机选一订单点
		point1 = random.sample(range(len(remaining_subroutes[subroute1])),1)[0]
		#在第二条子路径中随机选取一订单点
		point2 = random.sample(range(len(remaining_subroutes[subroute2])),1)[0]
		#将两个订单点交换
		remaining_subroutes[subroute1][point1],remaining_subroutes[subroute2][point2] = \
		remaining_subroutes[subroute2][point2],remaining_subroutes[subroute1][point1]

		subterminal = self.updatetime(remaining_subtime,remaining_subroutes)
		fit = self.obj_value(remaining_subroutes,subterminal)
		remaining_subroutes.append(fit)
		## 将remaining_subroutes 末尾添加fit,便于解的目标值的读取。
		## [[],[],[],2999.90]
		return remaining_subroutes


	# Function: Local Search
	def local_search(self, route, max_attempts = 50, neighbourhood_size = 5):
		count = 0
		solution = copy.deepcopy(route) 
		while (count < max_attempts): 
			for i in range(0, neighbourhood_size):
				candidate = self.stochastic_2_opt(route[:-1])
			if candidate[-1] < solution[-1]:
				#如果当前解优于一开始设定得最优解，则将局部搜索设定得
				#开始解设定为当前解 继续从头开始搜索，直至无法找出比当前要优秀的解。
				solution  = copy.deepcopy(candidate)
				count = 0
			else:
				count = count + 1                             
		return solution 
	# Function: Variable Neighborhood Search


	def updatetime(self,remain_subtime,remain_subroutes):
		calculateDist = lambda x1, y1, x2, y2: math.sqrt(((x1-x2) ** 2) + ((y1 - y2) ** 2 ))
		subterminal = []
		for subtime,subroute in zip(remain_subtime,remain_subroutes):
			starttime = subtime[0]
			current_time = starttime
			i = 1
			while i < len(subroute):
				distance = calculateDist(self.demand[3][subroute[i]], self.demand[4][subroute[i]], self.demand[3][subroute[i-1]], \
				self.demand[4][subroute[i-1]])
				time  = round((distance / speed),1)
				### get travel time between two customer id 
				current_time = current_time + time
				subtime[i] = current_time
				i += 1
			### update the time for every customer id  
			Terminal_distance = calculateDist(self.demand[3][CustNumber+1],self.demand[4][CustNumber+1],self.demand[3][subroute[i-1]], self.demand[4][subroute[i-1]])				
			time = Terminal_distance / speed
			terminaltime = current_time + time
			subterminal.append(terminaltime)
		return subterminal


	# Function: Distance
	def obj_value(self,remain_subroutes,subterminal): 
		## 给一个剩余的子路径，此函数将计算出他的适应值
		calculateDist = lambda x1, y1, x2, y2: math.sqrt(((x1-x2) ** 2) + ((y1 - y2) ** 2 ))
		fit = 0; dist_cost = 0; 
		subroutes = remain_subroutes
		tempsub = copy.deepcopy(subroutes)
		dist = []
		for subroute in tempsub:
			#subroute.insert(0,0)  
			subroute.append(Terminal)
			i = 1
			while i < len(subroute):
				dist.append(calculateDist(self.demand[3][subroute[i]], self.demand[4][subroute[i]], self.demand[3][subroute[i-1]], \
				 self.demand[4][subroute[i-1]]))
				i += 1
		### calculate the total distance for a all subroutes
		dist_cost = sum(dist) * costunitdist

		time_cost = 0
		for subroute,subterminaltime in zip(subroutes,subterminal):
			## subterminal =[0,2,1] 每个子路径到达终点的时刻
			for cusid in subroute:
				wait = expect_low[cusid] - subterminaltime
				delay = subterminaltime - expect_upper[cusid]
				sub_time_cost = waitcost * max(expect_low[cusid] - subterminaltime,0) + \
				delay_cost * max(subterminaltime - expect_upper[cusid],0)
				time_cost = time_cost + sub_time_cost
				if DEBUG and wait > 0:
					print("订单%d早到%.1f个单位时间"%(cusid,wait))
				if DEBUG and delay > 0:
					print("订单%d延误%.2f个单位时间"%(cusid,delay))

		start_cost = 0
		start_number = len(subroutes)
		start_cost = start_number * VehStartCost
		## v车辆启动费用
		detour_cost = 0
		for subroute in tempsub:
			i = 0
			while i < len(subroute):
				current_origin = calculateDist(self.demand[3][subroute[i]], self.demand[4][subroute[i]], self.demand[3][subroute[0]], \
				 self.demand[4][subroute[0]])
				current_terminal = calculateDist(self.demand[3][subroute[i]], self.demand[4][subroute[i]], self.demand[3][subroute[-1]], \
				 self.demand[4][subroute[-1]])

				if i >= 1:
					cost1 = max(last_origin - current_origin,0) * penaty_detour
					cost2 = max(current_terminal - last_terminal,0) * penaty_detour
					detour_cost = detour_cost + cost1 + cost2

				last_origin = current_origin + 0
				last_terminal = current_terminal
				i += 1

		total_cost = time_cost + dist_cost + start_cost + detour_cost

		if DEBUG:
			print("time_cost",time_cost)
			print("dist_cost",dist_cost)
			print("start_cost",start_cost)
			print("detour_cost",detour_cost)
		fitness =  total_cost
		return fitness

def update_last(this_step,last_step):
	for i in this_step:
		pass



if __name__ == '__main__':

	initial_demand = [[],[],[],[],[]]
	initial_demand[0] = demand
	initial_demand[1] = expect_low
	initial_demand[2] = expect_upper
	initial_demand[3] = X_coordinate
	initial_demand[4] = Y_coordinate

	ga_solution = GA()
	##inherit BEST solution class which got in last step
	interval = 1
	## 将demand记录下来的同时，

	while interval <= 1:
		insertion1 = insertion(ga_solution)
		insertion1.current = interval
		insertion1.update()
		interval += 1
		
