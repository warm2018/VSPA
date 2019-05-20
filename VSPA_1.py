import random 
import math
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib import animation 
import itertools
import time


DEBUG = 0

sampleSolution1 = [[3, 5, 1, 8, 2, 4, 25, 23, 24, 21, 6, 7, 9, 10, 15, 14, 12, 11, 17, 16, 18, 13, 19, 22, 20], [2.0, 2.8, 3.8, 4.6, 5.5, 6.1, 18.9, 20.299999999999997, 20.699999999999996, 23.099999999999994, 5.8, 6.2, 7.0, 8.9, 9.8, 10.700000000000001, 12.3, 12.700000000000001, 12.000000000000002, 13.000000000000002, 14.400000000000002, 16.200000000000003, 16.0, 17.3, 19.3]]
sampleSolution2 = [[6, 3, 7, 1, 5, 2, 8, 11, 12, 4, 15, 14, 9, 10, 18, 16, 20, 13, 17, 25, 23, 19, 22, 24, 21], [0.2, 1.2, 1.9, 2.3, 3.3, 4.6, 4.8, 5.3999999999999995, 5.8, 7.4, 8.700000000000001, 9.600000000000001, 10.700000000000001, 12.600000000000001, 10.7, 12.1, 14.399999999999999, 16.099999999999998, 14.899999999999999, 15.7, 17.099999999999998, 18.7, 17.3, 18.0, 20.4]]

Popuation = 200
## The number of chromosome in a pupolation
Iteration = 500
## The number of iterations in one process
CustNumber = 25
## The number of orders which the online platform provide
StartStation = 0
## The id of Start deport 
Terminal = CustNumber + 1
## The terminal id of terminal station (airport)
VehicleCpacity = 7
## capacity for per CAR

VaryProb = 0.1
## the probability of gene mutation
EliteProb = 0.2
## the probability of choosing elites ihen operate mutation process
SubsaveProb = 0.2
## the subroutes saving probility for crossing

## SOME base parameters for this model
X_coordinate = [-10,56, 66, 56, 88, 88, 24, 40, 32, 16, 88, 48, 32, 80, 48, 23, 48, 16, 8, 32, 24, 72, 72, 72, 88, 34, 120]
Y_coordinate = [-10,56, 78, 27, 72, 32, 48, 48, 80, 69, 96, 96, 104, 56, 40, 16, 8, 32, 48, 64, 96, 104, 32, 16, 8, 56, 60]
demand       = [0,1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1,0]
expect_low   = [0,4, 4, 4, 4, 6, 6, 6, 6, 10, 10, 10, 10, 15, 15, 15, 15, 18, 18, 18, 18, 22, 22, 22, 22, 22,0]
expect_upper = [100,8, 8, 8, 8, 10, 10, 10, 10, 14, 14, 14, 14, 19, 19, 19, 19, 22, 22, 22, 22, 25, 25, 25, 25, 25,100]
service      = [0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0]

MaxVehicle = int(CustNumber / VehicleCpacity) + 2
delay_cost = 4000
waitcost = 2000
VehStartCost = 200
# Penaty for early arrival and late arrival
speed = 40
# The average speed for car
costunitdist = 10

penaty_detour = 300
## 绕行距离惩罚

class Gene:
	def __init__(self,name='Gene',data = None):
		self.name = name
		self.length = CustNumber 
		self.subterminal = []
		if data is None:
			self.data = self._getGene(self.length)
			self.subroutes, self.subtime = self._readsub(self.data)
			self._updatetime()
		else:
			assert(self.length  == CustNumber)
			self.data = data
			self.subroutes, self.subtime = self._readsub(self.data)
			self._updatetime()			
		self.fit = self.getFit()
		self.chooseProb = 0

	def _generate(self,length):
		data = [[],[]]
		data1 = [i for i in range(1,length + 1)]
		random.shuffle(data1)

		## random the list order
		## customer chromosome
		data2 = [round(random.random()*36,1) for i in range(CustNumber)]
		## 36 个 interval  每一个 表示10分钟
		data[0]=data1
		data[1]=data2
		## time interval chromosome
		return data

	def _getGene(self,length):
		data = self._generate(length)
		return data

	def _readsub(self,data):
		data_1 = data[0]
		data_2 = data[1]
		route = []
		time = []
		sub_route = []
		sub_time = []
		vehicle_load = 0
		last_customer_id = 0
		for i in range(len(data_1)):
			temp_demand = demand[data_1[i]]
			updated_vehicle_load = vehicle_load + temp_demand
			if updated_vehicle_load <= VehicleCpacity:
				sub_route.append(data_1[i])
				sub_time.append(data_2[i])
				vehicle_load = updated_vehicle_load
			else:
				route.append(sub_route)
				time.append(sub_time)
				sub_route = [data_1[i]]
				sub_time = [data_2[i]]
				vehicle_load = temp_demand
		if sub_route != []:
			route.append(sub_route)
			time.append(sub_time)
		return route,time
		# This function return some subroutes for a chrosome
		# Like this : [[1,2,3],[3,6,7,8],[11,7,5,3]]
		
	def _updatetime(self):
		calculateDist = lambda x1, y1, x2, y2: math.sqrt(((x1-x2) ** 2) + ((y1 - y2) ** 2 ))
		for subtime,subroute in zip(self.subtime,self.subroutes):
			starttime = subtime[0]
			current_time = starttime
			i = 1
			while i < len(subroute):
				distance = calculateDist(X_coordinate[subroute[i]], Y_coordinate[subroute[i]], X_coordinate[subroute[i-1]], \
				Y_coordinate[subroute[i-1]])
				time  = round((distance / speed),1)
				### get travel time between two customer id 
				current_time = current_time + time
				subtime[i] = current_time
				i += 1
			### update the time for every customer id  
			Terminal_distance = calculateDist(X_coordinate[CustNumber+1],Y_coordinate[CustNumber+1],X_coordinate[subroute[i-1]], Y_coordinate[subroute[i-1]])
				
			time = Terminal_distance / speed
			terminaltime = current_time + time
			self.subterminal.append(terminaltime)

		datacount = 0
		for i in self.subtime:
			for j in i:
				self.data[1][datacount] = j
				datacount += 1
				

	def getFit(self):
		calculateDist = lambda x1, y1, x2, y2: math.sqrt(((x1-x2) ** 2) + ((y1 - y2) ** 2 ))
		fit = 0; dist_cost = 0; 
		subroutes = self.subroutes
		tempsub = deepcopy(subroutes)
		dist = []
		for subroute in tempsub:
			#subroute.insert(0,0)  
			subroute.append(Terminal)
			i = 1
			while i < len(subroute):
				dist.append(calculateDist(X_coordinate[subroute[i]], Y_coordinate[subroute[i]], X_coordinate[subroute[i-1]], \
				 Y_coordinate[subroute[i-1]]))
				i += 1
		### calculate the total distance for a all subroutes
		dist_cost = sum(dist) * costunitdist

		time_cost = 0
		for subroute,subterminaltime in zip(self.subroutes,self.subterminal):
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
		start_number = len(self.subroutes)
		start_cost = start_number * VehStartCost
		## v车辆启动费用


		detour_cost = 0
		for subroute in tempsub:
			i = 0
			while i < len(subroute):
				current_origin = calculateDist(X_coordinate[subroute[i]], Y_coordinate[subroute[i]], X_coordinate[subroute[0]], \
				 Y_coordinate[subroute[0]])
				current_terminal = calculateDist(X_coordinate[subroute[i]], Y_coordinate[subroute[i]], X_coordinate[subroute[-1]], \
				 Y_coordinate[subroute[-1]])

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

		fitness = 1.0 / total_cost

		return fitness

	def updateChooseProb(self,sumFit):
		self.chooseProb = self.fit / sumFit



	def moveRandSubPathLeft(self):
		route_temp = []
		time_temp = []
		other_route_temp = []
		other_time_temp = []
		moved_number = 0
		for subroute,subtime in  zip(self.subroutes,self.subtime):
			if random.random() < SubsaveProb:
				moved_number += 1
				i = 0
				while i < len(subroute):
					route_temp.append(subroute[i])
					time_temp.append(subtime[i])
					i += 1
			else:
				i = 0
				while i < len(subroute):
					other_route_temp.append(subroute[i])
					other_time_temp.append(subtime[i])
					i += 1
		mark_index = -1
		if moved_number != 0:
			route_temp.append(-1)
			time_temp.append(-1)
			mark_index = route_temp.index(-1)
		for i,j in zip(other_route_temp,other_time_temp):
			route_temp.append(i)
			time_temp.append(j)

		if mark_index != -1:
			self.data[0].insert(mark_index,-1)
			self.data[1].insert(mark_index,-1)
			## insert the mark for gene
		assert(len(self.data[0]) == len(route_temp))
		for i in range(len(route_temp)):
			self.data[0][i] = route_temp[i]
			self.data[1][i] = time_temp[i]
		#get moved data which has a position mark(-1)



def getRandomGenes(size):
	genes = []
	for i in range(size):
		genes.append(Gene("Gene " + str(i)))
	return genes

def getSumFit(genes):
	sumFit = 0
	for gene in genes:
		sumFit += gene.fit
	return sumFit

def updateChooseProb(genes):
	sumFit = getSumFit(genes)
	for gene in genes:
		gene.updateChooseProb(sumFit)

def getSumProb(genes):
	sum = 0
	for gene in genes:
		sum += gene.chooseProb
	return sum

def choose(genes):
	num = int(Popuation / 6) * 2 
	## choose 1/3 population
	key = lambda gene: gene.chooseProb
	genes.sort(reverse=True,key = key)
	## sort genes 
	return genes[0:num]

def crossPair(gene1,gene2,crossedGenes):
	gene1.moveRandSubPathLeft()
	gene2.moveRandSubPathLeft()
	newGene1 = [[],[]]
	newGene2 = [[],[]]
	for pos0,pos1 in zip(gene1.data[0],gene1.data[1]):
		if pos0 ==  -1:
			break
		newGene1[0].append(pos0)
		newGene1[1].append(pos1)

	for pos0,pos1 in zip(gene2.data[0],gene2.data[1]):
		if pos0 ==  -1:
			break
		newGene2[0].append(pos0)
		newGene2[1].append(pos1)

	## 得到两基因的固定部分，然后在对余下剩余的基因进行补全
	## 余下基因补全规则：按照另一染色体的基因顺序
	for pos0,pos1 in zip(gene2.data[0],gene2.data[1]):
		if pos0 == -1:
			continue
		if pos0 not in newGene1[0]:
			newGene1[0].append(pos0)
	
			newGene1[1].append(pos1)

	for pos0,pos1 in zip(gene1.data[0],gene1.data[1]):
		if pos0 == -1:
			continue
		if pos0 not in newGene2[0]:
			newGene2[0].append(pos0)
			newGene2[1].append(pos1)
	## 此时有多种处理方式
	gene1_temp = deepcopy(newGene1)
	gene2_temp = deepcopy(newGene2)
	possible1 = Gene(data=gene1_temp.copy())
	possible2 = Gene(data=gene2_temp.copy())
	assert(possible1)
	assert(possible2)
	crossedGenes.append(possible1)
	crossedGenes.append(possible2)

def cross(genes):
	crossedGenes = []
	for i in range(0,len(genes),2):
		crossPair(genes[i],genes[i+1],crossedGenes)
	return crossedGenes


def mergeGenes(genes,crossedGenes):
	key = lambda gene: gene.chooseProb
	genes.sort(reverse=True,key=key)
	pos = Popuation - 1
	for gene in crossedGenes:
		genes[pos] = gene
		pos -= 1
	## 从后往前替换
	return genes


def varyOne(gene):
	varyNum = 20
	variedGenes = []
	for i in range(varyNum):
		p1,p2 = random.choices(list(range(0,len(gene.data[0]))),k=2)
		newGene = deepcopy(gene.data)
		newGene[0][p1],newGene[0][p2] = newGene[0][p2], newGene[0][p1]
		#newGene[1][p1],newGene[1][p2] = newGene[1][p2], newGene[1][p1]
		for subroute in gene.subroutes:	
			count  = 0
			for i in range(len(subroute)):
				if i == len(subroute) - 1:
					newGene[1][count] = round(random.uniform(2,36), 1)
				count += 1		
		variedGenes.append(Gene(data=deepcopy(newGene)))
	key = lambda gene: gene.fit
	variedGenes.sort(reverse=True,key=key)
	return variedGenes[0]

'''
		for subroute in gene.subroutes:	
			count  = 0
			for i in range(len(subroute)):
				if i == len(subroute) - 1:
					newGene[1][count] = round(random.random() * 36,1)
				count += 1
'''

def vary(genes):
	for index,gene in enumerate(genes):
		##精英主义
		if index < Popuation * EliteProb:
			continue
		if random.random() < VaryProb:
			genes[index] = varyOne(gene)
	return genes


def plot(gene,ax2):
	ax2 = plt.axes(projection='3d')
	routes_temp = deepcopy(gene.subroutes)
	time_temp = deepcopy(gene.subtime)
	plot_list_X = []
	plot_list_Y = []
	plot_list_Z = []
	color_set = itertools.cycle(['b','g','r','c','m','k'])
	elev = 90
	azim = 0
	ax2.view_init(elev=elev,azim=azim)

	for subroute,subtime,terminaltime in zip(routes_temp, time_temp,gene.subterminal):
		## subroute.insert(0,0)  单车场运用的方法
		subroute.append(CustNumber + 1)
		subtime.append(terminaltime)
		Xorder = [X_coordinate[i] for i in subroute]
		Yorder = [Y_coordinate[i] for i in subroute]
		Zorder = subtime
		plot_list_X.append(Xorder)
		plot_list_Y.append(Yorder)
		plot_list_Z.append(Zorder)

	for Xorder,Yorder,Zorder in zip(plot_list_X,plot_list_Y,plot_list_Z):
		print(Xorder,Yorder,Zorder)

		ax2.scatter3D(Xorder,Yorder,Zorder,alpha=0.3)     #生成散点.利用c控制颜色序列,s控制大小
		ax2.plot3D(Xorder,Yorder,Zorder,c=next(color_set),zorder=1)

	ax2.set_xlabel('Coordination_X', fontsize=next(fontsizes))
	ax2.set_ylabel('Coordination_Y', fontsize=next(fontsizes))
	ax2.set_title('Routes Update', fontsize=next(fontsizes))

	'''
	ax2.scatter(X_coordinate, Y_coordinate,zorder=2)
	ax2.scatter([X_coordinate[0]], [Y_coordinate[0]],marker='o',zorder=3)
	for i in range(CustNumber+2):
		ax2.annotate('{}'.format(i),(X_coordinate[i],Y_coordinate[i]))
		ax2.scatter([X_coordinate[CustNumber+1]], [Y_coordinate[CustNumber+1]],marker='o',zorder=3)
	'''

	print("data",gene.data)
	print("subroutes",gene.subroutes)
	print("subtime",gene.subtime)
	print("subtime",gene.fit)


def plot_3D(Gene,ax2):
	routes_temp = deepcopy(gene.subroutes)
	time_temp = deepcopy(gene.subtime)
	plot_list_X = []
	plot_list_Y = []
	color_set = itertools.cycle(['b','g','r','c','m','k'])

	ax2.scatter(X_coordinate, Y_coordinate,zorder=2)
	ax2.scatter([X_coordinate[0]], [Y_coordinate[0]],marker='o',zorder=3)
	for i in range(CustNumber+2):
		ax2.annotate('{}'.format(i),(X_coordinate[i],Y_coordinate[i]))
		ax2.scatter([X_coordinate[CustNumber+1]], [Y_coordinate[CustNumber+1]],marker='o',zorder=3)

	for subroute,subtime in zip(routes_temp, time_temp):
		subroute.insert(0,0)
		subroute.append(CustNumber + 1)
		Xorder = [X_coordinate[i] for i in subroute]
		Yorder = [Y_coordinate[i] for i in subroute]
		plot_list_X.append(Xorder)
		plot_list_Y.append(Yorder)

	for Xorder,Yorder in zip(plot_list_X,plot_list_Y):		
		ax2.plot(Xorder,Yorder,c=next(color_set),zorder=1)
	## 连线

	ax2.set_xlabel('Coordination_X', fontsize=next(fontsizes))
	ax2.set_ylabel('Coordination_Y', fontsize=next(fontsizes))
	ax2.set_title('Routes Update', fontsize=next(fontsizes))



	print("data",gene.data)
	print("subroutes",gene.subroutes)
	print("subtime",gene.subtime)
	print("subtime",gene.fit)


if __name__ == '__main__' and not DEBUG :
	genes = getRandomGenes(Popuation)
	best = []
	fontsizes = itertools.cycle([10,10,14,10,10,14])
	ax1 = plt.subplot(1,2,1)
	ax2 = plt.subplot(1,2,2)

	ax1.set_xlabel('Generation', fontsize=next(fontsizes))
	ax1.set_ylabel('Fitness', fontsize=next(fontsizes))
	ax1.set_title('Fitenss Update', fontsize=next(fontsizes))

	plt.ion()
	for i in range(Iteration):
		plt.cla()
		updateChooseProb(genes)
		sumProb = getSumProb(genes)
		chosenGenes = choose(deepcopy(genes))
		## 选择的同时，按照适应度大小对种群进行了排序，选择了最优秀的多个个体进行交叉
		crossedGenes = cross(chosenGenes)
		genes = mergeGenes(genes, crossedGenes)
		genes = vary(genes)
		print("GENERATION %d"%i)

		key = lambda gene: gene.fit
		genes.sort(reverse=True,key=key)
		best.append(genes[0].fit)
		ax1.plot(range(i+1),best)
		plot(genes[0],ax2)
		plt.pause(0.01)
		print("Processing Time",time.clock())
	total_best = genes[0]

	print("Find The best gene is: ", total_best.data )
	print("The best fitness is : ",total_best.fit )
	plt.ioff()  


if DEBUG:
	print("START")
	fontsizes = itertools.cycle([10,10,14,10,10,14])
	gene = Gene(data=sampleSolution1)
	#ax2 = plt.subplot()
	#plot(gene, ax2)
	#plt.pause(60)
	for subroute in gene.subroutes:
		load = 0
		for order in subroute:
			load += demand[order]
		print(load)

	print(gene.subroutes)
	print(gene.subterminal)
	print(gene.subtime)
	print(gene.fit)
	print("FINISH")
