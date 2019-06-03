import os
import io
import itertools
import random
from utils import make_dirs_for_file, exist, load_instance
import numpy as np
import matplotlib.pyplot as plt
import copy
from gurobipy import *

Capacity = 7
Terminal = 0
Unit_cost = 5
Start = 200

def generate_orders(instance_name='R105'):
	json_data_dir = os.path.join('..', 'data', 'json')
	print(json_data_dir)
	json_file = os.path.join(json_data_dir, '{}.json'.format(instance_name))
	global instance
	instance = load_instance(json_file=json_file)
	if instance is None:
		return
	return instance
 

def get_seedset(orders):
	order_temp = copy.deepcopy(orders)
	set_big = {}
	for order_i in orders: # order_i 为种子客户
		set_small = []
		order_temp.remove(order_i)
		for order_j in order_temp:
			jude_overlap = instance['{}'.format(order_i)]['Latest'] <= \
			instance['{}'.format(order_j)]['Earliest'] or \
			instance['{}'.format(order_i)]['Earliest'] >= \
			instance['{}'.format(order_j)]['Latest'] 
			## 如果两者时间窗没有交集，则为1
			if not jude_overlap: #如果两者有时间窗交集
				set_small.append(order_j)
		set_big[order_i] = set_small
	return set_big

def get_number(combination):
	number_list = [instance['{}'.format(i)]['demand'] for i in combination]
	i = 0;total_number=0
	while i < len(number_list):
		total_number += number_list[i]
		i += 1
	return total_number


def get_cost(subroute):
	subroute.append(Terminal)
	i = 1
	dist = []
	while i < len(subroute):
		dist.append(instance['distance_matrix'][subroute[i]][subroute[i-1]])
		i += 1
	### calculate the total distance for a all subroutes
	dist_cost = sum(dist) * Unit_cost
	start_cost = Start
	return dist_cost + start_cost

def jude_time(combine): ## 判断路径是否符合有效路径约束
	for com in itertools.combinations(combine, 2):
		order_i = com[0]
		order_j = com[1]
		jude_overlap = instance['{}'.format(order_i)]['Latest'] <= \
			instance['{}'.format(order_j)]['Earliest'] or \
			instance['{}'.format(order_i)]['Earliest'] >= \
			instance['{}'.format(order_j)]['Latest'] 
		if jude_overlap:
			return False
		else:
			return True

def get_comblations(key,Seed_set,seed_number):
	Result_routes = []
	Result_costs = []
	remain_capacity = Capacity - seed_number
	print(Seed_set)
	if Seed_set != []:
		for i in range(1,min(len(Seed_set),remain_capacity) + 1):
			for combine in itertools.combinations(Seed_set, i):
				if i >=2:
					time_windows = jude_time(combine)
				else:
					time_windows = True
				if get_number(combine) <= remain_capacity and time_windows: ##满足容量约束
					combine_after = list(combine)
					combine_after.insert(0,key)
					assert(get_number(combine_after) <= Capacity)
					best_cost = 10000
					for route_sequence in itertools.permutations(combine_after, len(combine_after)):
						subroute = list(route_sequence) ##将元组的排列改为列表的排列
						current_cost = get_cost(subroute)
						if current_cost <= best_cost:
							best_cost = current_cost
							best_subroute = subroute
					assert(best_cost <= 10000)		
					Result_routes.append(best_subroute)
					Result_costs.append(best_cost)
	else:
		subroute = [key]
		cost = get_cost(subroute)
		Result_routes.append(subroute) 
		Result_costs.append(cost)

	return Result_routes, Result_costs


def get_total(set_big):
	total_routes = []
	total_cost = []
	for key,values in set_big.items():
		seed_number = instance['{}'.format(key)]['demand']
		Seed_set = values
		Result_routes, Result_costs = get_comblations(key,Seed_set, seed_number)
		total_routes = total_routes + Result_routes
		total_cost = total_cost + Result_costs
	return total_routes, total_cost


def solve_problem(total_routes,total_cost):
	m = Model('airport')
	orders_number = orders_number = len(instance) - 4
	wideth = [i for i in range(1,orders_number + 1)]
	length = [i for i in range(len(total_routes))]

	#airport = m.addVars(wideth, length,vtype=GRB.BINARY,name="airport")
	Binary_list = []
	Binary_value = []
	for i in wideth:
		for j in range(len(total_routes)):
			Binary_list.append((i,j))

	jude = m.addVars(Binary_list,vtype=GRB.BINARY,name="jude")
	for i in wideth:
		for j,value in enumerate(total_routes):
			if i in value:
				jude[i,j] = 1
			else:
				jude[i,j] = 0
	choose = m.addVars(length,vtype=GRB.BINARY,name="choose")
	for i in wideth:
		m.addConstr(sum(jude[i,r] * choose[r] for r in range(len(length))) == 1,name='{}'.format(i))
	m.setObjective(quicksum(total_cost[i] * choose[i] for i in length), GRB.MINIMIZE)
	m.optimize()

	result = []

	if m.status == GRB.Status.OPTIMAL:
		solution = m.getAttr('x',choose)
		for i in length:
			if solution[i] == 1:
				result.append(total_routes[i])
	return result

def plot_result(result):
	X_plot = [instance['{}'.format(i)]['coordinates']['x'] for i in range(1,len(instance)-4+1)]
	Y_plot = [instance['{}'.format(i)]['coordinates']['y'] for i in range(1,len(instance)-4+1)]
	plt.scatter(X_plot,Y_plot)
	for subroute in result:
		X_plot = [];Y_plot =[]
		for i in subroute:
			if i!=0:
				X_plot.append(instance['{}'.format(i)]['coordinates']['x'])
				Y_plot.append(instance['{}'.format(i)]['coordinates']['y'])
			else:
				X_plot.append(instance['deport']['coordinates']['x'])
				Y_plot.append(instance['deport']['coordinates']['y'])
		assert(len(X_plot) == len(Y_plot))
		print(X_plot,Y_plot)
		plt.plot(X_plot,Y_plot)
	plt.show()


if __name__ == '__main__':
	generate_orders()
	orders_number = len(instance) - 4
	orders = list(range(1,orders_number + 1))
	##将orders按照最早时间窗从早到晚排序
	#orders.sort(key=lambda x: instance['{}'.format(x)]['Earliest'])
	set_big = get_seedset(orders)
	total_routes,total_cost =get_total(set_big)
	assert(len(total_routes) == len(total_cost))
	result = solve_problem(total_routes, total_cost)
	plot_result(result)







