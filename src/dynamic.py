from Set_Partion import *
from Static import order_time,routes_result,departure
import copy
import random
import time

STATIC = 0
instance = generate_orders()
if STATIC:
	routes_result, order_time  = get_static()
record_delay = []
Time_penalty = 1

random.seed(41)
def visited(current_time):
	## 让目前的路径随着时间更新,并返回未遍历
	## + 路径 + 路径的预计出发时 + 接待下一个顾客点之后
	global remain_subroutes 
	global visited_order
	global remain_subtime
	global visited_ordertime
	global subloaded
	global substart
	remain_subroutes = [];visited_order = []; remain_subtime =[]; 
	visited_ordertime =[];subloaded =[]; subloaded =[]; substart =[]
	temp_subtime =copy.deepcopy(order_time)
	temp_subroutes = copy.deepcopy(routes_result)

	for i in range(len(temp_subtime)):
		load = 0
		for j in range(len(temp_subtime[i])):
			if temp_subroutes[i][j] != 0:
				load += instance['{}'.format(temp_subroutes[i][j])]['demand']
			else:
				load += 0
			if temp_subtime[i][j] > current_time:
				#self.subroutes[i].insert(j,-1)
				#self.subtime[i].insert(j,-1)
				#标记现在路径的状态
				subloaded.append(load)
				substart.append(temp_subtime[i][j])
				if j+1 <= len(temp_subtime[i]):
				   remain_subroutes.append(temp_subroutes[i][j:])
				   visited_order.append(temp_subroutes[i][:j])
				   remain_subtime.append(temp_subtime[i][j:])
				   visited_ordertime.append(temp_subtime[i][:j])
				assert(len(remain_subroutes) == len(remain_subtime))
				   #更新remain,断言
				break ##跳出第二个循环
	## 此函数将已访问的订单与还未访问的订单区别开来，方便下一步插入优化操作


def get_initload():
	## 得到原始路线各条子路径的载客量
	global subload
	subload = []
	for subroute in routes_result:
		load = 0
		for order in subroute:
			if order != 0:
				load += instance['{}'.format(order)]['demand']
			else:
				load += 0
		subload.append(load)



def dynamic_delay():
	#延误，随机选取还未遍历的顾客点，延后其时间窗
	current_delay = []
	for subroute,subtime in zip(remain_subroutes,remain_subtime):
		for i in subroute[1:]:
			if i != 0 and i not in current_delay :
				if random.random() < Prob_delay:
					delaytime_low =  random.randint(40, 50)
					delaytime_upper =  random.randint(50, 60)
					instance['{}'.format(i)]['Earliest'] = min(instance['{}'.format(i)]['Earliest'] + delaytime_low,MAX_TW)
					instance['{}'.format(i)]['Latest'] = min(instance['{}'.format(i)]['Latest'] + delaytime_upper,MAX_TW)
					print("******************************\n订单%d出现延误,变化前的时间窗：[%d,%d].变化后的时间窗：[%d,%d]"% \
					(i,instance['{}'.format(i)]['Earliest']-delaytime_low,\
					instance['{}'.format(i)]['Latest'] - delaytime_upper,instance['{}'.format(i)]['Earliest'],instance['{}'.format(i)]['Latest']),"变化前所在路径:",subroute)
					current_delay.append(i) ##还需记录延误的订单
	return current_delay


def merge_orders(remain_subroutes,visited_order,record_delay):
	force_order = [[],[]]
	force_memeber = []
	remain_order = []
	for i,remain in enumerate(remain_subroutes):
		visit = visited_order[i]
		if visit !=[] and len(remain)>=2:
			force_order[0].append(remain[0])
			force_order[1].append(substart[i])
			force_memeber.append(visit)
			remain_temp = copy.deepcopy(remain)
			remain_temp.remove(0)
			remain_order += remain_temp
		else:
			force_order[0].append(-1)
			force_order[1].append(-1)
			force_memeber.append([])			
			remain_temp = copy.deepcopy(remain)
			remain_temp.remove(0)
			remain_order += remain_temp
	merge_order = force_order[0] + remain_order
	set_big = dynamic_seedset(remain_order,force_order,force_memeber)
	return set_big,force_order,force_memeber

def judge_overlap(order_i,order_j):
	judge = \
		instance['{}'.format(order_i)]['Latest'] <= \
		instance['{}'.format(order_j)]['Earliest'] or \
		instance['{}'.format(order_i)]['Earliest'] >= \
		instance['{}'.format(order_j)]['Latest'] 
	if judge:
		return False
	else:
		return True

def dynamic_seedset(remain_order,force_order,force_memeber):
	remain_order_temp = copy.deepcopy(remain_order)
	for order in force_order[0]:
		if order != -1:
			remain_order_temp.remove(order)
	remain_bigset = get_seedset(remain_order_temp)
	for i,force in enumerate(force_order[0]):
		force_overlap = []
		if force != -1:
			for remain in remain_order_temp:
				if judge_overlap(force,remain):
					overlap_front = 1
					for member in force_memeber[i]:
						if not judge_overlap(remain,member):
							overlap_front = 0
							break
					if overlap_front:
						force_overlap.append(remain)
		if force_overlap != []:
			remain_bigset[force] = force_overlap
	return remain_bigset


def get_number(combination):
	number_list = [instance['{}'.format(i)]['demand'] for i in combination]
	i = 0;total_number=0
	while i < len(number_list):
		total_number += number_list[i]
		i += 1
	return total_number


def get_cost(subroute,key=None,force=False):
	subroute.append(Terminal)
	i = 1
	dist = []
	while i < len(subroute):
		dist.append(instance['distance_matrix'][subroute[i]][subroute[i-1]])
		i += 1
	### calculate the total distance for a all subroutes
	dist_cost = sum(dist) * Unit_cost
	start_cost = Start
	time_windows_cost = 0
	if force:
		current_time = key[1]
		expect_low = [instance['{}'.format(i)]['Earliest'] for i in subroute[:-1]]
		expect_upper =[instance['{}'.format(i)]['Latest'] for i in subroute[:-1]]
		subroute_time = []
		for i in range(0,len(subroute) - 1):
			distance = instance['distance_matrix'][subroute[i]][subroute[i+1]]
			time = round((distance / Vehicle_speed) * 30,1)
			current_time = current_time + time
			subroute_time.append(round(current_time,2))
			assert(subroute[i] != Terminal)
		Terminal_time = subroute_time[-1]
		time_windows_cost = sum(Time_penalty*(Terminal_time - (a_l+ b_u)/2)*(Terminal_time - (a_l+ b_u)/2) for a_l, b_u in zip(expect_low,expect_upper))
	total_cost = dist_cost + start_cost + time_windows_cost
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
	return True


def jude_detour(sequence):
	for i,order in enumerate(sequence):
		if i == 0:
			last_order = order
			continue
		if instance['distance_matrix'][order][0] > instance['distance_matrix'][last_order][0]:
			return False
		last_order = order
	return True


def get_comblations(key,Seed_set,seed_number,force=False):
	Result_routes = []
	Result_costs = []
	remain_capacity = Capacity - seed_number
	if Seed_set != []:
		for i in range(1,min(len(Seed_set),remain_capacity) + 1):
			for combine in itertools.combinations(Seed_set, i):
				if i >=2:
					time_windows = jude_time(combine)
				else:
					time_windows = True
				if get_number(combine) <= remain_capacity and time_windows: ##满足容量约束
					combine_after = list(combine)
					if not force:
						combine_after.insert(0,key)
						assert(get_number(combine_after) <= Capacity)
						best_cost = 10000
						for route_sequence in itertools.permutations(combine_after, len(combine_after)):
							subroute = list(route_sequence) ##将元组的排列改为列表的排列
							if jude_detour(subroute):
								current_cost = get_cost(subroute)
								if current_cost <= best_cost:
									best_cost = current_cost
									best_subroute = subroute
						if best_cost < 9999:	
							Result_routes.append(best_subroute)
							Result_costs.append(best_cost)
					else:
						best_cost = 10000
						for route_sequence in itertools.permutations(combine_after, len(combine_after)):
							subroute = list(route_sequence) ##将元组的排列改为列表的排列
							subroute.insert(0,key[0])
							if jude_detour(subroute):
								current_cost = get_cost(subroute,key,force)
								if current_cost <= best_cost:
									best_cost = current_cost
									best_subroute = subroute
						if best_cost < 9999:	
							Result_routes.append(best_subroute)
							Result_costs.append(best_cost)

		if len(Result_routes) == 0:
			if force:
				subroute = [key[0]]
			else:
				subroute = [key]	
			cost = get_cost(subroute)
			Result_routes.append(subroute) 
			Result_costs.append(cost)

	if Seed_set == [] and not force:
		subroute = [key]
		cost = get_cost(subroute)
		Result_routes.append(subroute) 
		Result_costs.append(cost)
	if Seed_set == [] and force:
		subroute = [key[0]]
		cost = get_cost(subroute,key,force)
		Result_routes.append(subroute) 
		Result_costs.append(cost)
	return Result_routes, Result_costs
	

def get_total(set_big,force_order,force_memeber):
	total_routes = []
	total_cost = []
	for key,values in set_big.items():
		if key not in force_order[0]:
			seed_number = instance['{}'.format(key)]['demand']
			Seed_set = values
			Result_routes, Result_costs = get_comblations(key,Seed_set,seed_number)
			total_routes = total_routes + Result_routes
			total_cost = total_cost + Result_costs
		else:
			combination =  [key] + force_memeber[force_order[0].index(key)]
			seed_number = get_number(combination)
			Seed_set = values
			key_force = [key] + [force_order[1][force_order[0].index(key)]]
			Result_routes, Result_costs = get_comblations(key_force,Seed_set,seed_number,force=True)
			total_routes = total_routes + Result_routes
			total_cost = total_cost + Result_costs			
	return total_routes, total_cost


def solve_problem(total_routes,total_cost,set_big):
	m = Model('airport')
	wideth = list(set_big.keys())
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


def solve_time(routes_result,force_order):
	terminal_time = []
	for subroute in routes_result:
		if subroute[0] not in force_order[0]:
			time = solve_lp(subroute)
			terminal_time.append(time)
		else:
			terminal_time.append(force_order[1][force_order[0].index(subroute[0])])
	return terminal_time



def solve_lp(subroute):
	m = Model("LP")
	expect_low = [instance['{}'.format(i)]['Earliest'] for i in subroute[:-1]]
	expect_upper =[instance['{}'.format(i)]['Latest'] for i in subroute[:-1]]
	x = m.addVar(vtype=GRB.CONTINUOUS,name="x")
	m.setObjective(quicksum((x - (a_l+ b_u)/2)*(x - (a_l+ b_u)/2) \
	for a_l, b_u in zip(expect_low,expect_upper)), GRB.MINIMIZE)
	m.addConstrs((x <= b_u for b_u in expect_upper), "c0")
	m.addConstrs((x >= a_l for a_l in expect_low), "c1")
	m.optimize()
	if m.status == GRB.Status.OPTIMAL:
		solution = m.getVarByName('x')
		result = solution.x
	return result


def get_ordertime(routes_result,time_result,force_order):
	time_order = []
	for subroute,subterminaltime in zip(routes_result,time_result):
		subroute_time = [subterminaltime]
		current_time = subterminaltime
		if subroute[0] not in force_order:
			for i in range(len(subroute)-1,-1,-1):
				distance = instance['distance_matrix'][subroute[i]][subroute[i-1]]
				time = round((distance / Vehicle_speed) * 30,1)
				current_time = current_time - time
				subroute_time.append(round(current_time,2))
				if i == 1:
					break
			subroute_time.reverse()
		else:
			for i in range(0,len(subroute)):
				distance = instance['distance_matrix'][subroute[i]][subroute[i+1]]
				time = round((distance / Vehicle_speed) * 30,1)
				current_time = current_time + time
				subroute_time.append(round(current_time,2))
				if i == len(subroute) - 2:
					break
		time_order.append(subroute_time)
	assert(len(time_order) == len(time_result))
	departure = []
	for depart in time_order:
		departure.append(depart[0])
	return time_order,departure



def plot_result(result,current,force_order,visited_order,record_delay,departure=None):
	plt.style.use('fivethirtyeight')
	X_plot = [instance['{}'.format(i)]['coordinates']['x'] for i in range(1,len(instance)-4+1)]
	Y_plot = [instance['{}'.format(i)]['coordinates']['y'] for i in range(1,len(instance)-4+1)]
	plt.scatter(X_plot,Y_plot,zorder=2)
	plt.scatter(instance['deport']['coordinates']['x'],instance['deport']['coordinates']['y'],s=60,color='r',zorder=2)
	plt.annotate('Airport',(instance['deport']['coordinates']['x'],instance['deport']['coordinates']['y']),size=12)
	if record_delay != []:
		for delay in record_delay:
			plt.scatter(instance['{}'.format(delay)]['coordinates']['x'],instance['{}'.format(delay)]['coordinates']['y'],color='green',zorder=3)
			
	for i in range(1,len(instance)-4+1):
		plt.annotate('{}'.format(i),(instance['{}'.format(i)]['coordinates']['x'],instance['{}'.format(i)]['coordinates']['y']),size=8)
	for i,subroute in enumerate(visited_order):
		if subroute != []:
			if force_order[0][i] != -1:
				subroute.append(force_order[0][i])
			else:
				subroute.append(Terminal)
			X_plot = [];Y_plot =[]
			for i in subroute:
				if i!=0:
					X_plot.append(instance['{}'.format(i)]['coordinates']['x'])
					Y_plot.append(instance['{}'.format(i)]['coordinates']['y'])			
				else:
					X_plot.append(instance['deport']['coordinates']['x'])
					Y_plot.append(instance['deport']['coordinates']['y'])					
			assert(len(X_plot) == len(Y_plot))	
			plt.plot(X_plot,Y_plot,zorder=1)
	plt.pause(1)

	for i,subroute in enumerate(result):
		X_plot = [];Y_plot =[]
		for i in subroute:
			if i!=0:
				X_plot.append(instance['{}'.format(i)]['coordinates']['x'])
				Y_plot.append(instance['{}'.format(i)]['coordinates']['y'])
			else:
				X_plot.append(instance['deport']['coordinates']['x'])
				Y_plot.append(instance['deport']['coordinates']['y'])
		assert(len(X_plot) == len(Y_plot))
		if departure[i] <= current:
			plt.plot(X_plot,Y_plot,zorder=1)
		else:
			if current >= 10:
				plt.plot(X_plot,Y_plot,zorder=1,color='lightslategray')
			else:
				plt.plot(X_plot,Y_plot,zorder=1,color='lightgray')		
						
	print("######",len(force_order[0]),len(visited_order))
	assert(len(force_order[0]) == len(visited_order))

	print(plt.style.available)
	plt.title('Airport Bus Routes',size=14)
	plt.xlabel('X_coordinates',size=10)
	plt.ylabel('Y_coordinates',size=10)
	plt.tight_layout()
	plt.pause(1)
	plt.savefig('scatter.png',dpi=600)


if __name__ == '__main__':
	current = 0; force_order = [[],[]]; visited_order=[]; record_delay = []
	plt.ion()
	for i,current in enumerate([-10,40]):
		if i != 0:
			visited(current)
			get_initload()
			current_delay = dynamic_delay()
			record_delay = record_delay + current_delay
			set_big,force_order,force_memeber = merge_orders(remain_subroutes, visited_order,record_delay)
			print("*******",visited_order)
			total_routes, total_cost = get_total(set_big,force_order,force_memeber)
			print(force_order,force_memeber)
			routes_result = solve_problem(total_routes, total_cost,set_big)
			terminal_time = solve_time(routes_result,force_order)
			time_order,departure = get_ordertime(routes_result, terminal_time, force_order)
			print('********Routes result********\n\n',routes_result)	
			print('********Departure time in every routes********\n\n',departure)
			print('********service time in every order of every routes********\n\n',time_order)
		plot_result(routes_result,current,force_order,visited_order,record_delay,departure)
		plt.pause(0.01)
	plt.pause(10)
	plt.ioff()

