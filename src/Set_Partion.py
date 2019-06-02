import os
import io
import itertools
import random
from utils import make_dirs_for_file, exist, load_instance
import numpy as np
import matplotlib.pyplot as plt


def generate_orders(instance_name='R105'):
	json_data_dir = os.path.join('..', 'data', 'json')
	print(json_data_dir)
	json_file = os.path.join(json_data_dir, '{}.json'.format(instance_name))
	instance = load_instance(json_file=json_file)
	if instance is None:
		return
	return instance
 
def get_seedset(orders):
	
	return 


def plot_orders(instance):
	X_plot = [instance['{}'.format(i)]['coordinates']['x'] for i in range(1,len(instance)-4+1)]
	Y_plot = [instance['{}'.format(i)]['coordinates']['y'] for i in range(1,len(instance)-4+1)]
	plt.scatter(X_plot,Y_plot)
	plt.show()
		

def get_number(combination):

	return number


def get_cost(route_sequence):
	return


def get_comblations(Seed_set):
	Result_routes = []
	Result_costs = []
	for  i in range(1,len(Seed_set)+1):
		for combine in itertools.combinations(Seed_set, i):
			if get_number(combine) <= Capacity: ##满足容量约束
				best_cost = 10000
				for subroute in itertools.permutations(combine, len(combine)):
					current_cost = get_cost(subroute)
					if current_cost <= best_cost:
						best_cost = current_cost
						best_subroute = subroute
				assert(best_cost <= 10000)		
				Result_routes.append(best_subroute)
				Result_costs.append(best_cost)
	return Result_routes, Result_costs


if __name__ == '__main__':
	instance = generate_orders()
	orders_number = len(instance) - 4
	orders = list(range(1,orders_number + 1))
	##将orders按照最早时间窗从早到晚排序
	orders.sort(key=lambda x: instance['{}'.format(x)]['Earliest'])
	plot_orders(instance)


	


