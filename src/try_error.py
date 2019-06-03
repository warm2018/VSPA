import itertools

Seed_set = [1,2]

for i in range(1,1):
	print("lll",i)
	for combine in itertools.combinations(Seed_set, i):
		print("lll",combine)
		list1 = list(combine)
		list1.insert(0,7)
		print(list1)
