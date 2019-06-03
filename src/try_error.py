t = [[1,3,4],[1,3,4],[1,3,4]]
s = [1,2]
re = []
for i in s:
	for j in t:
		re.append(0 if i in j else 1)
		
print(re)

