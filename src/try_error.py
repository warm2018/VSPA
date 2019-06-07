import random
import io

N = 50
random.seed(42)

with open("../data/text/N105.txt","w") as f1:
	with open("../data/text/R1010.txt", "r") as f2:
	    for linecount,line in enumerate(f2.readlines()):
	    	s1 = list(line)
	    	if linecount >= N + 10:
	    		break
	    	if linecount >= 10:
	    		expect_low = random.randint(1, 210)
	    		expect_upper = expect_low + 30

	    		string_low = list(str(expect_low))
	    		if len(string_low) <= 3:
	    			for i in range(3-len(string_low)):
	    				string_low.insert(0,' ')	
	    		string_low.reverse()
    			for i,value in enumerate(string_low):
    				s1[45-i] = value

	    		string_upper = list(str(expect_upper))
	    		if len(string_upper) <= 3:
	    			for i in range(3-len(string_upper)):
	    				string_upper.insert(0,' ')	
	    		string_upper.reverse()
    			for i,value in enumerate(string_upper):
    				s1[56-i] = value

	    		new_line =''.join(s1)

	    	else:
	    		new_line = line	
	    	f1.write(new_line)


