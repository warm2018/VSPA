
#How to use it?
Please run the script Set_Partion in src Dir, you will see the static result plot

Please run the script dynamic in src Dir, you will see the dynacmic result plot

if you have any questions about these scripts, please contact me by GitHub (warm2018) or Email :(transliufeng.qq.com)


#Some notes about these directories:
*Dir:data*,includes the original *txt data* files which can be visualized straightly, and *json data* that provide a support to following algorithm.
*Dir:gurobipy*,An api which provided by gurobi optimization tool. 
*Dir:src*,all scripts which support this algorithm. In other words, this is the core of this project.
	***dynamic.py***  the dynamic relization of VSPA, but it is only one update and lack two constriants 
					  in the model. 	
	***Set_Partion.py*** the static 
	***utils.py***    some functions which can help to transfer the txt data to json data.
	***VNS_2.py & VSPA_1.py***  old algorithm which based on GA algorithm, but it have some small bugs in the realization the VSPA algorithm.
	***wirte_data.py*** You can use this function to change txt data files efficiently and quikly.  