
from Set_Partion import *

routes_result = [[17, 48, 1, 24, 0], [2, 21, 39, 0], [9, 3, 0], [13, 4, 25, 0], [45, 5, 12, 0], [16, 6, 40, 0], [49, 7, 18, 0], [46, 8, 10, 50, 0], [36, 47, 19, 11, 29, 0], [14, 42, 15, 0], [32, 20, 35, 34, 0], [38, 44, 41, 22, 23, 0], [43, 37, 26, 0], [27, 28, 33, 0], [31, 30, 0]]
order_time = [[92.6, 110.2, 131.3, 152.1, 164.0], [10.6, 18.4, 31.2, 57.0], [81.2, 92.4, 111.5], [117.6, 136.7, 144.2, 163.0], [-5.3, 3.7, 30.2, 53.0], [33.93, 47.43, 59.33, 91.33333333333333], [28.4, 45.5, 53.0, 98.0], [46.8, 53.9, 73.6, 89.7, 115.0], [0.6, 6.0, 12.0, 17.3, 54.5, 66.6], [-8.6, -1.7, 5.2, 51.0], [91.65, 99.75, 113.25, 120.85, 136.75], [12.0, 20.1, 43.9, 47.1, 55.5, 87.8], [77.27, 90.17, 110.37, 137.66666666666666], [1.07, 6.07, 20.47, 42.666666666666664], [23.0, 32.0, 65.5]]
result_number = [7, 7, 6, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5]
result_cost = [675.829658909191, 509.2480378854053, 402.4754878398196, 502.4754878398196, 589.0537314480539, 582.7958170237475, 664.0175425099138, 654.0899038738434, 640.5885497845736, 597.3593349662714, 500.907860385255, 705.1587895018238, 602.9452499185359, 477.32075025093116, 483.8147706439405]
departure = [92.6, 10.6, 81.2, 117.6, -5.3, 33.93, 28.4, 46.8, 0.6, -8.6, 91.65, 12.0, 77.27, 1.07, 23.0]


dynamic_routes_result = [[17, 13, 4, 25, 0], [8, 48, 32, 1, 50, 0], [37, 24, 0], [9, 34, 0], [43, 26, 3, 0], [46, 10, 20, 35, 0], [6, 40, 0], [7, 18, 0], [29, 0], [41, 22, 23, 0]]
dynami_departure = [98.25, 87.3, 152.0, 87.7, 72.97, 62.1, 63.53, 57.0, 62.4, 53.6]

'''
###订单 24、8、50、37出现
延误订单24出现延误,变化前的时间窗：[143,173].变化后的时间窗：[189,232] 变化前所在路径: [17, 48, 1, 24, 0]
******************************
订单8出现延误,变化前的时间窗：[85,115].变化后的时间窗：[128,175] 变化前所在路径: [46, 8, 10, 50, 0]
******************************
订单50出现延误,变化前的时间窗：[114,144].变化后的时间窗：[156,196] 变化前所在路径: [46, 8, 10, 50, 0]
******************************
订单37出现延误,变化前的时间窗：[124,154].变化后的时间窗：[173,204] 变化前所在路径: [43, 37, 26, 0]
<<<<<<< HEAD
'''
=======
'''

print(100/105)
>>>>>>> 02f4f41eb66ec602af20850f648e0eb6cc4aad66
