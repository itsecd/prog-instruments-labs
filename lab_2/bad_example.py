import math

def process_data(data_list, flag=False, x=10):
    res = []
    for i in range(len(data_list)):
        if data_list[i] is not None and data_list[i] != "":
            if flag == True:
                calc = (data_list[i] * 2 + x) / 1.5
                if calc > 10:
                    res.append(calc)
                else:
                    res.append(0)
            else:
                res.append(data_list[i] * 2)
    
    final_res = []
    for i in range(len(res)):
        if res[i] not in final_res:
            final_res.append(res[i])
            
    for item in final_res:
        print("Result: " + str(item))
        
    return final_res

my_data = [1, 2, 3, "4", 5, None, 5, 2]
process_data(my_data, True)