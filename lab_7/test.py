from functools import reduce
import time
import timeit
import cProfile

def manual_sum(n):
    res = 0
    for i in range(1, n+1):
        res += i
    return res


def standart_sum(n):
    return sum(list(range(1,n+1)))


def func_sum(n):
    return reduce(lambda x,y: x+y, list(range(1, n+1)))


print("""manual""")
start = time.time()
manual_sum(10000000)
end = time.time()
print(end-start)
timeit_time = timeit.timeit("manual_sum(10000000)", "from __main__ import manual_sum", number=1)
print((timeit_time))

print("""stand""")
start = time.time()
standart_sum(10000000)
end = time.time()
print(end-start)
timeit_time = timeit.timeit("standart_sum(10000000)", "from __main__ import standart_sum", number=1)
print((timeit_time))

print("""func""")
start = time.time()
func_sum(10000000)
end = time.time()
print(end-start)
timeit_time = timeit.timeit("func_sum(10000000)", "from __main__ import func_sum", number=1)
print((timeit_time))


cProfile.run("manual_sum(10000000)")
cProfile.run("standart_sum(10000000)")
cProfile.run("func_sum(10000000)")