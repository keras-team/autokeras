import multiprocessing
from time import sleep


def foo(a, b):
    return a + b


d = {}
p = multiprocessing.Pool(2)
data = p.map_async(foo, [[3, 4], [5, 6]])
print(data.get())
p.close()
p.join()
