# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import collections
tup1 = (1,3,5,7,9)
dq1 = collections.deque(tup1)
dq2 = dq1.copy()

dq2.pop()

print(dq1)
print(dq2)