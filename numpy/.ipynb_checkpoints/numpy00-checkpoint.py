import numpy as np 

A = np.arange(9).reshape(3, 3)
print('A :', A)
B = np.arange(11, 11+9).reshape(3,3)
print('B: ', B)

x = np.arange(3)
print(x)
y = np.arange(3).reshape(3,1)
print(y)
z = np.arange(3).reshape(1,3)
print(z)

C1 = np.dot(A, B)
C2 = np.matmul(A, B)
C3 = A@B
print('------------')
print('C1: ',C1)
print('C2: ',C2)
print('C3: ',C3)