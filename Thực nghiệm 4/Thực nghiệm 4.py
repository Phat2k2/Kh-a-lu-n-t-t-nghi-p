import numpy as np
from numpy import linalg as LA
import time
import math

n = 100

# Sinh ma trận A và vectơ b
def gen_matrix(n,num):
    I = np.identity(n)
    U = I
    for i in range(3):
        w = np.random.uniform(-1,1,n)
        Q = I - 2*np.outer(w,w.T)/LA.norm(w)**2
        U = U.dot(Q)
    g = np.random.uniform(-1,1,n)
    b = U.dot(g)
    if num==1:
        d = np.random.uniform(0,5,n)
        while min(d) == 0:
            d = np.random.uniform(0,5,n)        
    else:
        d = np.random.uniform(-5,5,n)
        while min(d) > 0:
            d = np.random.uniform(-5,5,n)            
    A = U.dot(np.diag(d)).dot(U.T)
    return [A,b]

# A1 là ma trận xác định dương
A1, b1 = gen_matrix(n,1)
# A2 là ma trận không xác định dương
A2, b2 = gen_matrix(n,0)
b = (b1+b2)/2
# ma tran don vi trong R^n
I = np.identity(n)
r = np.random.uniform(1,100)
# cong thuc tim hinh chieu
def hc(x):
  if LA.norm(x) > r:
    x = r*x/LA.norm(x)
  return x

def er(x,k):
    if LA.norm(x[k%2]) > 1:
        return LA.norm(x[(k+1)%2] - x[k%2])/LA.norm(x[k%2])
    else:
        return LA.norm(x[(k+1)%2] - x[k%2])

# Phần dành cho ma trận xác định dương
start_time1 = time.time()
k1 = 0
w,v=LA.eig(A1)
p1 = abs(max(w)) + 1        
X0 = np.tile(r/math.sqrt(n), n)
X = np.array([X0, np.tile(0, n)])    
while er(X,k1) >= 1e-6:
    X[(k1+1)%2] = hc(  1/p1*np.dot(X[k1%2], p1*I - A1) - b/p1)
    k1 = k1 + 1
end_time1 = time.time()


# Phần dành cho ma trận không xác định dương
start_time2 = time.time()
w,v=LA.eig(A2)
p2 = abs(max(w)) + 1        
X0 = np.tile(r/math.sqrt(n), n)
X = np.array([X0, np.tile(0, n)])
k2 = 0
while er(X,k2) >= 1e-6:
    X[(k2+1)%2] = hc(  1/p2*np.dot(X[k2%2], p2*I - A2) - b/p2)
    k2 = k2 + 1
end_time2 = time.time()
# In ra kết quả cần thu được
print("Số vòng lặp khi ma trận là xác định dương:", k1)
print("Số vòng lặp khi ma trận không là xác định dương:", k2)
print("Thời gian thực hiện khi ma trận là xác định dương:",  end_time1-start_time1 )
print("Thời gian thực hiện khi ma trận không là xác định dương:",  end_time2-start_time2 )
