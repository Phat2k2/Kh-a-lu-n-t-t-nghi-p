x=[1 , 2 , 3 , 4 , 5]
y=[1 , 4 , 9 , 16 , 25]
n=len(x)
a=6.5
p=0
for i in range(n):
    t=1
    for j in range(n):
        if i!=j:
            t=t*(a-x[j])/(x[i]-x[j])
    p+=y[i]*t
print("Giá trị P(x) là P(x) = "+str(p))