import numpy as np
from matplotlib import pyplot as plt

#
#  Does a simulation to calculate the probability that
#  the next record is r considering the last one was s.
#
#
def sim_poisson(r,s, n=100):
    total = 0.
    for i in range(n):
        x = s
        while x <= s:
            x = np.random.poisson(lam=1)
        if r == x:
            total = total + 1
    return total/n



def sim_uniform(r,s,N =25, n=100):
    total = 0.
    for i in range(n):
        x = s
        while x <= s:
            x = np.random.randint(N)
        if r == x:
            total = total + 1
    return total/n


def sim_geometric(r,s,p=0.5,n=100):
    total = 0.
    for i in range(n):
        x = s
        while x <= s:
            x = np.random.geometric(p)
        if r == x:
            total = total + 1
    return total / n

#
#
#  Poisson distribution.
#
def compute_poisson(r,s):

    if r < 20:
        a_r = np.exp(-1)/(np.math.factorial(r))
    else:
        a_r = 0

    sum = 0.
    for i in range(1,s+1):
        sum = sum + np.exp(-1)/(np.math.factorial(i))

    return a_r/(1- sum)

#
# uniform on the range 0 to N-1
#
def compute_uniform(r,s, N=25):

    a_r = 1./N

    sum = 0.
    for i in range(1, s + 1):
        sum = sum + 1./N

    return (a_r / (1 - sum))


def compute_geometric(r,s, p =0.5):
    a_r = p*np.power(1-p, r-1)

    sum = 0.
    for i in range(1, s + 1):
        sum = sum + p*np.power(1-p, i-1)

    return (a_r / (1 - sum))


print(compute_geometric(5,3,p=0.25))
print(sim_geometric(5,3,p=0.25,n=10000))
