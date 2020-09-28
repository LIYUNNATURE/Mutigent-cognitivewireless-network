import numpy as np

def number_of (n):
    temp_out = n[0]*1000 +n[1]*100 +n[2]*10 +n[3]
    return temp_out

def number_of_d (n):
    temp_out = n[3]*1000 +n[2]*100 +n[1]*10 +n[0]
    return temp_out
    

n = [0,0,0,0]

for a in range(10):
    for b in range(10):
        for c in range(b):
            for d in range(a):
                print (n[0])
                n[0] = a
                n[1] = b
                n[2] = c
                n[3] = d
                if number_of(n) + number_of_d(n) == 8888:
                    n = np.array(n).tolist
                    print(n)