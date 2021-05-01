import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random


def func(x,chrom_len,weight_mat):
    s=0
    for i in range(chrom_len):
        s+= weight_mat[int(x[i])][int(x[(i+1)%chrom_len])]
        
    return 1/s


def plotting(n,x,y):
    fig= plt.figure()
    plt.plot(n,x,'-',color='red',alpha=0.6,label='Population average')
    plt.plot(n,y,'--',color='green',label='Population maximum')
    plt.legend()
    plt.show()

    
def roulette_wheel(fx,gen_num):
    expected_val=np.zeros(gen_num)
    for i in range(gen_num):
        expected_val[i]= fx[i]/fx.mean()

    T= math.ceil(expected_val.sum())

    r= random.randint(0,T)

    s=0
    for i in range(gen_num):
        s+=expected_val[i]
        if s>=r:
            break
    return i


def next_gen(gen,fx,gen_num,chrom_len,weight_mat):

    crossover_prob=0.8
    mutation_prob=1

    child_gen= np.zeros((gen_num,chrom_len),dtype=int)
    row=0

    for iteration in range(int(gen_num/2)):
        #SELECTION OF PARENTS
        ind1= roulette_wheel(fx,gen_num)
        ind2= roulette_wheel(fx,gen_num)
        while(ind1==ind2):
            ind2= roulette_wheel(fx,gen_num)

        for i in range(chrom_len):
            child_gen[row]=gen[ind1]
            child_gen[row+1]=gen[ind2]

        #PMX- CROSSOVER
        prob= random.random()
        if prob<= crossover_prob:
            cr= random.randint(1,chrom_len-1)
            indx=[ind1,ind2]
            for j in range(row,row+2):
                for i in range(cr):
                    val=gen[indx[j-row]][i]
                    for p in range(i,chrom_len):
                        if child_gen[j][p]==val:
                            child_gen[j][p],child_gen[j][i]=child_gen[j][i],child_gen[j][p]
                            break

        #MUTATION
        for i in range(row,row+2):
            prob= random.random()
            if prob<= mutation_prob:
                k= random.randint(0,chrom_len-1)
                p= random.randint(0,chrom_len-1)
                child_gen[i][p],child_gen[i][k]=child_gen[i][k],child_gen[i][p]

        row+=2

    ch_fx= np.zeros(gen_num,dtype=float)

    for i in range(gen_num):
        ch_fx[i]=func(child_gen[i],chrom_len,weight_mat)

    dummy= np.vstack([gen,child_gen])
    dfx= np.vstack([fx.reshape((gen_num,1)),ch_fx.reshape((gen_num,1))])
    dummy= np.hstack([dummy,dfx])
    
    dummy= dummy[dummy[:,chrom_len].argsort()]

    for i in range(gen_num):
        fx[i]=dummy[gen_num*2-i-1][chrom_len]

    dummy=np.delete(dummy,np.s_[-1:],1)

    for i in range(gen_num):
        gen[i]=dummy[gen_num*2-i-1]
        
    return gen,fx
    
    
def ga(weight_mat,chrom_len):

    gen_num= 60

    gen= np.zeros((gen_num,chrom_len),dtype=int)
    fx= np.zeros(gen_num)

    #GENERATE INITIAL POPULATION
    for i in range(gen_num):
        r= np.random.permutation(chrom_len)
        gen[i]= r
        fx[i]=func(r,chrom_len,weight_mat)

    avg=[]
    maxi=[]
    n=[]
    for i in range(500):
        #FOR PLOTTING
        n.append(i)
        avg.append(fx.mean())
        maxi.append(fx.max())

        gen,fx= next_gen(gen,fx,gen_num,chrom_len,weight_mat)
        result= np.where(fx==np.amax(fx))
        print("iteration ",i+1)
        print("population mean : ", fx.mean())
        print("population max : ", fx.max())
        print("optimal path: ",gen[result[0][0]])
        print("min cost: ",1/fx[result[0][0]])
        print()

    result= np.where(fx==np.amax(fx))
    print("population mean converges to: ", fx.mean())
    print("population max converges to: ", fx.max())
    print("optimal path: ",gen[result[0][0]])
    print("min cost: ",1/fx[result[0][0]])
    plotting(n,avg,maxi)


df= pd.read_csv("tsp_data.csv")
city= 15
inf= float('inf')
city= int(city)
weight_mat=np.zeros(shape=(city,city),dtype='float')
index=0
for i in range(city):
    for j in range(i+1,city):
        inp=df[df.columns[0]][index]
        if inp=='0':
            weight_mat[i][j]= inf
        else:
            weight_mat[i][j]= float(inp)
        weight_mat[j][i]=weight_mat[i][j]
        index+=1

print('path matrix: ')
print(weight_mat)
ga(weight_mat,city)
