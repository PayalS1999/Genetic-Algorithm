import numpy as np
import math
import random
import matplotlib.pyplot as plt


def dec_item(x,size,cons,w,weight):
    while(w>cons):
        i=random.randint(0,size-1)
        while x[i]!=1:
            i=random.randint(0,size-1)
        w-=weight[i]
        x[i]=0
    return x,w
        


def func(x,size,price,weight,cons):
    w=0
    p=0
    for i in range(size):
        if x[i]==1:
            w+=weight[i]
    if w>cons:
        x,w=dec_item(x,size,cons,w,weight)
    for i in range(size):
        if x[i]==1:
            p+=price[i]
    return x,p


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

    sum=0
    for i in range(gen_num):
        sum+=expected_val[i]
        if sum>=r:
            break
    return i


def next_gen(gen,fx,gen_num,chrom_len,price,weight,cons):

    crossover_prob=0.7
    mutation_prob=0.3

    child_gen= np.zeros((gen_num,chrom_len),dtype=int)
    row=0

    for iteration in range(int(gen_num/2)):
        #SELECTION OF PARENTS
        ind1= roulette_wheel(fx,gen_num)
        ind2= roulette_wheel(fx,gen_num)
        while(ind1==ind2):
            ind2= roulette_wheel(fx,gen_num)

        child_gen[row]=gen[ind1]
        child_gen[row+1]=gen[ind2]

        #CROSSOVER
        prob= random.random()
        if prob<= crossover_prob:
            cr= random.randint(1,chrom_len-1)
            for i in range(cr):
                child_gen[row][i],child_gen[row+1][i]=child_gen[row+1][i],child_gen[row][i]

        #MUTATION
        for i in range(row,row+2):
            prob= random.random()
            if prob<= mutation_prob:
                mu= random.randint(0,chrom_len-1)
                if child_gen[i][mu]==0:
                    child_gen[i][mu]=1
                else:
                    child_gen[i][mu]=0

        row+=2
        
    ch_fx= np.zeros(gen_num,dtype=float)

    for i in range(gen_num):
        child_gen[i],ch_fx[i]=func(child_gen[i],chrom_len,price,weight,cons)

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
    
    
def ga(chrom_len,price,weight,cons):

    gen_num= 10

    gen= np.zeros((gen_num,chrom_len),dtype=int)
    fx= np.zeros(gen_num)

    #GENERATE INITIAL POPULATION
    for i in range(gen_num):
        r= random.randint(0,2**chrom_len-1)
        bi= np.binary_repr(r, width=chrom_len)
        for j in range(chrom_len):
            gen[i][j]=bi[j]
        gen[i],fx[i]=func(gen[i],chrom_len,price,weight,cons)

    print(gen,fx)
    avg=[]
    maxi=[]
    n=[]
    for i in range(300):
        #FOR PLOTTING
        n.append(i)
        avg.append(fx.mean())
        maxi.append(fx.max())

        avg1= fx.mean()
        gen,fx= next_gen(gen,fx,gen_num,chrom_len,price,weight,cons)
        result= np.where(fx==np.amax(fx))
        print("iteration ",i+1)
        print("population mean : ", fx.mean())
        print("population max : ", fx.max())
        print("optimal item list: ",gen[result[0][0]])
        print()
        avg2= fx.mean()
        if(abs(avg1-avg2)<0.01):
            break

    result= np.where(fx==np.amax(fx))
    print("population mean converges to: ", fx.mean())
    print("Max profit: ", fx.max())
    print("optimal item list: ",gen[result[0][0]])
    plotting(n,avg,maxi)

    
item=input('Enter no of items: ')
item=int(item)
price=np.zeros(item)
weight=np.zeros(item)
print('Enter the price of each item: ')
for i in range(item):
    inp=input()
    price[i]=float(inp)
print('Enter the weight of each item: ')
for i in range(item):
    inp=input()
    weight[i]=float(inp)
cons=input('Enter weight constraint: ')
cons=float(cons)
print(price,weight,cons)
ga(item,price,weight,cons)
