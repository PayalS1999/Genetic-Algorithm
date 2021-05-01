import numpy as np
import math
import random
import matplotlib.pyplot as plt


def func(x):
    return x**2 + 48*x -385


def plotting(n,x,y):
    fig= plt.figure()
    plt.plot(n,x,'-',color='red',alpha=0.6,label='Population average')
    plt.plot(n,y,'--',color='green',label='Population maximum')
    plt.legend()
    plt.show()

    
def roulette_wheel(fx,gen_num):
    #print('Inside roulette wheel')
    fx=fx-min(fx)
    fx=max(fx)-fx
    m=sum(fx)
    pick=random.uniform(0,m)
    s=0
    for idx,ch in enumerate(fx):
        s+=ch
        #print('m=',m,' s=',s,' pick=',pick)
        if s>=pick:
            break
    return idx


def next_gen(gen,x,fx,gen_num,chrom_len,lower):

    crossover_prob=0.7
    mutation_prob=0.3

    child_gen= np.zeros((gen_num,chrom_len),dtype=int)
    row=0

    for iteration in range(int(gen_num/2)):
        #SELECTION OF PARENTS
        ind1= roulette_wheel(fx,gen_num)
        #print('after 1')
        ind2= roulette_wheel(fx,gen_num)
        #print('after 2')
        #while(ind1==ind2):
        #    ind2= roulette_wheel(fx,gen_num)

        child_gen[row]=gen[ind1]
        child_gen[row+1]=gen[ind2]

        #2 POINT CROSSOVER
        prob= random.random()
        if prob<= crossover_prob:
            cr1= random.randint(1,chrom_len-1)
            cr2=random.randint(1,chrom_len-1)
            while(cr2==cr1):
                cr2=random.randint(1,chrom_len-1)
            if cr2<cr1:
                cr2,cr1=cr1,cr2
            for i in range(cr1,cr2):
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

    ch_x= np.zeros(gen_num,dtype=int)
    ch_fx= np.zeros(gen_num,dtype=float)

    bit=np.zeros(chrom_len,dtype=int)
    
    for i in range(chrom_len):
        bit[chrom_len-i-1]=math.pow(2,i)

    for i in range(gen_num):
        for j in range(chrom_len):
            ch_x[i]+=bit[j]*child_gen[i][j]
        ch_x[i]+=lower
        ch_fx=func(ch_x)

    dummy= np.vstack([gen,child_gen])

    dx= np.vstack([x.reshape((gen_num,1)),ch_x.reshape((gen_num,1))])
    dfx= np.vstack([fx.reshape((gen_num,1)),ch_fx.reshape((gen_num,1))])
    
    dummy= np.hstack([dummy,dx])
    dummy= np.hstack([dummy,dfx])
    
    dummy= dummy[dummy[:,chrom_len+1].argsort()]

    for i in range(gen_num):
        x[i]=dummy[i][chrom_len]
        fx[i]=dummy[i][chrom_len+1]

    dummy=np.delete(dummy,np.s_[-2:],1)

    for i in range(gen_num):
        gen[i]=dummy[i]
        
    return gen,fx,x
    
    
def ga(lower,upper):

    x2=upper-lower
    x1=0
    gen_num= 10
    for chrom_len in range(11):
        if math.pow(2,chrom_len)>x2:
            break

    gen= np.zeros((gen_num,chrom_len),dtype=int)
    x= np.zeros(gen_num,dtype=int)
    fx= np.zeros(gen_num)

    #GENERATE INITIAL POPULATION
    for i in range(gen_num):
        r= random.randint(x1,x2)
        bi= np.binary_repr(r, width=chrom_len)
        x[i]=r+lower
        fx[i]=func(x[i])
        for j in range(chrom_len):
            gen[i][j]= bi[j]

    print(x,fx)
    avg=[]
    maxi=[]
    n=[]
    for i in range(50):
        #FOR PLOTTING
        n.append(i)
        avg.append(fx.mean())
        maxi.append(fx[0])

        avg1= fx.mean()
        gen,fx,x= next_gen(gen,x,fx,gen_num,chrom_len,lower)
        
        print("iteration ",i+1)
        print("population mean : ", fx.mean())
        print("population max : ", fx[0])
        print("optimal value: ",x[0])
        print()
        avg2= fx.mean()
        if(abs(avg1-avg2)<0.01):
            break

    print("population mean converges to: ", fx.mean())
    print("population max converges to: ", fx[0])
    print("optimal value: ",x[0])
    plotting(n,avg,maxi)

    
ga(-55,7)
