import numpy as np
import math
import random
import matplotlib.pyplot as plt


def func(x):
    return -3*x**2/10 + 27*x


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


def next_gen(gen,x,fx,gen_num,chrom_len):

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

    ch_x= np.zeros(gen_num,dtype=int)
    ch_fx= np.zeros(gen_num,dtype=float)

    bit=np.zeros(chrom_len,dtype=int)
    
    for i in range(chrom_len):
        bit[chrom_len-i-1]=math.pow(2,i)

    for i in range(gen_num):
        for j in range(chrom_len):
            ch_x[i]+=bit[j]*child_gen[i][j]
        ch_fx=func(ch_x)

    dummy= np.vstack([gen,child_gen])

    dx= np.vstack([x.reshape((gen_num,1)),ch_x.reshape((gen_num,1))])
    dfx= np.vstack([fx.reshape((gen_num,1)),ch_fx.reshape((gen_num,1))])
    
    dummy= np.hstack([dummy,dx])
    dummy= np.hstack([dummy,dfx])
    
    dummy= dummy[dummy[:,chrom_len+1].argsort()]

    for i in range(gen_num):
        x[i]=dummy[gen_num*2-i-1][chrom_len]
        fx[i]=dummy[gen_num*2-i-1][chrom_len+1]

    dummy=np.delete(dummy,np.s_[-2:],1)

    for i in range(gen_num):
        gen[i]=dummy[gen_num*2-i-1]
        
    return gen,fx,x
    
    
def ga(x1,x2):

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
        x[i]=r
        fx[i]=func(x[i])
        for j in range(chrom_len):
            gen[i][j]= bi[j]

    avg=[]
    maxi=[]
    n=[]
    for i in range(50):
        #FOR PLOTTING
        n.append(i)
        avg.append(fx.mean())
        maxi.append(fx.max())

        avg1= fx.mean()
        gen,fx,x= next_gen(gen,x,fx,gen_num,chrom_len)
        result= np.where(fx==np.amax(fx))
        print("iteration ",i+1)
        print("population mean : ", fx.mean())
        print("population max : ", fx.max())
        print("optimal value: ",x[result[0][0]])
        print()
        avg2= fx.mean()
        if(abs(avg1-avg2)<0.01):
            break

    result= np.where(fx==np.amax(fx))
    print("population mean converges to: ", fx.mean())
    print("population max converges to: ", fx.max())
    print("optimal value: ",x[result[0][0]])
    plotting(n,avg,maxi)

    
ga(0,90)
