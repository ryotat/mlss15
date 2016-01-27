import numpy as np
from scipy import sparse
import pickle

def loaddata(file):
    I=[]; J=[]; Y=[]
    for line in open(file):
        tmp=line.split('\t')
        I.append(int(tmp[0])-1)
        J.append(int(tmp[1])-1)
        Y.append(float(tmp[2]))
    Y=np.array(Y)
    return I,J,Y


# Gradient descent with L2 regularization
def gradientdescent(sz, rr, Itr, Jtr, Ytr, lmd, step, maxiter):
    # Initialize decomposition
    U=np.random.randn(sz[0],rr)
    V=np.random.randn(sz[1],rr)
    Bu=np.random.randn(sz[0],1).ravel()
    Bv=np.random.randn(sz[1],1).ravel()

    M=len(Ytr)
    yvec = Ytr.ravel()/Ytr.std()

    # Prepare index matrices
    indU = sparse.coo_matrix((np.ones(M),(Itr,range(M))),shape=(sz[0],M))
    indV = sparse.coo_matrix((np.ones(M),(Jtr,range(M))),shape=(sz[1],M))
    for kk in range(maxiter):
        pred = (U[Itr,:]*V[Jtr,:]).sum(axis=1)+Bu[Itr]+Bv[Jtr]
        gg = (pred - yvec).reshape((M,1))
        Unew  = (1-step*lmd) * U - step * (indU*(gg * V[Jtr,:]))
        Vnew  = (1-step*lmd) * V - step * (indV*(gg * U[Itr,:]))
        Bunew = Bu - step * (indU * gg).ravel()
        Bvnew = Bv - step * (indV * gg).ravel()
        U, V=Unew, Vnew; Bu, Bv=Bunew, Bvnew
        if kk % 100 == 0:
            print 'kk=%d error=%g' % (kk, (gg**2).sum()/M)
    return (U,V,Bu,Bv)

if __name__== '__main__':
    sz=(943, 1682)
    rr=10
    maxiter=1000
    step=2e-3
    lmds=np.exp(np.linspace(np.log(1), np.log(100), 20))

    Itr,Jtr,Ytr=loaddata('datasets/ml-100k/ua.base')
    Ite,Jte,Yte=loaddata('datasets/ml-100k/ua.test')
    file_save = 'result_100k_r=%d.pck' % rr
    err=[]
    solution=[]

    # Test
    with open(file_save,'w') as f:
        pickle.dump({'rr':rr,'maxiter':maxiter,'step':step,'lmds':lmds,'err':err,'solution':solution}, f)

    for lmd in lmds:
        U,V,Bu,Bv = gradientdescent(sz, rr, Itr, Jtr, Ytr, lmd, step, maxiter)

        pred = (U[Ite,:]*V[Jte,:]).sum(axis=1)+Bu[Ite]+Bv[Jte]
        err.append(np.mean((Ytr.std()*pred - Yte)**2))
        solution.append((U,V,Bu,Bv))
        print 'lmd=%g err=%g' % (lmd, err[-1])
    with open(file_save,'w') as f:
        pickle.dump({'rr':rr,'maxiter':maxiter,'step':step,'lmds':lmds,'err':err,'solution':solution}, f)
