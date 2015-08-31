import numpy as np
from numpy.random import randn
from numpy.linalg import norm

def khatrirao(A, B):
    return np.squeeze(np.array([np.kron(colpair[0], colpair[1]) for colpair in zip(A.T, B.T)])).T

def randomortho(m,n):
    return np.linalg.qr(randn(m,n))[0]

def randtensor3(sz, dtrue):
    if type(dtrue) is tuple:
        # Tucker
        U=[randomortho(s[0], s[1]) for s in zip(sz,dtrue)]
        C=randn(dtrue[0]*dtrue[1], dtrue[2])
        X=np.dot(np.dot(np.kron(U[0], U[1]),C),U[2].T)
    else:
        # CP
        U=[randn(s, dtrue) / (dtrue**(1.0/2/len(sz))) for s in sz]
        C=np.diag(randn(dtrue))
        X=np.dot(np.dot(tensor.khatrirao(U[0],U[1]),C),U[2].T)
    return X.reshape(sz)

def unfold(X, kk):
    sz=X.shape; ndim=len(sz)
    pind=range(kk,ndim)+range(0,kk)
    psz=[sz[i] for i in pind]
    return X.transpose(pind).reshape((psz[0], np.prod(psz[1:])))

def outer(tup):
    sz=tuple([len(x) for x in tup])
    X = tup[0]
    for uu in tup[1:]:
        X = np.kron(X, uu)
    return X.reshape(sz)

def poweriteration(X, ncomp, niter=50):
    Uhat=[]; Vhat=[]; What=[]
    vals=[]
    for rr in range(ncomp):
        uu=np.random.randn(50); uu/=norm(uu)
        vv=np.random.randn(50); vv/=norm(vv)
        ww=np.random.randn(50); ww/=norm(ww)

        for kk in range(niter):
            unew = ((X * ww).sum(2) * vv).sum(1)
            vnew = ((X.transpose((1,2,0)) * uu).sum(2) * ww).sum(1)
            wnew = ((X.transpose((2,0,1)) * vv).sum(2) * uu).sum(1)
            val=np.sqrt((norm(unew)**2 + norm(vnew)**2 + norm(wnew)**2)/3)
            unew/=norm(unew); vnew/=norm(vnew); wnew/=norm(wnew)
            res  = 1-(np.dot(unew.T,uu)+np.dot(vnew.T,vv)+np.dot(wnew.T,ww))/3
            print 'kk=%d, val=%g res=%g' % (kk, val, res)
            uu=unew; vv=vnew; ww=wnew
            if res<1e-6:
                break
        Uhat.append(uu); Vhat.append(vv); What.append(ww)
        vals.append(val)
        X=X-val* outer((uu,vv,ww))
        print ' norm(X)=%g' % (norm(X.ravel()))        
    return (Uhat, Vhat, What, vals)
