"""287 paired task bootstrap; common integer resamples, bounded working memory."""
import hashlib
import numpy as np


def bootstrap_means(differences,repetitions=100000,seed=287193,block=64):
    d=np.asarray(differences,dtype=np.float64);assert d.ndim==2 and np.isfinite(d).all()
    n,columns=d.shape;assert n>1 and repetitions>0 and block>0
    rng=np.random.default_rng(seed);ans=np.empty((repetitions,columns));digest=hashlib.sha256()
    for first in range(0,repetitions,block):
        k=min(block,repetitions-first);indices=rng.integers(n,size=(k,n),dtype=np.int64);digest.update(indices.tobytes())
        weights=np.array([np.bincount(row,minlength=n) for row in indices],dtype=np.float64)/n
        ans[first:first+k]=weights@d
    return ans,digest.hexdigest()


def selftest():
    d=np.array([[1.,3.,0.],[-2.,6.,0.],[4.,-5.,0.],[.5,2.,0.]])
    rr=np.random.default_rng(287193);indices=rr.integers(4,size=(103,4),dtype=np.int64)
    explicit=d[indices].mean(1);checks=0
    for block in [1,7,64,103]:
        actual,digest=bootstrap_means(d,103,287193,block)
        np.testing.assert_allclose(actual,explicit,rtol=0,atol=1e-15)
        assert digest==hashlib.sha256(indices.tobytes()).hexdigest();checks+=1
        assert not actual[:,2].any()
    # Single task, not each query point, is the resampling unit.
    return dict(passed=True,explicit_index_checks=checks,zero_difference_retained=True,block_independent_random_stream=True)
