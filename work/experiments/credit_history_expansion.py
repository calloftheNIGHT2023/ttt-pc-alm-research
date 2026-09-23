"""Offline exact historical expansion; never used to initialize a solver.

Dyadic rounded proposal coefficients avoid enormous arbitrary denominators.
Their complete error against the original certificate is checked exactly.
"""
from fractions import Fraction as F
import numpy as np
from audit_joint_credit_minimax import rational_optimum
import optimized_branch_dual as cert


def rational_history(history):
    return np.array([F(float(v)) for v in history['residual'].ravel()],dtype=object).reshape(history['residual'].shape)


def shadow_error(history,raw):
    current=np.full(raw.shape[1:],F(0),dtype=object);maximum=F(0);rows=0
    for step in range(len(raw)):
        actual=np.array([F(float(v)) for v in history['dual'][step].ravel()],dtype=object).reshape(current.shape)
        maximum=max(maximum,sum(abs(v) for v in (actual-current).ravel()));current+=raw[step]/2;rows+=1
    return dict(steps=rows,max_ideal_dual_l1=float(maximum),numerator=str(maximum.numerator),denominator=str(maximum.denominator))


def expand(x,v,reg,certificate,weights,history,banks,raw=None):
    if raw is None:raw=rational_history(history)
    t,restarts,d,n=raw.shape;coeff=np.full((t,restarts),F(0),dtype=object)
    assert len(weights)==len(banks['native']) and np.all(weights>=0)
    # Float division only proposes nonnegative dyadic coefficients. Any error,
    # including division, normalization and multiplier accumulation, is charged
    # in the exact L1 checks below; it is not asserted to be symbolic equality.
    for k in np.flatnonzero(weights>0):
        value=F(float(weights[k]/banks['native_norm'][k]));step=int(banks['native_steps'][k]);restart=int(banks['native_restart'][k]);kind=int(banks['native_kinds'][k])
        if kind in [1,2]:coeff[:step-1,restart]+=value/2
        if kind in [0,2]:coeff[step-1,restart]+=value
    ideal=np.full((d,n),F(0),dtype=object);credit=np.full((d,n),F(0),dtype=object);z=F(0);active=0
    normalized=banks['history_full'];mapping=banks['full_map'];norms=banks['full_norm'];flatcoeff=coeff.ravel();flatraw=raw.reshape(-1,d,n)
    fullfraction=np.array([F(float(v)) for v in normalized.ravel()],dtype=object).reshape(normalized.shape)
    for j in np.flatnonzero(np.array([q>0 for q in flatcoeff])):
        c=flatcoeff[j];ideal+=c*flatraw[j];index=int(mapping[j])
        if index<0:assert norms[j]==0;continue
        weight=c*F(float(norms[j]));assert weight>0;credit+=weight*fullfraction[index];z+=weight;active+=1
    assert z>0
    target=np.array([F(float(q)) for q in certificate.ravel()],dtype=object).reshape(d,n)
    original=cert.exact_optimum(x,v,reg,certificate);assert original['positive'] and 'empty_layer' not in original
    original_value=F(int(original['numerator']),int(original['denominator']))
    ideal_error=sum(abs(q) for q in (ideal-target).ravel());bank_error=sum(abs(q) for q in (credit-target).ravel())
    ideal_value=rational_optimum(x,v,reg,ideal.tolist());bank_value=rational_optimum(x,v,reg,credit.tolist())
    assert ideal_value>=original_value-ideal_error and bank_value>=original_value-bank_error
    value=bank_value/z;normalized_lower=(original_value-bank_error)/z
    return dict(positive=bool(value>0),lower_positive=bool(normalized_lower>0),ideal_positive=bool(ideal_value>0),
        original_value=float(original_value),ideal_l1_error=float(ideal_error),bank_l1_error=float(bank_error),
        full_history_value=float(value),numerator=str(value.numerator),denominator=str(value.denominator),normalized_lower=float(normalized_lower),
        normalizer=float(z),active_history_terms=active,scope='Exact constructed convex mixture of stored full-history floats, offline only')
