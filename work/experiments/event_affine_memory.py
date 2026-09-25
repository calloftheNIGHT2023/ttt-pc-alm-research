"""Sequential dependency adapter; unchanged frozen ALM equations, faster bias solve."""
import affine_eliminated_memory as original
import event_bias_block as bias_solver
base=original.base
first=original.first


def local(*args,**kwargs):
    previous=original.bias_solver
    try:
        original.bias_solver=bias_solver
        return original.local(*args,**kwargs)
    finally:original.bias_solver=previous


def fit(*args,**kwargs):
    previous=original.bias_solver
    try:
        original.bias_solver=bias_solver
        return original.fit(*args,**kwargs)
    finally:original.bias_solver=previous
