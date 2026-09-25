"""Contiguous query-array boundary for frozen Torch controls; no model changes."""
import numpy as np
import band_conditioned_meta as original

load = original.load
arrays = original.arrays
torch = original.torch
models = original.models
ORIGINAL_FIT = original.fit


def fit(model, x, v, state):
    predict, new, metadata = ORIGINAL_FIT(model, np.ascontiguousarray(x), np.ascontiguousarray(v), state)
    def contiguous_predict(q):
        return predict(np.ascontiguousarray(q))
    return contiguous_predict, new, dict(metadata, numpy_contiguous_input_boundary=True)
