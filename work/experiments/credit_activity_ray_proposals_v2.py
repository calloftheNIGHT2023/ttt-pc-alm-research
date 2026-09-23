"""Keep pending-array references alive during snapshot change detection.

The frozen v1 collector compared object IDs without retaining old arrays.
An allocator could reuse an ID within a callback and hide a real replacement.
Holding the arrays makes the existing identity comparison unambiguous. No
optimizer, direction selection, certificate or ray arithmetic is modified.
"""
import credit_activity_ray_proposals as previous
old = previous.old
base = previous.base
ray_patterns = previous.ray_patterns
verify = previous.verify


class CapturingCollector(previous.CapturingCollector):
    def activity(self, *args, **kwargs):
        held_arrays = tuple(value[1] for value in self.pending.values())
        result = super().activity(*args, **kwargs)
        # Retain strong references through the end of the parent callback.
        assert all(array is not None for array in held_arrays)
        return result


def capture(*args, **kwargs):
    before = previous.CapturingCollector
    try:
        previous.CapturingCollector = CapturingCollector
        return previous.capture(*args, **kwargs)
    finally:
        previous.CapturingCollector = before
