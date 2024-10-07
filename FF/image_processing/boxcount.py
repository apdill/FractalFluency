from .get_sizes import get_sizes
from .get_mincount import get_mincount

def boxcount(array, num_sizes=10, min_size=None, max_size=None, invert=False):
    """
    array: 2D numpy array (counts elements that aren't 0, or elements that aren't 1 if inverted)
    num_sizes: number of box sizes
    min_size: smallest box size in pixels (defaults to 1)
    max_size: largest box size in pixels (defaults to 1/5 smaller dimension of array)
    invert: 1 - array, if you want to count 0s instead of 1s
    """
    if invert:
        array = 1 - array
    min_size = 1 if min_size is None else min_size
    max_size = max(min_size + 1, min(array.shape) // 5) if max_size is None else max_size
    sizes = get_sizes(num_sizes, min_size, max_size)
    counts = []
    for size in sizes:
        counts.append(get_mincount(array, size))
    return sizes, counts