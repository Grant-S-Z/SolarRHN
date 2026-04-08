"""
This module provides helper functions and decorators for:
- Performance monitoring
- Debugging utilities
"""

from functools import wraps
from time import time
from tqdm import tqdm


def timer(func):
    """Decorator to measure and print function execution time.
    
    This decorator wraps a function and prints its execution time
    when it completes. Useful for performance profiling and optimization.
    
    Parameters
    ----------
    func : callable
        The function to be timed
    
    Returns
    -------
    callable
        Wrapped function that prints execution time
    
    Examples
    --------
    >>> @timer
    ... def slow_function():
    ...     time.sleep(1)
    ...     return "done"
    >>> result = slow_function()
    """
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time()
        value = func(*args, **kwargs)
        end_time = time()
        run_time = end_time - start_time
        # print(f"[TIMER] >>> {func.__name__!r} finished in {run_time:.2f} s")
        print(f"⏱️: {func.__name__!r} finished in {run_time:.2f} s")
        # print(f"@timer -> f {func.__name__!r} took: {run_time:.2f} secs")
        return value
    return wrapper_timer


def tqdmer(default_total=None, default_desc=None, default_unit="it"):
    """Wrap a generator function so iteration shows a tqdm progress bar.

    Usage:
        @tqdmer("Processing", "evt")
        def gen(...):
            for x in iterable: yield x
        for x in gen(..., total=N, desc="Custom", unit="item"): ...
    """
    def deco(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            total = kwargs.pop("total", default_total)
            desc = kwargs.pop("desc", default_desc)
            unit = kwargs.pop("unit", default_unit)
            iterable = func(*args, **kwargs)
            return tqdm(iterable, total=total, desc=desc, unit=unit)
        return wrapped
    return deco
