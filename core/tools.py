"""
Utility tools for Solar RHN calculations.

This module provides helper functions and decorators for:
- Performance monitoring (timing decorators)
- Debugging utilities
- General-purpose tools
"""

from functools import wraps
from time import time


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
    func: 'slow_function' took: 1.0012 secs
    """
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time()
        value = func(*args, **kwargs)
        end_time = time()
        run_time = end_time - start_time
        print(f"func: {func.__name__!r} took: {run_time:.4f} secs")
        return value
    return wrapper_timer


__all__ = ['timer']
