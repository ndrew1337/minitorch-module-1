"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

from numpy import iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float: 
    """Multiplies two numbers"""
    return x * y 

def id(x: float) -> float:
    """Returns the input unchanged"""
    return x

def add(x: float, y: float) -> float:
    """Adds two numbers"""
    return x + y

def neg(x: float) -> float:
    """Negates a number"""
    return -1.0 * x

def lt(x: float, y: float) -> bool: 
    """Checks if one number is less than another"""
    return x < y

def eq(x: float, y: float) -> bool: 
    """Checks if two numbers are equal"""
    return x == y

def max(x: float, y: float) -> float: 
    """Returns the larger of two numbers"""
    return y if lt(x, y) else x

def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value""" 
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))

def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return x if x >= 0.0 else 0.0

def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x)

def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)

def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return y / x

def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1.0 / x

def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -y / (x ** 2)

def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(a: Iterable[float], fn: Callable[[float], float]) -> list[float]:
    """Higher-order function that applies a given function to each element of an iterable"""
    return [fn(x) for x in a]

def zipWith(a: Iterable[float], b: Iterable[float], fn: Callable[[float, float], float]) -> list[float]:
    """Higher-order function that combines elements from two iterables using a given function"""
    return [fn(x, y) for x, y in zip(a, b)]

def reduce(a: Iterable[float], start: float, fn: Callable[[float, float], float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    cur = start
    for x in a:
        cur = fn(x, cur)
    return cur

def negList(a: list[float]) -> list[float]:
    """Negate all elements in a list using map"""
    return map(a, neg)

def addLists(a: list[float], b: list[float]) -> list[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(a, b, add)

def sum(a: list[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(a, 0, add)

def prod(a: list[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(a, 1, mul)

