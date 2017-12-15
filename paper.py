# Paper domain functions
import tensorflow as tf
import math

def get_m(p_r, n, c, k):
    pass

def get_m_r(p_r, n, c, k):
    if k == 0:
        return p_r / (n + c)

    return -1 * (n + c) / (2 * k) + math.sqrt(
        (
            (n + c) / (2 * k)
        ) ** 2
        +
        p_r / k
    )

def get_m_c(p_r, n, c, k):
    if k == 0:
        return p_r / (2 * (n + c))

    return -1 * (n + c) / (2 * k) + tf.sqrt(
        (
            (n + c) / (2 * k)
        ) ** 2
        +
        p_r / (2 * k)
    )

def get_p_r(n,m):
    return 2*(n*m) + 2*m

def get_p_c(n,m):
    return n*m + m
