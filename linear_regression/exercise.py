import numpy as np


# getting result from function
def get_z(point):
    x = point[0]
    y = point[1]
    return (x + 2.0 * y - 7.0) ** 2.0 + (2.0 * x + y - 5.0) ** 2.0


def grad(point):
    x = point[0]
    y = point[1]
    dx_fun = 10.0 * x + 8.0 * y - 34.0
    dy_fun = 8.0 * x + 10.0 * y - 38.0
    return np.array([dx_fun, dy_fun])


# Steepest Gradient Descent


def steepest_descent(point_0, B, iterations, epsilon=10 ** -12):
    point = point_0
    for i in range(0, iterations):
        d = - B * grad(point)
        if np.all(np.abs(d) <= epsilon):
            return point
        point += d
    return point


# Newton Method


def newton_method(point_0, B, iterations, epsilon=10 ** -12):
    # getting hessyan
    point = point_0
    dxdx_fun = 10.0
    dxdy_fun = 8.0
    dydy_fun = 10.0
    a = 1 / (dxdx_fun * dydy_fun - dxdy_fun ** 2)
    # hes - inverse hessyan
    hes = a * np.array([np.array([dydy_fun, -dxdy_fun]),
                       np.array([-dxdy_fun, dxdx_fun])])
    for i in range(0, iterations):
        d = - B * np.dot(hes, grad(point))
        if np.all(np.abs(d) <= epsilon):
            return point
        point += d
    return point
