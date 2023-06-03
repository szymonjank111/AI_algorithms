from exercise import newton_method, steepest_descent, get_z
import numpy as np
import time


def time_newton(point_0, Beta, i):
    start = time.process_time()
    newton_method(point_0, Beta, i)
    stop = time.process_time()
    insert_time = 1000 * (stop - start)
    return [point_0, get_z(point_0), i, insert_time]


def time_descent(point_0, Beta, i):
    start = time.process_time()
    steepest_descent(point_0, Beta, i)
    stop = time.process_time()
    insert_time = 1000 * (stop - start)
    return [point_0, get_z(point_0), i, insert_time]


def print_descent(point_0, Beta, list_of_iter):
    print("Descent method:\n")
    for i in list_of_iter:
        point = np.copy(point_0)
        descent = time_descent(point, Beta, i)
        print(f"X = {descent[0][0]}, Y = {descent[0][1]}, Z = {descent[1]}")
        print(f"Iterations: {descent[2]}")
        print(f"Time: {descent[3]}\n")
    print(point_0)


def print_newton(point_0, Beta, list_of_iter):
    print("Newton method:\n")
    for i in list_of_iter:
        point = np.copy(point_0)
        newton = time_newton(point, Beta, i)
        print(f"X = {newton[0][0]}, Y = {newton[0][1]}, Z = {newton[1]}")
        print(f"Iterations: {newton[2]}")
        print(f"Time: {newton[3]}\n")
    print(point_0)


def print_results(point_0, Beta, i):
    newton = time_newton(point_0, Beta, i)
    descent = time_descent(point_0, Beta, i)
    print("Descent method:")
    print(f"X = {descent[0][0]}, Y = {descent[0][1]}, Z = {descent[1]}")
    print(f"Iterations: {descent[2]}")
    print(f"Time: {descent[3]}")
    print("Newton method:")
    print(f"X = {newton[0][0]}, Y = {newton[0][1]}, Z = {newton[1]}")
    print(f"Iterations: {newton[2]}")
    print(f"Time: {newton[3]}")


iterations = [1]
# print_descent(np.array([0.0, -5.0]), 0.09, iterations)
print_newton(np.array([0.0, -5.0]), 1.0, iterations)
# print_results(np.array([1.0, 2.0]), 0.2, 10)
