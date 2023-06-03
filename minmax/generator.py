from main import run_game, GreedyAgent, MinMaxAgent, NinjaAgent
from random import randint
from statistics import stdev
import time
import copy
import matplotlib.pyplot as plt


def generate_vector(N):
    vector = []
    for i in range(N):
        vector.append(randint(-10, 10))
    return vector


def get_results_first(vector, first_agent, second_agent):
    start = time.process_time()
    run_game(vector, first_agent, second_agent)
    stop = time.process_time()
    exec_time = stop - start
    return (sum(first_agent.numbers), sum(second_agent.numbers), exec_time)


def get_results_second(vector, first_agent, second_agent):
    start = time.process_time()
    run_game(vector, first_agent, second_agent)
    stop = time.process_time()
    exec_time = stop - start
    return (sum(second_agent.numbers), sum(first_agent.numbers), exec_time)


def final_result(depths, games, n, second_ag):
    for depth in depths:
        values1 = []
        values2 = []
        times = []
        for i in range(int(games/2)):
            vector = copy.deepcopy(generate_vector(n))
            first_agent = MinMaxAgent(depth)
            second_agent = copy.deepcopy(second_ag)
            val1, val2, time = get_results_first(vector, first_agent, second_agent)
            values1.append(val1)
            values2.append(val2)
            times.append(time)
            first_agent = MinMaxAgent(depth)
            second_agent = copy.deepcopy(second_ag)
            val1, val2, time = get_results_second(vector, second_agent, first_agent)
            values1.append(val1)
            values2.append(val2)
            times.append(time)
        avg_val1 = sum(values1) / len(values1)
        avg_val2 = sum(values2) / len(values2)
        avg_time = sum(times) / len(times)
        print(f"Głębokość: {depth}")
        print(f"Średni czas: {avg_time*1000} ms")
        print("MinMaxAgent")
        print(f"Średnia suma punktów: {avg_val1}")
        print(f"Odchylenie standardowe: {stdev(values1)}")
        print("OpponentAgent")
        print(f"Średnia suma punktów: {avg_val2}")
        print(f"Odchylenie standardowe: {stdev(values2)}\n")


def result_greedy(depths, games, n):
    print("GreedyAgent\n")
    second_agent = GreedyAgent()
    final_result(depths, games, n, second_agent)


def result_ninja(depths, games, n):
    print("NinjaAgent\n")
    second_agent = NinjaAgent()
    final_result(depths, games, n, second_agent)


def result_minmax(depths, games, n):
    print("AnotherMinMaxAgent\n")
    second_agent = MinMaxAgent(50)
    final_result(depths, games, n, second_agent)


def plotter(games, depth, second_ag, n, ag):
    values = []
    for i in range(int(games/2)):
        vector = copy.deepcopy(generate_vector(n))
        first_agent = MinMaxAgent(depth)
        second_agent = copy.deepcopy(second_ag)
        val = get_results_first(vector, first_agent, second_agent)[0]
        values.append(val)
        first_agent = MinMaxAgent(depth)
        second_agent = copy.deepcopy(second_ag)
        val = get_results_second(vector, second_agent, first_agent)[0]
        values.append(val)
    plt.hist(values, bins=30)
    plt.title(f"{ag} Agent - Depth: {depth}")
    plt.xlabel("Number of points")
    plt.ylabel("Occurances")
    plt.show()


# result_greedy([1, 5, 10, 50], 1000, 15)

# result_ninja([1, 5, 10, 50], 1000, 15)

# result_minmax([1, 5, 10, 50], 1000, 15)

# plotter(1000, 50, MinMaxAgent(50), 15, "Another MinMax")
