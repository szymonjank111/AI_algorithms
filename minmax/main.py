import random
# TODO: For final results set seed as your student's id modulo 42
random.seed(8)


class RandomAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if random.random() > 0.5:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class GreedyAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if vector[0] > vector[-1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class NinjaAgent:
    """   ⠀⠀⠀⠀⠀⣀⣀⣠⣤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠴⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠠⠶⠶⠶⠶⢶⣶⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣶⣶⣶⣶⣶⣶⣦⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
⠀⠀⠀⠀⣸⣿⡿⠟⠛⠛⠋⠉⠉⠉⠁⠀⠀⠀⠈⠉⠉⠉⠙⠛⠛⠿⣿⣿⡄⠀
⠀⠀⠀⠀⣿⠋⠀⠀⠀⠐⢶⣶⣶⠆⠀⠀⠀⠀⠀⢶⣶⣶⠖⠂⠀⠀⠈⢻⡇⠀
⠀⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠉⢁⣠⣤⣶⣶⣶⣤⣄⣀⠀⠀⠀⠀⠀⣀⣾⠃⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣶⣶⣶⣾⣿⣿⣿⡿⠿⠿⣿⣿⣿⣿⣷⣶⣾⣿⣿⡿⠀⠀
⠀⠀⢀⣴⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀
⠀⠀⣾⡿⢃⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
⠀⢸⠏⠀⣿⡇⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀
⠀⠀⠀⢰⣿⠃⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⣉⣁⣤⡶⠁⠀⠀⠀⠀⠀
⠀⠀⣠⠟⠁⠀⠀⠀⠀⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀
                かかって来い! """
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if len(vector) % 2 == 0:
            left = sum(vector[::2])
            right = sum(vector) - left
            if left >= right:
                self.numbers.append(vector[0])
                return vector[1:]
            self.numbers.append(vector[-1])
            return vector[:-1]
        else:
            left = max(sum(vector[1::2]), sum(vector[2::2]))
            right = max(sum(vector[:-1:2]), sum(vector[:-2:2]))
            if left >= right:
                self.numbers.append(vector[-1])
                return vector[:-1]
            self.numbers.append(vector[0])
            return vector[1:]


class MinMaxAgent:
    def __init__(self, max_depth=50):
        self.numbers = []
        self.max_depth = max_depth
        self.my_move = False

    def get_value(self, list, depth, val=0):
        if depth > 0 and len(list) > 0:
            if self.my_move:
                self.my_move = False
                new_val = list[0]
                left = self.get_value(list[1:], depth-1, val+new_val)
                new_val = list[-1]
                right = self.get_value(list[:-1], depth-1, val+new_val)
                if left >= right:
                    val = left
                else:
                    val = right
                self.my_move = True
            else:
                self.my_move = True
                new_val = list[0]
                left = self.get_value(list[1:], depth-1, val-new_val)
                new_val = list[-1]
                right = self.get_value(list[:-1], depth-1, val-new_val)
                if left <= right:
                    val = left
                else:
                    val = right
                self.my_move = False
            return val
        return val

    def act(self, vector: list):
        left = self.get_value(vector[1:], self.max_depth-1, vector[0])
        self.my_move = False
        right = self.get_value(vector[:-1], self.max_depth-1, vector[-1])
        self.my_move = False
        if left >= right:
            self.numbers.append(vector[0])
            return vector[1:]
        else:
            self.numbers.append(vector[-1])
            return vector[:-1]


def run_game(vector, first_agent, second_agent):
    while len(vector) > 0:
        vector = first_agent.act(vector)
        if len(vector) > 0:
            vector = second_agent.act(vector)


def main():
    vector = [random.randint(-10, 10) for _ in range(15)]
    print(f"Vector: {vector}")
    first_agent, second_agent = NinjaAgent(), GreedyAgent()
    run_game(vector, first_agent, second_agent)

    print(f"First agent: {sum(first_agent.numbers)} Second agent: {sum(second_agent.numbers)}\n"
          f"First agent: {first_agent.numbers}\n"
          f"Second agent: {second_agent.numbers}")


if __name__ == "__main__":
    main()
