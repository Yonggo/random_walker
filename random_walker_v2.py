import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib.animation import FuncAnimation

# Global variables
size_row = 10
size_col = 10
max_visit = 100
scope_for_reduce_value = 10
max_scope_for_torus_value = 10
random_walker_init_pos = [[5,5],[0,0],[1,3]]
prob_for_the_deepest_value = [80, 70, 50]
random_walker_amount = len(random_walker_init_pos)
current_x_pos = [None] * random_walker_amount
current_y_pos = [None] * random_walker_amount


def pick_value_based_on_probability(array, probabilities):
    # chooses a value based on probability
    chosen_value = np.random.choice(array, 1, probabilities)
    return chosen_value


def pick_value_based_on_weights(array, weights):
    # chooses a value based on weights
    chosen_value = random.choices(array, weights=weights, k=1)
    return chosen_value


def calcu_probs(highest_prob):
    rest_prob = None
    if highest_prob == 100:
        highest_prob = 1
        rest_prob = 0
    elif highest_prob > 1:
        list_of_digits = [int(i) for i in str(int(highest_prob))]
        digit_length = len(list_of_digits)
        full_value = math.pow(10, digit_length)
        highest_prob /= full_value
        rest_prob = (1 - highest_prob) / 3.0
    else:
        rest_prob = (1 - highest_prob) / 3.0
    probs = [highest_prob, rest_prob, rest_prob, rest_prob]
    return probs


# if a node is not the the lowest node in its neighborhood, altitude will be calculated as the average value of all neighbors
def calcu_value_of_neighbor(x_pos, y_pos, torus):
    values_of_neighbors = []
    for ix in [-1, 1]:
        neighbor_x_pos = x_pos + ix
        if neighbor_x_pos < 0:
            neighbor_x_pos = size_row - 1
        if neighbor_x_pos >= size_row:
            neighbor_x_pos -= size_row
        values_of_neighbors.append(torus[neighbor_x_pos][y_pos])
    for iy in [-1, 1]:
        neighbor_y_pos = y_pos + iy
        if neighbor_y_pos < 0:
            neighbor_y_pos = size_col - 1
        if neighbor_y_pos >= size_col:
            neighbor_y_pos -= size_col
        values_of_neighbors.append(torus[x_pos][neighbor_y_pos])
    min_of_neighbor = min(values_of_neighbors)
    if torus[x_pos][y_pos] <= min_of_neighbor:
        return torus[x_pos][y_pos]
    else:
        return sum(values_of_neighbors) / len(values_of_neighbors)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #size_row = int(input("Enter size for rows: "))
    # print("Row size: " + str(size_row))

    #size_col = int(input("Enter size for colums: "))
    # print("Colum size: " + str(size_col))

    #prob_for_the_deepest_value = float(input("Enter probability to choose the deepest value: "))

    #max_visit = int(input("Enter size for maximal visit: "))

    #init_x = int(input("Enter initial position X of Random Walker: "))
    #init_y = int(input("Enter initial position Y of Random Walker: "))

    #max_scope_for_torus_value = int(input("Enter maximal value for initialize value of torus: "))
    #scope_for_reduce_value = int(input("Enter value for reduce value of torus: "))

    if max_scope_for_torus_value == 0:
        torus = np.empty(shape=(size_row, size_col))
        torus.fill(0)
    else:
        torus = np.random.randint(-1 * max_scope_for_torus_value, -1, (size_row, size_col), dtype='int64')

    x_locations = []
    y_locations = []

    fig, ax = plt.subplots()

    current_xy_pos_dic = {"x": current_x_pos, "y": current_y_pos}

    for i in range(random_walker_amount):
        current_x_pos[i] = random_walker_init_pos[i][0]
        current_y_pos[i] = random_walker_init_pos[i][1]
        x_locations.append(current_x_pos[i])
        y_locations.append(current_y_pos[i])
        torus[current_x_pos[i]][current_y_pos[i]] = 0

    count = [0]


    def animate(i):
        if count[0] >= max_visit:
            return
        text_to_be_visited_neighbor = ""
        for idx in range(random_walker_amount):
            # saves values on neighbors. used combined with dic_neighbors to find out position of specific cell in torus
            neighbors = []
            # map for neighbors with key = value of torus(2D array) and value = position of torus(2D array)
            dic_neighbors = {}
            # to get values on right/left neighbors
            for ix in [-1, 1]:
                x_pos = current_xy_pos_dic.get("x")[idx] + ix
                y_pos = current_xy_pos_dic.get("y")[idx]
                if x_pos < 0:
                    x_pos = size_row - 1
                if x_pos >= size_row:
                    x_pos = x_pos - size_row
                value = calcu_value_of_neighbor(x_pos, y_pos, torus)
                if str(value) in dic_neighbors:
                    # to avoid to overwrite dic_neighbors if a value is already existing
                    value += (i * 1.0) / max_visit
                neighbors.append(value)
                dic_neighbors[str(value)] = str(x_pos) + "," + str(y_pos)
            # to get values on up/down neighbors
            for iy in [-1, 1]:
                x_pos = current_xy_pos_dic.get("x")[idx]
                y_pos = current_xy_pos_dic.get("y")[idx] + iy
                if y_pos < 0:
                    y_pos = size_col - 1
                if y_pos >= size_col:
                    y_pos = y_pos - size_col
                value = calcu_value_of_neighbor(x_pos, y_pos, torus)
                if str(value) in dic_neighbors:
                    # to avoid to overwrite dic_neighbors if a value is already existing
                    value += (i * 1.0) / max_visit
                neighbors.append(value)
                dic_neighbors[str(value)] = str(x_pos) + "," + str(y_pos)

            # sorts ascend values of neighbors
            neighbors = sorted(neighbors)
            # calculates the rest probabilities to choose other neighbors not with the smallest value
            probabilities = calcu_probs(prob_for_the_deepest_value[idx])
            # chooses a value based on probability
            chosen_value = pick_value_based_on_probability(neighbors, probabilities)
            # returned value from numpy.random.choice is float-type
            if str(chosen_value[0]) not in dic_neighbors:
                chosen_value = chosen_value.astype(int)
            to_be_visited_neighbor = dic_neighbors.get(str(chosen_value[0])).split(",")
            x_pos_to_be_visited_neighbor = int(to_be_visited_neighbor[0])
            y_pos_to_be_visited_neighbor = int(to_be_visited_neighbor[1])
            x_locations.append(x_pos_to_be_visited_neighbor)
            y_locations.append(y_pos_to_be_visited_neighbor)
            current_xy_pos_dic.get("x")[idx] = x_pos_to_be_visited_neighbor
            current_xy_pos_dic.get("y")[idx] = y_pos_to_be_visited_neighbor

            text_to_be_visited_neighbor += str(to_be_visited_neighbor)+" "

            ax.clear()
            #ax.matshow(torus, cmap='gray')

            # set value of visited node after reducing all values of the torus
            torus[x_pos_to_be_visited_neighbor][y_pos_to_be_visited_neighbor] = 0
            #ax.text(x_pos_to_be_visited_neighbor, y_pos_to_be_visited_neighbor, str(0), va='center', ha='center')

        # reduces all the values of torus
        for x in range(size_row):
            for y in range(size_col):
                is_at_current_position = False
                for i in range(random_walker_amount):
                    if x is current_xy_pos_dic.get("x")[i] and y is current_xy_pos_dic.get("y")[i]:
                        is_at_current_position = True
                        break
                if not is_at_current_position:
                    torus[x][y] = torus[x][y] - scope_for_reduce_value
                # ax.text(y, x, str(torus[x][y]), va='center', ha='center')

        # ax.plot(x_locations, y_locations, color='blue', marker='o', linestyle='dashed', markersize=3)
        #ax.scatter(x_locations, y_locations, color='blue', marker='.')
        count[0] += 1
        #fig.suptitle("Visit Count: " + str(count[0]) + "/" + str(max_visit), fontsize=16)
        ax.set_xlim([-1, size_row])
        ax.set_ylim([-1, size_col])
        print("Visit Count: " + str(count[0]), text_to_be_visited_neighbor, sep=" at ")

        # print the torus with its values as matrix at the end to show the 3d landscape
        print(np.matrix(torus))
        # plot the torus as a greyscale matrix
        plt.imshow(torus, cmap='gray')

        text_prob_for_the_deepest_value = ""
        for prob in prob_for_the_deepest_value:
            text_prob_for_the_deepest_value += str(prob) + "% "

        text_current_xy_pos = ""
        for horizontal, vertical in zip(current_x_pos, current_y_pos):
            text_current_xy_pos += "[" + str(vertical) + "," + str(horizontal) + "]" + " "

        plot_text = 'Torus size: ' + str(size_row) + ' x ' + str(size_col) \
                    + '\nProbabilty: ' + text_prob_for_the_deepest_value \
                    + '\nTime steps: ' + str(count[0]) \
                    + '\nAltitude reduction:  = -' + str(scope_for_reduce_value) \
                    + '\nRandom Walker current position: ' + text_current_xy_pos
        plt.suptitle(plot_text,  fontsize=10, horizontalalignment='left', verticalalignment='top', x=.3, y=.99)
        # plot the torus as a greyscale matrix
        plt.imshow(torus, cmap='gray')

    ani = FuncAnimation(fig, animate, frames=max_visit, interval=10, repeat=False)
    plt.show()


