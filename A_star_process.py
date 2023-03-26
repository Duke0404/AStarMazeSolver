from time import sleep
from IPython.display import clear_output
import numpy as np
import heapq

WALL = "#"
OPEN = "-"
MANHATTAN = True
EUCLIDEAN = False

DISPLAY_TIME = 0.1 # seconds

HEURISTIC_TYPE = MANHATTAN
HEURISTIC_TYPE = EUCLIDEAN
# FILENAME = "maze1"
# START = (0, 1)
# GOAL = (2, 4)

# FILENAME = "maze2"
# START = (0, 0)
# GOAL = (3, 3)

# FILENAME = "maze3"
# START = (0, 0)
# GOAL = (4, 7)

# FILENAME = "maze4"
# START = (0, 0)
# GOAL = (2, 0)

FILENAME = "maze5"
START = (16, 0)
GOAL = (16, 16)

# FILENAME = "maze6"
# START = (1, 1)
# GOAL = (29, 35)

class PriorityQueue:
    def __init__(self):
        self.queue = []
        self._index = 0
    def push(self, item, priority):
        # heappush 在队列 _queue 上插入第一个元素
        heapq.heappush(self.queue, (priority, self._index, item))
        self._index += 1
    def pop(self):
        # heappop 在队列 _queue 上删除第一个元素
        return heapq.heappop(self.queue)[-1]


class Node:
    def __init__(self, x: int, y: int, g=0, h=0):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f
    
    def coordinates(self):
        return (self.x, self.y)

def read_maze_file(filename: str):
    with open(filename + ".txt", "r") as file:
        maze = np.array([list(line.strip()) for line in file])

    if len(maze) == 0:
        raise Exception("Maze file is empty")
    
    for row in maze:
        if len(row) != len(maze[0]):
            raise Exception("Maze file is malformed")
    
    return maze

def position(coordinates: tuple[int, int], maze: np.ndarray):
    x = coordinates[0]
    y = coordinates[1]

    if x < 0 or x >= len(maze) or y < 0 or y >= len(maze[0]) or maze[x][y] == WALL:
        raise ValueError("Invalid position")
    else:
        return Node(x, y)


def heuristic(a: Node, b: Node, type=MANHATTAN):
    if type == MANHATTAN:
        return abs(a.x - b.x) + abs(a.y - b.y)
    elif type == EUCLIDEAN:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def print_maze(maze: np.ndarray, checked: set[tuple[int, int]], path: list[tuple[int, int]], start: Node, end: Node):
    clear_output(wait=False if DISPLAY_TIME > 0 else True)

    START_SYMBOL = "▶️"
    END_SYMBOL = "★"
    PATH_SYMBOL = "x"
    CHECKED_SYMBOL = "o"
    OPEN_SPACE_SYMBOL = "□"
    WALL_SYMBOL = "■"

    output = ""
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if (i, j) == start.coordinates():
                output += "\x1b[32m" + START_SYMBOL + "\x1b[0m "
            elif (i, j) == end.coordinates():
                output += "\x1b[31m" + END_SYMBOL + "\x1b[0m "
            elif (i, j) in path:
                output += "\x1b[33m" + PATH_SYMBOL + "\x1b[0m "
            elif (i, j) in checked:
                output += "\x1b[36m" + CHECKED_SYMBOL + "\x1b[0m "
            elif maze[i][j] == OPEN:
                output += OPEN_SPACE_SYMBOL + " "
            elif maze[i][j] == WALL:
                output += WALL_SYMBOL + " "
        output += "\n"

    print(output[:-1])



def a_star(maze: np.ndarray, start: Node, goal: Node, display_time=0.25, h_type=MANHATTAN):
    checked = set()
    open = PriorityQueue()
    open.push(start,[0,0])
    # open = [start]
    while len(open.queue) > 0:
        #heapq.heapify(open.queue)
        current = open.pop()
        #current = heapq.heappop(open)
        checked.add((current.x, current.y))

        # Print maze at current step
        if display_time > 0:
            print_maze(maze, checked, [], start, goal)
            sleep(display_time)

        # If goal is reached
        if current.x == goal.x and current.y == goal.y:
            path = []
            while current is not None:
                path.append((current.x, current.y))
                current = current.parent

            # Print final maze with path
            print_maze(maze, checked, path[::-1], start, goal)
            return

        children = []

        # Generate children
        for dx, dy in [(-1, 0), (1, 0),(0, -1), (0, 1)]:
            node_position = (current.x + dx, current.y + dy)

            # If new position is out of bounds or a wall or already in checked set
            if (
                node_position[0] < 0 or node_position[0] >= len(maze) or
                node_position[1] < 0 or node_position[1] >= len(maze[0]) or
                maze[node_position[0]][node_position[1]] == WALL or
                node_position in checked
            ):
                continue

            heuristic_value = heuristic(
                Node(node_position[0], node_position[1]), goal, h_type)
            new_node = Node(
                node_position[0], node_position[1], current.g + 1, heuristic_value)
            new_node.parent = current
            children.append(new_node)

        # Loop through children
        for child in children:
            if (child.x, child.y) in checked:
                continue
            open.push(child,[child.f,child.h])
            # heapq.heappush(open, child)
            checked.add((child.x, child.y))

maze = read_maze_file(FILENAME)
start = position(START, maze)
goal = position(GOAL, maze)

a_star(maze, start, goal, DISPLAY_TIME, HEURISTIC_TYPE)
