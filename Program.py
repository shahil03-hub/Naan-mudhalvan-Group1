##phase 3
import heapq
import matplotlib.pyplot as plt

grid = [[0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]]

start, end = (0, 0), (4, 4)

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, end):
    heap, visited, came_from = [(0, start)], set(), {}
    g_score = {start: 0}
    while heap:
        _, current = heapq.heappop(heap)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        visited.add(current)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0<=neighbor[0]<len(grid) and 0<=neighbor[1]<len(grid[0]) and grid[neighbor[0]][neighbor[1]]==0:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(end, neighbor)
                    heapq.heappush(heap, (f_score, neighbor))
    return []

path = astar(grid, start, end)
print("Path found:", path)

# Plotting
for r in range(len(grid)):
    for c in range(len(grid[0])):
        if grid[r][c] == 1:
            plt.scatter(c, -r, color='black')
plt.plot([p[1] for p in path], [-p[0] for p in path], marker='o', color='blue')
plt.title("A* Path Planning")
plt.grid()
plt.show()
##1. AI Decision Engine – Path Planning (A)*
##2. Chatbot Command Simulator (Simple Python Version)
 def chatbot(command):
    responses = {
        "start": "Robot is starting...",
        "stop": "Robot has stopped.",
        "status": "All systems are running smoothly.",
        "help": "Available commands: start, stop, status, help"
    }
    return responses.get(command.lower(), "Command not recognized.")

print(chatbot("start"))
print(chatbot("status"))
print(chatbot("stop"))
## 3. IoT Sensor Data Simulation
 import random
import time

def read_ultrasonic_sensor():
    return round(random.uniform(10.0, 100.0), 2)

for i in range(5):
    distance = read_ultrasonic_sensor()
    print(f"Distance: {distance} cm")
    time.sleep(1)
##Secure Communication Simulation (Simple Message Encryption)
from cryptography.fernet import Fernet

# Generate a key
key = Fernet.generate_key()
cipher = Fernet(key)

# Simulated sensor data
data = "Sensor reading: 82.5 cm"

# Encrypt
encrypted = cipher.encrypt(data.encode())
print("Encrypted:", encrypted)

# Decrypt
decrypted = cipher.decrypt(encrypted)
print("Decrypted:", decrypted.decode())
##A* Pathfinding + Chatbot Interaction Example
##phase4
import matplotlib.pyplot as plt from queue import PriorityQueue import ipywidgets as widgets from IPython.display import display, clear_output
 grid (0 = free, 1 = obstacle)

grid_data = [ [0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 0], [0, 0, 0, 0, 0] ]

rows, cols = len(grid_data), len(grid_data[0]) grid = [(r, c) for r in range(rows) for c in range(cols)]

##Helper functions

def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(node, grid): neighbors = [] for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]: r, c = node[0] + dr, node[1] + dc if 0 <= r < rows and 0 <= c < cols and grid_data[r][c] == 0: neighbors.append((r, c)) return neighbors

def cost(a, b): return 1

def reconstruct_path(came_from, current): path = [current] while current in came_from: current = came_from[current] path.append(current) path.reverse() return path

def a_star_optimized(grid, start, end): open_set = PriorityQueue() open_set.put((0, start)) came_from = {} g_score = {node: float('inf') for node in grid} g_score[start] = 0 f_score = {node: float('inf') for node in grid} f_score[start] = heuristic(start, end)

while not open_set.empty():
    current = open_set.get()[1]
    if current == end:
        return reconstruct_path(came_from, current)

    for neighbor in get_neighbors(current, grid):
        tentative_g = g_score[current] + cost(current, neighbor)
        if tentative_g < g_score[neighbor]:
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            f_score[neighbor] = tentative_g + heuristic(neighbor, end)
            open_set.put((f_score[neighbor], neighbor))
return []

##Visualize the path

def visualize_path(path, start, end): for r in range(rows): for c in range(cols): if grid_data[r][c] == 1: plt.plot(c, -r, 'ks') for r, c in path: plt.plot(c, -r, 'go-') plt.plot(start[1], -start[0], 'bo', label="Start") plt.plot(end[1], -end[0], 'ro', label="End") plt.title("A* Path Planning") plt.axis('off') plt.legend() plt.grid() plt.show()

##Chatbot simulation

def chatbot_response(user_input): user_input = user_input.lower() if "navigate" in user_input: start = (0, 0) end = (4, 4) path = a_star_optimized(grid, start, end) print("Navigating to destination...") print("Optimal Path:", path) visualize_path(path, start, end) elif "stop" in user_input: print("Vehicle stopped.") elif "status" in user_input: print("System running. All sensors active.") elif "emergency" in user_input: print("Emergency protocol activated. Notifying control center.") else: print("Sorry, I didn't understand that command.")

##Interface

input_box = widgets.Text(placeholder='Enter command (e.g., navigate)') button = widgets.Button(description="Send") output = widgets.Output()

def on_button_click(b): with output: clear_output() chatbot_response(input_box.value)

button.on_click(on_button_click) display(widgets.VBox([input_box, button, output]))

##phase 5
import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue
from cryptography.fernet import Fernet

# Define environment grid values
EMPTY, OBSTACLE, PEDESTRIAN, VEHICLE = 0, 1, 2, 3
colors = {EMPTY: "white", OBSTACLE: "black", PEDESTRIAN: "magenta", VEHICLE: "cyan"}

# Generate a grid environment
np.random.seed(10)
grid_size = (15, 15)
env = np.random.choice([EMPTY, OBSTACLE, PEDESTRIAN], size=grid_size, p=[0.75, 0.15, 0.1])
env[0, 0] = EMPTY
env[-1, -1] = EMPTY

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, env):
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    neighbors = []
    for dr, dc in directions:
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < env.shape[0] and 0 <= c < env.shape[1] and env[r][c] != OBSTACLE:
            neighbors.append((r, c))
    return neighbors

def a_star(env, start, goal):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g = {start: 0}
    f = {start: heuristic(start, goal)}

    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        for neighbor in get_neighbors(current, env):
            temp_g = g[current] + 1
            if neighbor not in g or temp_g < g[neighbor]:
                came_from[neighbor] = current
                g[neighbor] = temp_g
                f[neighbor] = temp_g + heuristic(neighbor, goal)
                open_set.put((f[neighbor], neighbor))
    return []

def draw_grid(env, path, start, goal):
    fig, ax = plt.subplots(figsize=(8, 8))
    for r in range(env.shape[0]):
        for c in range(env.shape[1]):
            ax.add_patch(plt.Rectangle((c, -r), 1, 1, facecolor=colors[env[r, c]], edgecolor='gray'))
    for r, c in path:
        ax.plot(c + 0.5, -r - 0.5, 'green', marker='o')
    ax.plot(start[1]+0.5, -start[0]-0.5, 'bo', label='Start')
    ax.plot(goal[1]+0.5, -goal[0]-0.5, 'ro', label='Goal')
    ax.set_xlim(0, env.shape[1])
    ax.set_ylim(-env.shape[0], 0)
    ax.set_aspect('equal')
    plt.title("AI Pathfinding in Autonomous Navigation")
    plt.legend()
    plt.show()

def simulate_sensor(current, env):
    print(f"Sensor scanning at position {current}...")
    r, c = current
    for i in range(r-1, r+2):
        for j in range(c-1, c+2):
            if 0 <= i < env.shape[0] and 0 <= j < env.shape[1]:
                cell = env[i][j]
                if cell == OBSTACLE:
                    print(f"Obstacle at {(i, j)}")
                elif cell == PEDESTRIAN:
                    print(f"Pedestrian at {(i, j)}")

def encrypt_path_data(path):
    key = Fernet.generate_key()
    fernet = Fernet(key)
    path_str = ','.join([f"{r}-{c}" for r, c in path])
    encrypted = fernet.encrypt(path_str.encode())
    print("Encrypted path data:", encrypted)
    decrypted = fernet.decrypt(encrypted).decode()
    print("Decrypted path data:", decrypted)
    return encrypted

# Chatbot simulation
def chatbot():
    print("Chatbot: Hello! Type 'start' to begin pathfinding, 'sensor' to scan area, or 'exit' to quit.")
    while True:
        cmd = input("You: ").lower()
        if cmd == "start":
            path = a_star(env, start, goal)
            if path:
                print("Chatbot: Path found! Encrypting and drawing...")
                encrypt_path_data(path)
                draw_grid(env, path, start, goal)
            else:
                print("Chatbot: No path found.")
        elif cmd == "sensor":
            simulate_sensor(start, env)
        elif cmd == "exit":
            print("Chatbot: Goodbye!")
            break
        else:
            print("Chatbot: Unknown command. Try 'start', 'sensor', or 'exit'.")

# Initialize start and goal
start = (0, 0)
goal = (grid_size[0]-1, grid_size[1]-1)

# Run chatbot interface
chatbot()
##Final project

import matplotlib.pyplot as plt
import numpy as np from queue 
import PriorityQueue 
import hashlib
import random


EMPTY, OBSTACLE, PEDESTRIAN, VEHICLE = 0, 1, 2, 3 colors = {EMPTY: "white", OBSTACLE: "black", PEDESTRIAN: "magenta", VEHICLE: "cyan"}

def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, env): directions = [(-1,0), (1,0), (0,-1), (0,1)] neighbors = [] for dr, dc in directions: r, c = pos[0] + dr, pos[1] + dc if 0 <= r < env.shape[0] and 0 <= c < env.shape[1] and env[r][c] != OBSTACLE: neighbors.append((r, c)) return neighbors

def a_star(env, start, goal): open_set = PriorityQueue() open_set.put((0, start)) came_from = {} g = {start: 0} f = {start: heuristic(start, goal)}

while not open_set.empty():
    _, current = open_set.get()
    if current == goal:
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        return path[::-1]

    for neighbor in get_neighbors(current, env):
        temp_g = g[current] + 1
        if neighbor not in g or temp_g < g[neighbor]:
            came_from[neighbor] = current
            g[neighbor] = temp_g
            f[neighbor] = temp_g + heuristic(neighbor, goal)
            open_set.put((f[neighbor], neighbor))
retutn[]

def simulate_sensor_data(): return { "temperature": round(random.uniform(18, 35), 2), "humidity": round(random.uniform(30, 70), 2), "obstacle_distance": round(random.uniform(0.5, 5.0), 2) }

def encrypt_data(data): return hashlib.sha256(data.encode()).hexdigest()


def chatbot_response(user_input): responses = { "function": "I manage autonomous vehicle navigation and environment mapping.", "obstacles": "I use sensor input to detect and avoid obstacles.", "security": "All sensor data is encrypted for safety.", "developer": "Built for AI project under robotics and automation." } return responses.get(user_input.lower(), "Sorry, I didn't understand that.")

def visualize(env, path, start, goal): fig, ax = plt.subplots(figsize=(6, 6)) for r in range(env.shape[0]): for c in range(env.shape[1]): ax.add_patch(plt.Rectangle((c, -r), 1, 1, facecolor=colors[env[r, c]], edgecolor='gray')) for r, c in path: ax.plot(c + 0.5, -r - 0.5, 'go') ax.plot(start[1]+0.5, -start[0]-0.5, 'bo', label='Start') ax.plot(goal[1]+0.5, -goal[0]-0.5, 'ro', label='Goal') ax.set_xlim(0, env.shape[1]) ax.set_ylim(-env.shape[0], 0) ax.set_aspect('equal') plt.title("AI Navigation with Chatbot and IoT Integration") plt.legend() plt.axis('off') plt.show()


np.random.seed(21) grid_size = (10, 10) env = np.random.choice([EMPTY, OBSTACLE, PEDESTRIAN], size=grid_size, p=[0.75, 0.15, 0.1]) env[0, 0] = EMPTY env[-1, -1] = EMPTY

start, goal = (0, 0), (9, 9) path = a_star(env, start, goal) visualize(env, path, start, goal)

Sensor simulation and encryption

sensor_data = simulate_sensor_data() encrypted = {k: encrypt_data(str(v)) for k, v in sensor_data.items()}

Chatbot interaction

sample_query = "function" chat_reply = chatbot_response(sample_query)

Display logs

print("Chatbot Query:", sample_query) print("Chatbot Reply:", chat_reply) print("Sensor Data:", sensor_data) print("Encrypted Data:", encrypted)
