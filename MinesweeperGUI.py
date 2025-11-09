"""
minesweeper_dqn_gui.py

Single-file Minesweeper + GUI + DQN agent.

Controls:
- Left click: (when MANUAL) reveal cell (human)
- F: toggle fullscreen (board scales into a centered 16:9 area)
- R: reset the game (new board)
- SPACE: toggle agent autoplay (agent will take actions)
- T: toggle training on/off (DQN trains in background thread)
- S: save model to 'dqn_minesweeper.pt'
- L: load model from 'dqn_minesweeper.pt'
- +/-: increase / decrease training speed (train steps per second)
- ESC or window close: quit

Notes:
- DQN is intentionally small/simple (suitable for demonstration).
- Training while rendering is slow for big boards, but works for small boards (5x5).
"""

import pygame
import sys
import random
import numpy as np
import threading
import time
import math
import os
from collections import deque, namedtuple

# PyTorch for DQN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --------------
# Environment (your MinesweeperEnv, slightly adapted)
# --------------
class MinesweeperEnv:
    def __init__(self, size=5, n_mines=3):
        self.size = size
        self.n_mines = n_mines
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.mines = set(random.sample(range(self.size * self.size), self.n_mines))
        for m in self.mines:
            x, y = divmod(m, self.size)
            self.board[x, y] = -1  # -1 = mine

        self.visible = np.full((self.size, self.size), False)
        self.done = False
        self._computed = np.copy(self.board)  # keep computed numbers only when revealed
        return self._get_observation()

    def _get_observation(self):
        # -2 hidden, -1 mine (if visible), 0..8 numbers
        obs = np.where(self.visible, self.board, -2)
        return obs.copy()

    def step(self, action):
        # action: (x,y)
        x, y = action
        if self.done or self.visible[x, y]:
            return self._get_observation(), 0.0, self.done, {}

        self.visible[x, y] = True

        if self.board[x, y] == -1:
            self.done = True
            reward = -10.0
        else:
            adj = self._count_adjacent_mines(x, y)
            self.board[x, y] = adj
            reward = 1.0

            if adj == 0:
                for nx, ny in self._neighbors(x, y):
                    if not self.visible[nx, ny]:
                        self.step((nx, ny))

            if self._check_win():
                self.done = True
                reward = 10.0

        return self._get_observation(), float(reward), self.done, {}

    def _neighbors(self, x, y):
        for i in range(max(0, x - 1), min(self.size, x + 2)):
            for j in range(max(0, y - 1), min(self.size, y + 2)):
                if (i, j) != (x, y):
                    yield i, j

    def _count_adjacent_mines(self, x, y):
        return sum((nx * self.size + ny) in self.mines for nx, ny in self._neighbors(x, y))

    def _check_win(self):
        return np.all(self.visible | (self.board == -1))

    def render(self):
        obs = self._get_observation()
        symbols = {-2: "■", -1: "*"}
        for row in obs:
            print(" ".join(symbols.get(v, str(v)) for v in row))
        print()

# --------------
# Simple DQN components
# --------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --------------
# Agent (DQN)
# --------------
class DQNAgent:
    def __init__(self, env, device='cpu',
                 buffer_capacity=50000, batch_size=64,
                 gamma=0.99, lr=1e-3, target_update=200, epsilon_start=1.0,
                 epsilon_final=0.05, epsilon_decay=5000):
        self.env = env
        self.size = env.size
        self.action_n = env.size * env.size
        self.device = torch.device(device)

        # state representation: flatten observation (size*size) with values normalized
        self.state_dim = self.size * self.size

        self.policy_net = DQNNet(self.state_dim, self.action_n).to(self.device)
        self.target_net = DQNNet(self.state_dim, self.action_n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update

        # epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

        self.train_steps = 0

    def state_from_obs(self, obs):
        # obs: matrix with -2 hidden, -1 mines when visible, 0..8 numbers when visible
        # convert to flat normalized vector in [-1,1]
        # mapping: hidden(-2) -> -1.0, mine(-1) -> -0.9, 0..8 -> map to [-0.8..1.0]
        flat = np.array(obs, dtype=float).reshape(-1)
        mapped = np.zeros_like(flat)
        for i, v in enumerate(flat):
            if v == -2:
                mapped[i] = -1.0
            elif v == -1:
                mapped[i] = -0.9
            else:
                # v in 0..8 -> map to [-0.8 .. 1.0]
                mapped[i] = -0.8 + (v / 8.0) * 1.8
        return torch.tensor(mapped, dtype=torch.float32, device=self.device).unsqueeze(0)  # shape [1, state_dim]

    def select_action(self, obs, deterministic=False):
        # returns (x,y)
        state = self.state_from_obs(obs)
        eps = self.epsilon()
        valid_actions = [(i // self.size, i % self.size) for i in range(self.action_n) if obs.flatten()[i] == -2]

        if len(valid_actions) == 0:
            return None

        if deterministic or random.random() > eps:
            # use policy_net, mask invalids
            with torch.no_grad():
                qvals = self.policy_net(state).cpu().numpy().flatten()
            # mask invalid
            masked = np.full_like(qvals, -1e9)
            for (x,y) in valid_actions:
                idx = x * self.size + y
                masked[idx] = qvals[idx]
            idx = int(np.argmax(masked))
            return (idx // self.size, idx % self.size)
        else:
            return random.choice(valid_actions)

    def epsilon(self):
        # linear/exponential decay
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1.0 * self.total_steps / self.epsilon_decay)

    def push_transition(self, state_obs, action_xy, reward, next_obs, done):
        sa = action_xy[0]*self.size + action_xy[1]
        s = self.state_from_obs(state_obs).cpu().numpy().flatten()
        ns = self.state_from_obs(next_obs).cpu().numpy().flatten()
        self.replay.push(s, sa, float(reward), ns, bool(done))

    def optimize_step(self):
        if len(self.replay) < self.batch_size:
            return

        transitions = self.replay.sample(self.batch_size)
        # convert to tensors
        state_batch = torch.tensor(np.array(transitions.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(transitions.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(transitions.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(transitions.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # target: r + gamma * max_a' Q_target(next, a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected = reward_batch + (1 - done_batch) * (self.gamma * next_q)

        loss = F.mse_loss(q_values, expected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_steps += 1

        # update target periodically
        if self.train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path='dqn_minesweeper.pt'):
        torch.save({
            'policy_state': self.policy_net.state_dict(),
        }, path)

    def load(self, path='dqn_minesweeper.pt'):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_state'])
            self.target_net.load_state_dict(checkpoint['policy_state'])
            print(f"[DQN] Loaded model from {path}")
        else:
            print(f"[DQN] No model file at: {path}")

# --------------
# GUI + integration with agent
# --------------
pygame.init()
pygame.font.init()

# Settings
DEFAULT_TILE_SIZE = 60
TILE_SIZE = DEFAULT_TILE_SIZE
FONT = pygame.font.SysFont("Arial", 26)
BIG_FONT = pygame.font.SysFont("Arial", 40)
WIN_FONT = pygame.font.SysFont("Arial", 56)

# Env & initial window
env = MinesweeperEnv(size=5, n_mines=3)
obs = env.reset()

# screen variables (updated in resize/fullscreen logic)
WIDTH = env.size * TILE_SIZE
HEIGHT = env.size * TILE_SIZE + 80
offset_x = 0
offset_y = 0

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Minesweeper DQN")

# Colors map (numbers use same color mapping)
COLORS = {
    -2: (100, 100, 100),  # hidden
    0: (200, 200, 200),
    1: (50, 150, 255),
    2: (50, 200, 50),
    3: (255, 50, 50),
    4: (120, 60, 200),
    -1: (0, 0, 0)
}

# offsets for fullscreen 16:9 handling will be updated globally
offset_x = 0
offset_y = 0

# DQN agent setup (use CPU by default; if CUDA available and desired, change device)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = DQNAgent(env, device=device)

# Agent / training flags
agent_autoplay = False   # starts in MANUAL mode by default
training_on = False
training_thread = None
training_thread_stop = threading.Event()

# training speed control (how many optimization steps per second)
train_speed = 8.0  # adjustable with +/- keys

# helper functions
def resize_window(width, height):
    global screen, WIDTH, HEIGHT, TILE_SIZE, offset_x, offset_y
    WIDTH = width
    HEIGHT = height
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # recalc tile size based on width & height minus UI area (80 px)
    TILE_SIZE = max(8, min(WIDTH // env.size, (HEIGHT - 80) // env.size))
    board_w = TILE_SIZE * env.size
    board_h = TILE_SIZE * env.size
    offset_x = (WIDTH - board_w) // 2
    offset_y = (HEIGHT - 80 - board_h) // 2

def enter_fullscreen_16_9():
    global screen, WIDTH, HEIGHT, TILE_SIZE, offset_x, offset_y
    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    info = pygame.display.Info()
    screen_w = info.current_w
    screen_h = info.current_h

    # 1) find largest 16:9 region that fits
    target_w = screen_w
    target_h = int(target_w * 9 / 16)
    if target_h > screen_h:
        target_h = screen_h
        target_w = int(target_h * 16 / 9)

    # 2) determine tile size from this region (reserve some space for UI below board)
    TILE_SIZE = max(8, min(target_w // env.size, (target_h - 80) // env.size))
    WIDTH = TILE_SIZE * env.size
    HEIGHT = TILE_SIZE * env.size + 80

    # center inside screen
    offset_x = (screen_w - WIDTH) // 2
    offset_y = (screen_h - HEIGHT) // 2

    # set a mode surface that matches full screen size (so we can blit board at offsets properly)
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)

def exit_fullscreen_windowed():
    global screen, WIDTH, HEIGHT, TILE_SIZE, offset_x, offset_y
    TILE_SIZE = DEFAULT_TILE_SIZE
    WIDTH = env.size * TILE_SIZE
    HEIGHT = env.size * TILE_SIZE + 80
    offset_x = 0
    offset_y = 0
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

fullscreen = False

def draw_board(current_obs, highlight_action=None):
    # highlight_action: (x,y) to draw an outline (agent's intended move)
    # clear (fill whole screen)
    screen.fill((30, 30, 30))

    # if fullscreen mode, screen may be larger than WIDTHxHEIGHT: draw board at offsets
    for x in range(env.size):
        for y in range(env.size):
            v = current_obs[x, y]
            rect = pygame.Rect(offset_x + y*TILE_SIZE, offset_y + x*TILE_SIZE, TILE_SIZE, TILE_SIZE)

            if v == -2:
                pygame.draw.rect(screen, COLORS[-2], rect)
            elif v == -1:
                pygame.draw.rect(screen, COLORS[-1], rect)
                # mine circle
                pygame.draw.circle(screen, (255, 0, 0), rect.center, max(6, TILE_SIZE//5))
            else:
                pygame.draw.rect(screen, COLORS[0], rect)
                if v > 0:
                    # pick a color for number
                    color = COLORS.get(v, (10,10,10))
                    text = FONT.render(str(v), True, color)
                    # center text inside rect
                    tw, th = text.get_size()
                    tx = rect.x + (TILE_SIZE - tw)//2
                    ty = rect.y + (TILE_SIZE - th)//2
                    screen.blit(text, (tx, ty))

            pygame.draw.rect(screen, (0,0,0), rect, 2)

    # optional highlight for agent's chosen action
    if highlight_action is not None:
        hx, hy = highlight_action
        if 0 <= hx < env.size and 0 <= hy < env.size:
            r = pygame.Rect(offset_x + hy*TILE_SIZE, offset_y + hx*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, (255,255,0), r, max(3, TILE_SIZE//12))

def draw_ui():
    # draw status text in reserved area below board (or overlay if fullscreen)
    # compute base top-left of UI area
    ui_x = offset_x
    ui_y = offset_y + TILE_SIZE*env.size + 8
    # background bar
    pygame.draw.rect(screen, (25,25,25), (0, ui_y - 6, screen.get_width(), 80))

    # left side: controls
    lines = [
        "F: Toggle Fullscreen | R: Reset | SPACE: Toggle Agent Autoplay",
        "T: Toggle Training | S: Save Model | L: Load Model",
        f"Agent autoplay: {'ON' if agent_autoplay else 'OFF'}    Training: {'ON' if training_on else 'OFF'}    Train speed: {train_speed:.1f} steps/sec",
        f"Epsilon: {agent.epsilon():.3f}    Replay: {len(agent.replay)}"
    ]
    for i, line in enumerate(lines):
        surf = FONT.render(line, True, (230,230,230))
        screen.blit(surf, (ui_x + 10, ui_y + 4 + i*20))

    # center / big message if done
    if env.done:
        if env._check_win():
            msg = "YOU WIN!"
            color = (50, 200, 50)
        else:
            msg = "GAME OVER"
            color = (255, 50, 50)
        txt = WIN_FONT.render(msg, True, color)
        # place center over board area
        bx = offset_x + (TILE_SIZE*env.size - txt.get_width())//2
        by = offset_y + (TILE_SIZE*env.size - txt.get_height())//2
        screen.blit(txt, (bx, by))

# --------------
# Training thread function (runs in background)
# --------------
def training_loop():
    global training_on, agent, training_thread_stop, train_speed
    print("[TRAINING] Background training thread started.")
    while not training_thread_stop.is_set():
        if not training_on:
            time.sleep(0.1)
            continue

        # run one episode per training iteration to gather experience
        state_obs = env.reset()
        done = False
        steps = 0
        # run episode using epsilon-greedy policy, store transitions
        while not done and (not training_thread_stop.is_set()):
            action = agent.select_action(state_obs, deterministic=False)
            if action is None:
                break
            next_obs, reward, done, _ = env.step(action)
            agent.push_transition(state_obs, action, reward, next_obs, done)
            # optimize a few steps
            # compute how many optimize steps to do depending on train_speed
            # we do small number per move to balance UI responsiveness
            for _ in range(max(1, int(train_speed / 4))):
                agent.optimize_step()
            state_obs = next_obs
            steps += 1

        # after episode, perform some more optimization
        for _ in range(int(max(1, train_speed / 2))):
            agent.optimize_step()

        agent.total_steps += steps
        # small sleep to let other operations run and control training rate
        time.sleep(max(0.001, 1.0 / max(1.0, train_speed)))

    print("[TRAINING] Background thread stopping.")

# start training thread (it will be idle until training_on == True)
training_thread_stop.clear()
training_thread = threading.Thread(target=training_loop, daemon=True)
training_thread.start()

# --------------
# Main loop
# --------------
clock = pygame.time.Clock()

# last action chosen by agent to visually highlight (when autoplay running)
agent_last_action = None

# We want the GUI to still be interactive even while training in background.
# Agent autoplay can be toggled and will use the policy_net (epsilon-greedy).
# While autoplay is ON, the GUI will repeatedly ask the agent for an action
# and step the environment at a limited rate (to avoid freezing).
AUTO_STEP_DELAY = 0.15  # seconds between agent steps when autoplay ON
last_auto_step_time = 0.0

def human_click_to_cell(mx, my):
    # convert mouse coords to (x,y) board coord considering offset
    col = (mx - offset_x) // TILE_SIZE
    row = (my - offset_y) // TILE_SIZE
    if row < 0 or col < 0 or row >= env.size or col >= env.size:
        return None
    return (row, col)

# default windowed to 1280x720 for nicer layout (you can call resize_window)
resize_window(1280, 720)

running = True
while running:
    now = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_f:
                # toggle fullscreen and compute 16:9 scaling
                fullscreen = not fullscreen
                if fullscreen:
                    enter_fullscreen_16_9()
                else:
                    exit_fullscreen_windowed()

            elif event.key == pygame.K_r:
                obs = env.reset()
                agent_last_action = None

            elif event.key == pygame.K_SPACE:
                agent_autoplay = not agent_autoplay
                agent_last_action = None
                print(f"[AGENT] Autoplay set to {agent_autoplay}")

            elif event.key == pygame.K_t:
                training_on = not training_on
                print(f"[TRAIN] training_on = {training_on}")

            elif event.key == pygame.K_s:
                agent.save()
                print("[DQN] Model saved to dqn_minesweeper.pt")

            elif event.key == pygame.K_l:
                agent.load()
                print("[DQN] Model loaded (if available)")

            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                train_speed = min(200.0, train_speed + 1.0)
                print(f"[TRAIN] train_speed = {train_speed:.1f}")

            elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                train_speed = max(0.5, train_speed - 1.0)
                print(f"[TRAIN] train_speed = {train_speed:.1f}")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # left click = reveal (only if manual or agent_autoplay==False)
            if event.button == 1:
                pos = pygame.mouse.get_pos()
                c = human_click_to_cell(*pos)
                if c is not None and not env.done:
                    # If autoplay is ON, ignore manual clicks (agent is playing)
                    if not agent_autoplay:
                        obs, r, done, _ = env.step(c)
                        agent_last_action = None

    # Agent autoplay step
    if agent_autoplay and (not env.done) and (now - last_auto_step_time) > AUTO_STEP_DELAY:
        last_auto_step_time = now
        # select action deterministically (use epsilon but allow greedy)
        action = agent.select_action(obs, deterministic=False)
        if action is not None:
            obs, r, done, _ = env.step(action)
            agent_last_action = action

    # If no autoplay and not done, show the agent's best action (deterministic) as hint
    hint_action = None
    if (not agent_autoplay) and (not env.done):
        hint_action = agent.select_action(obs, deterministic=True)

    # draw
    draw_board(obs, highlight_action=agent_last_action or hint_action)
    draw_ui()
    pygame.display.flip()

    clock.tick(60)  # cap frame rate to 60 FPS

# cleanup
training_thread_stop.set()
training_thread.join(timeout=1.0)
pygame.quit()
sys.exit()
