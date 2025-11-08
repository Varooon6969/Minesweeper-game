import numpy as np
import random

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
        return self._get_observation()

    def _get_observation(self):
        obs = np.where(self.visible, self.board, -2)
        return obs.copy()

    def step(self, action):
        x, y = action
        if self.done or self.visible[x, y]:
            return self._get_observation(), 0, self.done, {}

        self.visible[x, y] = True

        if self.board[x, y] == -1:
            self.done = True
            reward = -10
        else:
            adj = self._count_adjacent_mines(x, y)
            self.board[x, y] = adj
            reward = 1

            if adj == 0:
                for nx, ny in self._neighbors(x, y):
                    if not self.visible[nx, ny]:
                        self.step((nx, ny))

            if self._check_win():
                self.done = True
                reward = 10

        return self._get_observation(), reward, self.done, {}

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
