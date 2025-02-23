import numpy as np
import math
import heapq as hq
from typing import List, Tuple, Optional

def heuristic(start: Tuple[int, int], end: Tuple[int, int]):
    return math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

def a_star(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    pq = []
    hq.heappush(pq, (0, start)) # f, (x, y)
    
    g_score = {start: 0}
    parent = {start: None}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while pq:
        _, (x, y) = hq.heappop(pq)
        if (x, y) == end:
            path = []
            while (x, y) is not None:
                path.append((x, y))
                if parent[(x, y)] is None:
                    break
                x, y = parent[(x, y)]
            return path[::-1]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < rows and 0 <= ny < cols) or grid[nx][ny] == 1:
                continue

            g = g_score[(x, y)] + 1
            h = heuristic((nx, ny), end)
            f = g + h

            if (nx, ny) not in g_score or g < g_score[(nx, ny)]:
                parent[(nx, ny)] = (x, y)
                g_score[(nx, ny)] = g
                hq.heappush(pq, (f, (nx, ny)))

    return None

def plan_path(world: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[np.ndarray]:
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))

    world_list: List[List[int]] = world.tolist()

    path = a_star(world_list, start, end)

    return np.array(path) if path else None
