import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

from planner import plan_path

class Task:
    """Base class of a task"""
    def __init__(self, id):
        self.id = id

        self.world = np.load("data/grid_files/grid_"+str(id)+".npy")
        tasks = pd.read_csv("data/grid_tasks.csv")
        self.start = np.array([tasks["StartX"][id], tasks["StartY"][id]])
        self.end = np.array([tasks["EndX"][id], tasks["EndY"][id]])
        self.obstacles = np.array(np.where(self.world==1)).T

    def plan_path(self):
        """
        The main path planner

        path: the generated path is in the form of a N X 2 numpy array.
        """
        path = plan_path(self.world, self.start, self.end)

        if path is not None:
            if not os.path.exists("data/proj_i_solutions/"):
                os.makedirs("data/proj_i_solutions/")
            path_df = pd.DataFrame(data=path, columns=["X", "Y"])
            path_df.to_csv("data/proj_i_solutions/"+str(self.id)+".csv", index=False)

    def check_path(self):
        """
        Check the validity of the generated path

        True for a valid path
        """
        if os.path.exists("data/proj_i_solutions/"+str(self.id)+".csv"):
            path = pd.read_csv("data/proj_i_solutions/"+str(self.id)+".csv").to_numpy()
            is_start = np.all(path[0]==self.start)
            is_end = np.all(path[-1]==self.end)
            is_continuous = np.all(abs(path[1:] - path[0:-1]) <= 1)
            is_inbound = np.all(np.min(path,axis=1)>=0) and np.all(np.max(path,axis=1)<=29)
            is_safe = not set(map(tuple, path)) &  set(map(tuple, self.obstacles))
            if is_start and is_end and is_continuous and is_inbound and is_safe:
                return True
            else:
                return False
        else:
            raise ValueError("cannot find the path csv file")

    def visualize_path(self):
        """
        Visualizing the grid and the generated path with different colors of grids
        """
        if os.path.exists("data/proj_i_solutions/"+str(self.id)+".csv"):
            path = pd.read_csv("data/proj_i_solutions/"+str(self.id)+".csv").to_numpy()
            grid = np.zeros((30, 30))
            for (i,j) in self.obstacles:
                grid[i][j] = -1
            for (i,j) in path:
                grid[i][j] = -2
            grid[self.start[0], self.start[1]] = 1
            grid[self.end[0], self.end[1]] = 2
            plt.imshow(grid, interpolation='nearest')
            plt.ylabel("x")
            plt.xlabel("y")
            plt.show()
        else:
            raise ValueError("cannot find the path csv file")

if __name__ == "__main__":
    # Test all grid tasks
    for id in range(100):
        T = Task(id)
        T.plan_path()
        print ("Task ID: %s, Check Path: %s" % (str(id), str(T.check_path())))