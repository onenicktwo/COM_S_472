# Path Planning Algorithm Development

## Overview

This project aims to develop a path planning algorithm to solve a series of grid navigation tasks. Each task consists of a grid-based environment where an agent must navigate from a given **start** position to a defined **end** position while avoiding obstacles. 

## Project Structure
```plaintext
project_root/
│── data/                # Contains 100 task configurations (grid files with start and end positions), along with the generated paths, if applicable
│── main.py              # Provides utility functions for loading grids, visualization, and validity checks (DO NOT modify)
│── planner.py           # The main file for students to implement the path planning algorithm
│── devel.py             # An example script to test the implementation (modifiable for development but NOT submitted)
│── README.md            # Project documentation
```

## Task Configurations

- The `data/` folder contains **100 different task configurations**, each represented as a grid file.
- Each file specifies a **grid representation**, along with a **start** and **end** position in the .csv file.
- These configurations define the challenges that the implemented path planner must solve.

## Implementation Details

### `main.py` (DO NOT MODIFY)
- Loads grid files and initializes task configurations.
- Provides visualization utilities to display the grid and planned paths.
- Includes a function for path validity checks.

### `planner.py` (MAIN DEVELOPMENT FILE)
- This is where students will **implement their path planning algorithm**.
- A **DFS (Depth-First Search) example** is provided to illustrate the function structure.
- **The path_planning function name and input-output structure must remain unchanged**, but the internal implementation can be modified.
- Students can use any suitable path planning approach, such as **BFS, A*, Dijkstra, or other heuristic-based algorithms**.
- Students can only use the following packages:
  - The Python standard libraries
  - Numpy
  - Scipy
  - Matplotlib
  - Pandas
- Directly re-using the provided example is plagiarism. Using only heuristics without a formal searching algorithm is also plagiarism.

### `devel.py` (MODIFIABLE, BUT NOT SUBMITTED)
- Provides an example of solving **one task** using functions from `main.py` and `planner.py`.
- Students are **free to modify** this file for **development and testing purposes**.
- **This file will NOT be included in the final submission**.

## Submission Requirements

- **Only `planner.py` should be submitted** to Canvas (along with the project report in typeset pdf).
- The path_planning function signature inside `planner.py` must remain unchanged.
- The implementation should be tested using `main.py` before submission.

## Getting Started

1. Read the program in `main.py` to understand the task structure and validity criterion.
2. Implement the path planning algorithm inside `planner.py`.
3. Use `devel.py` to test and debug the implementation.
4. Ensure the planned paths are valid by executing
  ```bash
  python main.py
  ```
5. Submit only the `planner.py` file.
