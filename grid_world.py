import numpy as np
import random
import matplotlib.pyplot as plt
from torch import clone
import cv2
from PIL import Image
import wandb
from gym import spaces

class GridWorld:
    '''
    A simple grid world environment
    '''
    def __init__(self, maxEpisodeLength : int):
        '''
        Initialise the map and other env related params
        Map details:
            1 -> obstacle
            0 -> free space
        '''
        self.action_space = []
        self.observation_space = []
        self.maxEpisodeLength = maxEpisodeLength
        self.n_agents = 5
        self.dim = [400, 400]
        self.totalCollision = 0
        self.stepCount = 0

        self.action_space = spaces.Tuple(tuple(self.n_agents*[
            spaces.Box(low=-1, high=1, shape=(1,2), dtype=np.float32)
        ]))
        self.observation_space = spaces.Tuple(tuple(self.n_agents*[
            spaces.Box(low=0, high=255, shape=(400, 400), dtype=np.float32)
        ]))

        # self.env = np.array(self.generate_grid())
        # self.env = np.array([
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # ])
        self.env = np.array(self.generate_rgb_warehouse())

        # plt.figure(figsize=(8, 8))
        # plt.imshow(self.env, cmap="binary", origin="upper")
        # plt.title("50x50 Binary Grid World Map (1: Wall, 0: Free Space)")
        # plt.axis("off")
        # plt.show()

        self.start = [[12, 12], [2, 3], [3, 3], [1, 4], [10, 1]] 
        self.Goals = [[5, 14], [18, 9], [10, 2], [2, 14], [14, 18]] 

        self.prevPositions = [[1, 1], [2, 3], [3, 4], [4, 5], [5, 6]]
        self.currentPositions = [[1, 1], [2, 3], [3, 4], [4, 5], [5, 6]] 
        self.goals = [[3, 2], [8, 7], [4, 8], [6, 1], [1, 2]]
    
    
    def plot_warehouse_grid(self, warehouse):
        img = Image.fromarray(warehouse)
        img.show()
    
    def generate_rgb_warehouse(self, dim_x=400, dim_y=400, wall_color=(0, 0, 0), open_color=(255, 255, 255)):
        # Initialize a 400x400 grid with the open color (white)
        grid = np.ones((dim_x, dim_y, 3), dtype=np.uint8) * np.array(open_color, dtype=np.uint8)

        for i in range(dim_x):
            for j in range(dim_y):
                # Border walls (1 for walls, 0 for open space)
                if i == 0 or i == dim_x - 1 or j == 0 or j == dim_y - 1:
                    grid[i, j] = wall_color
                # Adding walls in a pattern that ensures connectivity
                elif (i % 80 == 0 and j % 80 < 40) or (j % 80 == 0 and i % 80 < 40):
                    grid[i, j] = wall_color
                elif (i % 80 == 40 and j % 40 == 20) or (j % 80 == 40 and i % 40 == 20):
                    grid[i, j] = wall_color

        return grid
    
    
    def generate_grid(self):
        grid = []
        for i in range(20):
            row = []
            for j in range(20):
                # Border walls
                if i == 0 or i == 19 or j == 0 or j == 19:
                    row.append(1)
                # Adding walls in a pattern that ensures connectivity
                elif (i % 8 == 0 and j % 8 != 7) or (j % 8 == 0 and i % 8 != 7):
                    row.append(1)
                elif (i % 8 == 4 and j % 4 == 2) or (j % 8 == 4 and i % 4 == 2): 
                    row.append(1)
                else:
                    row.append(0)
            grid.append(row)
        return grid


    def makeObs(self, agent : int)->list:
        '''
        Flattened 3x3 grid with the agent at the center
        '''
        agentPosition = self.currentPositions[agent]
        start = [agentPosition[0]-1, agentPosition[1]-1]
        end = [agentPosition[0]+1, agentPosition[1]+1]
        obs = self.env[start[0]:end[0]+1, start[1]:end[1]+1]
        return obs.flatten()
    
    def makeSharedObs(self, agent):

        pass

    # Function to find the orientation of the ordered triplet (p, q, r)
    # 0 -> p, q, r are collinear
    # 1 -> Clockwise
    # 2 -> Counterclockwise
    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counterclockwise

    # Function to check if point q lies on line segment pr
    def on_segment(self, p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    # Function to check if two line segments p1p2 and q1q2 intersect
    def do_intersect(self, p1, p2, q1, q2):
        # Find the four orientations needed for the general and special cases
        o1 = self.orientation(p1, p2, q1)
        o2 = self.orientation(p1, p2, q2)
        o3 = self.orientation(q1, q2, p1)
        o4 = self.orientation(q1, q2, p2)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases
        # p1, p2, q1 are collinear and q1 lies on segment p1p2
        if o1 == 0 and self.on_segment(p1, q1, p2):
            return True
        # p1, p2, q2 are collinear and q2 lies on segment p1p2
        if o2 == 0 and self.on_segment(p1, q2, p2):
            return True
        # q1, q2, p1 are collinear and p1 lies on segment q1q2
        if o3 == 0 and self.on_segment(q1, p1, q2):
            return True
        # q1, q2, p2 are collinear and p2 lies on segment q1q2
        if o4 == 0 and self.on_segment(q1, p2, q2):
            return True

        # If none of the cases are true, then the segments don't intersect
        return False

        
    def reset(self, flag=False)->list:
        # Sample starting points for agents
        self.stepCount = 0
        # for a in range(self.n_agents):
        #     dim0 = random.sample(range(1, self.dim[0]-1), 1)[0]
        #     dim1 = random.sample(range(1, self.dim[1]-1), 1)[0]
        #     while (self.env[dim0, dim1]==1):
        #         dim0 = random.sample(range(1, self.dim[0]-1), 1)[0]
        #         dim1 = random.sample(range(1, self.dim[1]-1), 1)[0]
        #     self.currentPositions[a] = [dim0, dim1]
        #     self.env[dim0, dim1] = 1
        
        # for a in range(self.n_agents):
        #     self.env[self.currentPositions[a][0], self.currentPositions[a][1]] = 0
        
        # print("Pos : ", self.currentPositions)
        
        # for a in range(self.n_agents):
        #     dim0 = random.sample(range(1, self.dim[0]-1), 1)[0]
        #     dim1 = random.sample(range(1, self.dim[1]-1), 1)[0]
        #     while (self.env[dim0, dim1]==1):
        #         dim0 = random.sample(range(1, self.dim[0]-1), 1)[0]
        #         dim1 = random.sample(range(1, self.dim[1]-1), 1)[0]
        #     self.goals[a] = [dim0, dim1]
        #     self.env[dim0, dim1] = 1
        
        # for a in range(self.n_agents):
        #     self.env[self.currentPositions[a][0], self.currentPositions[a][1]] = 0
        
        # print("Goals : ", self.goals)

        self.currentPositions = self.start.copy()
        self.goals = self.Goals.copy()

        # if (flag):
        #     print(self.currentPositions)
        
        # print(f"Starting positions : {self.currentPositions}\n"+
        #       f"Goals : {self.goals}\n")

        # Make observations for the agents
        obs_ = []
        for a in range(self.n_agents):
            obs_.append(self.env)
        return obs_
    
    def checkCollision(self, prev_coordinate, coordinate):
        if (coordinate[0]>=self.dim[0] or coordinate[0]<0 or coordinate[1]>=self.dim[1] or coordinate[1]<0):
            return 1
        else:
            diff = [coordinate[0]-prev_coordinate[0], coordinate[1]-prev_coordinate[1]]
            n = 5
            step = [diff[0]/5, diff[1]/5]
            for i in range(1, 6):
                coord = [int(prev_coordinate[0]+i*step[0]), int(prev_coordinate[1]+i*step[1])]
                if (all(self.env[coord[0], coord[1]]==[0, 0, 0])):
                    return 1
            return 0
    
    def step(self, actions : list):
        '''
        Takes a step in the environment
        Reward structure:
            1 -> Goal reached 
            0 -> Collision with the walls or other agents
            0 -> Everywhere else
        
        Actions :
            0 -> Move up
            1 -> Move down
            2 -> Move left
            3 -> Move right
            4 -> Stay
            
        Question : Should we have agents as obstacles in the observation?
        '''
        # for a in range(self.n_agents):
        #     self.env[self.currentPositions[a][0], self.currentPositions[a][1]] = 0
        self.stepCount += 1
        collisionRew = -1
        goalRew = 10
        timeRew = 0
        rewards = [0]*self.n_agents
        done = [0]*self.n_agents
        self.totalCollision = 0

        for a in range(self.n_agents):
            action = clone(actions[a]).detach()
            if (self.checkCollision(self.currentPositions[a], [self.currentPositions[a][0]+5*action[0], self.currentPositions[a][1]+5*action[1]])):
                rewards[a] = collisionRew
            else:
                self.prevPositions[a] = self.currentPositions[a].copy()
                self.currentPositions[a] = [self.currentPositions[a][0]+int(action[0]*5), self.currentPositions[a][1]+int(action[1]*5)]
            '''
            if actions[a]==0:
                if (self.env[self.currentPositions[a][0]-1, self.currentPositions[a][1]]==1):
                    rewards[a] = collisionRew 
                    self.totalCollision+=1
                else:
                    self.currentPositions[a] = [self.currentPositions[a][0]-1, self.currentPositions[a][1]]
            
            elif (actions[a]==1):
                if (self.env[self.currentPositions[a][0]+1, self.currentPositions[a][1]]==1):
                    rewards[a] = collisionRew
                    self.totalCollision+=1
                else:
                    self.currentPositions[a] = [self.currentPositions[a][0]+1, self.currentPositions[a][1]]

            elif (actions[a]==2):
                if (self.env[self.currentPositions[a][0], self.currentPositions[a][1]-1]==1):
                    rewards[a]= collisionRew
                    self.totalCollision+=1
                else:
                    self.currentPositions[a] = [self.currentPositions[a][0], self.currentPositions[a][1]-1]
            
            elif (actions[a]==3):
                if (self.env[self.currentPositions[a][0], self.currentPositions[a][1]+1]==1):
                    rewards[a] = collisionRew
                    self.totalCollision+=1
                else:
                    self.currentPositions[a] = [self.currentPositions[a][0], self.currentPositions[a][1]+1]
            
            elif (actions[a]==4):
                if (self.currentPositions[a]==self.goals[a]):
                    rewards[a] = goalRew
            
            else:
                print(f"Unknown Action")
            '''
            
        obs_ = []
        for a in range(self.n_agents):
            for b in range(a+1, self.n_agents):
                if (self.do_intersect(self.prevPositions[a], self.currentPositions[a], self.prevPositions[b],  self.currentPositions[b])):
                    rewards[a] = collisionRew
                    rewards[b] = collisionRew
            
            if (self.currentPositions[a]==self.goals[a]):
                # print(f"Agent {a} is done in step {self.stepCount}!")
                done[a] = 1
                rewards[a] = goalRew
            else:
                rewards[a] += timeRew
            
            obs_.append(self.env)
        
        return obs_, rewards, done, {}
    
    def close(self):
        pass
        
def main():
    env = GridWorld(100)
    map = env.generate_rgb_warehouse()
    env.plot_warehouse_grid(map)



if __name__=="__main__":
    main()


    
    


