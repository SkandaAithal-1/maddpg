import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
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
        self.dim = [20, 20]
        self.totalCollision = 0
        self.stepCount = 0

        self.action_space = spaces.Tuple(tuple(self.n_agents*[spaces.Discrete(5)]))
        self.observation_space = spaces.Tuple(tuple(self.n_agents*[
            spaces.Box(low=0, high=1, shape=(20, 20), dtype=np.float32)
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
        self.env = np.array(self.generate_grid())


        # Set the outer ring as walls (1)
        # self.env[0, :] = 1
        # self.env[-1, :] = 1
        # self.env[:, 0] = 1
        # self.env[:, -1] = 1

        # num_internal_walls = int(0.1 * self.dim[0] * self.dim[1])

        # # Place random internal walls
        # for _ in range(num_internal_walls):
        #     x, y = np.random.randint(1, self.dim[0] - 1), np.random.randint(1, self.dim[1] - 1)
        #     self.env[x, y] = 1
        
        # plt.figure(figsize=(8, 8))
        # plt.imshow(self.env, cmap="binary", origin="upper")
        # plt.title("50x50 Binary Grid World Map (1: Wall, 0: Free Space)")
        # plt.axis("off")
        # plt.show()

        self.start = [[12, 12], [2, 3], [3, 3], [1, 4], [10, 1]] 
        self.Goals = [[5, 14], [18, 9], [10, 2], [2, 14], [14, 18]] 

        self.currentPositions = [[1, 1], [2, 3], [3, 4], [4, 5], [5, 6]] 
        self.goals = [[3, 2], [8, 7], [4, 8], [6, 1], [1, 2]]
    
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
            
        obs_ = []
        for a in range(self.n_agents):
            for b in range(a+1, self.n_agents):
                if (self.currentPositions[a]==self.currentPositions[b]):
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
        



    
    


