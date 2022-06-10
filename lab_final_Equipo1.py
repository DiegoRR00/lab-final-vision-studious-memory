
"""
TE3002B Implementación de robótica inteligente
Equipo 1
    Diego Reyna Reyes A01657387
    Samantha Barrón Martínez A01652135
    Jorge Antonio Hoyo García A01658142
Laboratorio Final
Ciudad de México, 10/06/2022
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.h = 0
        self.g = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def pathPlanning(maze, start, end, fig,algo):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    startNode = Node(None, start)
    startNode.g =  startNode.f = 0
    endNode = Node(None, end)
    endNode.g = endNode.f = 0

    # Initialize both open and closed list
    openList = []
    closedList = []

    # Add the start node
    openList.append(startNode)
    
    n = 1
    # Loop until you find the end
    while len(openList) > 0:

        # Get the current node
        currentNode = openList[0]
        current_index = 0
        for index, item in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                current_index = index

        # Pop current off open list, add to closed list
        openList.pop(current_index)
        closedList.append(currentNode)

        # Found the goal
        if currentNode == endNode:
            path = []
            current = currentNode
            while current is not None:
                path.append(current.position)
                current = current.parent
            
            mm = genMaze(maze.copy(), start, end, openList, closedList, path[::-1])
            pltMaze(mm, fig)
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for x, y in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            nodePosition = (currentNode.position[0] + x, currentNode.position[1] + y)

            # Make sure within range
            if nodePosition[0] > (len(maze) - 1) or nodePosition[0] < 0 or nodePosition[1] > (len(maze[len(maze)-1]) -1) or nodePosition[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[nodePosition[0]][nodePosition[1]] != 0:
                continue

            # Create new node
            newNode = Node(currentNode, nodePosition)

            # Append
            children.append(newNode)

        # Loop through children
        for child in children:

            # Child is on the closed list
            b = False
            for closedChild in closedList:
                if child == closedChild:
                    b = True #continue
                    break
            if b:
                continue
            
            # Create the f and g
#            child.g = currentNode.g + 1
            if algo == 0 or algo == 2:
                child.h = ((child.position[0] - startNode.position[0]) ** 2) + ((child.position[1] - startNode.position[1]) ** 2)
            if algo == 1 or algo == 2:
                child.g = ((child.position[0] - endNode.position[0]) ** 2) + ((child.position[1] - endNode.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            b = False
            for openNode in openList:
                if child == openNode and child.f >= openNode.f:
                    b = True #continue
                    break
            if b:
                continue

            # Add the child to the open list
            openList.append(child)
            
        if (n % 15) == 1 :
            mm = genMaze(maze.copy(), start, end, openList, closedList)
            pltMaze(mm, fig)
            
        n = n + 1


# Supporting dunctions
def genMaze(mazeCo, start, end, openList, closedList, path = None):
    idx = mazeCo  == 1
    mx = 1
    for no in openList:
        mazeCo[no.position] = no.g
        if no.f > mx:
            mx = no.f
        
    for no in closedList:
        mazeCo[no.position] = no.f
        if no.f > mx:
            mx = no.f
    if path != None:
        for p in path:
            mazeCo[p] = mx + 15
    
    mazeCo[start] = mx + 2
    mazeCo[end] = mx + 2
    mazeCo[idx] = mx + 3
    return mazeCo
        
def pltMaze(maze, fig):
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    
    ax.matshow(maze, cmap=plt.cm.bone)
    for (i, j), z in np.ndenumerate(maze):
        plt.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')

 
    plt.draw()
    plt.pause(0.0001)
#    plt.clf()



def reduce(img):
    """
    Reduces the white spaces to oround 1 pixel
    """
    act = img
    k = np.ones((3,3),np.uint8) #Kernel for erode
    #Erode it
    for i in range(1):
        act = cv2.erode(act,k,iterations=1)  
    return act

#Load Image
maze = cv2.imread("maze.jpg")
maze = cv2.resize(maze,None,fx=0.2,fy=0.2) #Resize so it's easier to solve the maze
#Convert the image to Gray Scale
maze_g = cv2.cvtColor(maze,cv2.COLOR_BGR2GRAY)
#Binarize the image using a threshold and normalize it to either Zero or One
_,maze_thre = cv2.threshold(maze_g, 127, 255, cv2.THRESH_BINARY)
inicio = np.where(maze_thre[int(maze_thre.shape[0]/2)]==0) #Eliminate the edges
maze_thre = maze_thre[:,inicio[0][0]:inicio[0][-1]+1]
#Reduce the paths
reduct = reduce(maze_thre)
reduct = cv2.bitwise_not(reduct)
reduct = cv2.resize(reduct,None,fx=0.5,fy=0.5)
_,reduct = cv2.threshold(reduct, 129, 255, cv2.THRESH_BINARY)
maze_bin = reduct /255
#Define the start and finish points
start = (0,np.where(maze_bin[int(0)]==0)[0][0])
end = (maze_bin.shape[0]-1,np.where(maze_bin[-1]==0)[0][0])

#Create figure
fig = plt.figure(figsize = (8, 8))
mazeCopy = maze_bin.copy()
mazeCopy[start] = 40
mazeCopy[end] = 40
pltMaze(maze_bin, fig)

#For each algorithm
alogrithms = ["Dijkstra","Greedy","A_star"]
steps_algo = []
for algo in range(3):
    #Create the path
    path = pathPlanning(maze_bin, start, end, fig,algo)
    mazeCopy_2 = mazeCopy.copy()
    #Create the matrix to save
    for i,j in path:
        mazeCopy_2[i][j] = 20
    mazeCopy_2[start] = 40
    mazeCopy_2[end] = 40
    pltMaze(mazeCopy_2, fig)
    plt.savefig(alogrithms[algo]+".png")
    #Print the path
    print("The path for ",alogrithms[algo]," is: ",path)
    print()
    #Save the lenth of the file
    steps_algo.append(len(path))
    plt.pause(1)
#Find ann print the shortest
shortest = steps_algo.index(min(steps_algo))
print("The shortest algorithm is: ",alogrithms[shortest])


