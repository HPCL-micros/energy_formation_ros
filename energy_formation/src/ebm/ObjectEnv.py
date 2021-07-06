import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import MultipleLocator

potential_force=[]
potential = []



def generate_line_demo(batch_size=128): # No.1and3 line,No.2 random
    print('generating line demo...')
    line_demo_x0 = []
    line_demo_x1 = []
    line_demo_a = []
    for i in range(0,batch_size):
        demo_x0 = []
        demo_x1 = []
        demo_a = [10.0,0.0,10.0]                     #biased: No.2 no attention
        for j in range(0,2):
            demo_x0.append(random.uniform(-1,1))
            demo_x0.append(random.uniform(-1, 1))
            demo_x0.append(0.5)
            demo_x0.append(0.5)
        demo_x0.append(random.uniform(-1, 1))
        demo_x0.append(demo_x0[1])                      #same y: line
        demo_x0.append(0.5)
        demo_x0.append(0.5)
        line_demo_x0.append(demo_x0)
        line_demo_x1.append(demo_x0)                    #x1 =x0
        line_demo_a.append(demo_a)
    print('finished generating line demo')
    return line_demo_x0,line_demo_x1,line_demo_a

def generate_line_center(batch_size=128):  #dim=2 ignore color and shape
    print('generating line center...')
    line_demo_x0 = []
    line_demo_x1 = []
    line_demo_a = []
    for i in range(0, batch_size):
        demo_x0 = []
        demo_x1 = []
        demo_a = [10.0,0.0,10.0]                     #biased: No.2 no attention
        for j in range(0,2):
            demo_x0.append(random.uniform(-1,1))
            demo_x0.append(random.uniform(-1, 1))
        demo_x0.append(random.uniform(-1, 1))
        demo_x0.append(demo_x0[1])                      #same y: line
        for j in range(0,3):
            demo_x1.append(demo_x0[2*j])
            demo_x1.append(demo_x0[2 * j+1])
        demo_x1[2] = (demo_x0[0]+demo_x0[4])/2.0
        demo_x1[3] = demo_x0[1]

        line_demo_x0.append(demo_x0)
        line_demo_x1.append(demo_x1)
        line_demo_a.append(demo_a)
    print('finished generating line center')
    return line_demo_x0,line_demo_x1,line_demo_a


def generate_line_two(batch_size=128,min_index=-1.0,max_index=1.0):  #dim=2 ignore color and shape
    print('generating line two...')
    line_demo_x0 = []
    for i in range(0, batch_size):
        demo_x0 = []
        demo_x0.append(random.uniform(min_index,max_index))
        demo_x0.append(random.uniform(min_index, max_index))
        demo_x0.append(random.uniform(min_index, max_index))
        demo_x0.append(demo_x0[1])                      #same y: line
        line_demo_x0.append(demo_x0)
    print('finished generating line two')
    return line_demo_x0

def generate_fixed_triangle(batch_size=128,shuffle=False,min_index=-1.0,max_index=1.0,return_anchor=False,c=1.0):
    print('generating fixed triangle...')
    line_demo_x0 = []
    anchors = []
    #c = 1.0
    for i in range(0,batch_size):
        demo_x0=[]
        anchor = []
        demo_x0.append(random.uniform(min_index, max_index-c))
        demo_x0.append(random.uniform(min_index, max_index-c/2))
        demo_x0.append(demo_x0[0]+c)
        demo_x0.append(demo_x0[1])
        x3 = 0.5*(demo_x0[0]+demo_x0[2])
        y3 = 0.5*abs(demo_x0[0]-demo_x0[2])+demo_x0[1]
        demo_x0.append(x3)
        demo_x0.append(y3)
        anchor.append(demo_x0[0])
        anchor.append(demo_x0[1])
        if shuffle:
            demo_x0=np.reshape(np.array(demo_x0),(3,2)).tolist()
            #print (demo_x0)
            random.shuffle(demo_x0)
            #print (demo_x0)
            demo_x0 = np.reshape(np.array(demo_x0), (3*2)).tolist()
        line_demo_x0.append(demo_x0)
        anchors.append(anchor)
    print('finished generating fixed triangle')
    if return_anchor:
        return line_demo_x0,anchors
    return line_demo_x0

def generate_fixed_squire(batch_size=128,shuffle=False,min_index=-1.0,max_index=1.0,return_anchor=False,c=1.0):
    print('generating fixed squire...')
    line_demo_x0 = []
    anchors = []
    #c = 1.0
    for i in range(0,batch_size):
        demo_x0=[]
        anchor = []
        demo_x0.append(random.uniform(min_index, max_index-c))
        demo_x0.append(random.uniform(min_index, max_index-c))
        demo_x0.append(demo_x0[0]+c)
        demo_x0.append(demo_x0[1])
        demo_x0.append(demo_x0[0])
        demo_x0.append(demo_x0[1]+c)
        demo_x0.append(demo_x0[0] + c)
        demo_x0.append(demo_x0[1] + c)
        anchor.append(demo_x0[0])
        anchor.append(demo_x0[1])
        if shuffle:
            demo_x0 = np.reshape(np.array(demo_x0), (4, 2)).tolist()
            # print (demo_x0)
            random.shuffle(demo_x0)
            # print (demo_x0)
            demo_x0 = np.reshape(np.array(demo_x0), (4 * 2)).tolist()
        line_demo_x0.append(demo_x0)
        anchors.append(anchor)
    print('finished generating fixed squire')
    if return_anchor:
        return line_demo_x0,anchors
    return line_demo_x0

def triangle_error(c,x0,y0,x1,y1,x2,y2):
    total = 0
    total += abs(y1-y0)
    total += abs(x1-(x0+c))
    total += abs(x2-(x0+c/2.0))
    total += abs(y2-(y0+c/2.0))
    #print (abs(y1-y0),abs(x1-(x0+c)),abs(x2-(x0+c/2.0)),abs(y2-(y0+c/2.0)))
    return total, total/2.0

def squire_error(c,x0,y0,x1,y1,x2,y2,x3,y3):
    total = 0
    total += abs(y1-y0)
    total += abs(x1-(x0+c))
    total += abs(x2-x0)
    total += abs(y2-(y0+c))
    total += abs(x3-(x0+c))
    total += abs(y3 - (y0 + c))
    return total, total/3.0



