import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import MultipleLocator

def standAngleDiff(a1, a2):
    if (a1 - a2) < -np.pi and (a1 - a2) >= -2*np.pi:
        return (a1 - a2) + 2*np.pi
    else:
        if (a1 - a2) <= 2*np.pi and (a1 - a2) > np.pi:
            return ((a1 - a2) - 2*np.pi)
        else:
            return (a1 - a2)


def lpvf(v0,px,py,tx,ty,rd):
    r=np.sqrt((px-tx)**2+(py-ty)**2)
    A = r*r +rd*rd
    B = r*r - rd*rd
    C = r*rd
    vx_d = -v0/r/A*((px - tx) * B + 2 * (py - ty) * C)
    vy_d = -v0/r/A*((py - ty) * B - 2 * (px - tx) * C)
    phi_d = np.arctan2(-((py - ty) * B - 2 * (px - tx) * C), -((px - tx) * B + 2 * (py - ty) * C))
    diff_phi_d = 4 * v0 * r * C / (A * A)
    return vx_d,vy_d,phi_d,diff_phi_d
    #phi = np.arctan2(vy, vx)
    #u2 = -standAngleDiff(phi, phi_d) + diff_phi_d

def lpvf_yaw_control(px,py,yaw,v,w,tx,ty,rd):
    vx_d, vy_d, phi_d, diff_phi_d = lpvf(v0, px, py, tx, ty, rd)
    u2 = -standAngleDiff(yaw, phi_d) + diff_phi_d
    return u2

def diff_drive_iter(px,py,yaw,v,w,step=0.05):
    vx = v * np.cos(yaw)
    vy = v * np.sin(yaw)
    next_px = px+v*np.cos(yaw)*step
    next_py = py+v*np.sin(yaw)*step
    next_yaw = yaw + w*step
    if next_yaw<-np.pi:
        next_yaw+= 2*np.pi
    if next_yaw>np.pi:
        next_yaw-=2*np.pi
    return next_px,next_py,next_yaw

def phi_sync_control(phi_list,v0,Kdv):
    vel_list = np.array([v0]*len(phi_list))
    for i in range(0,len(phi_list)):
        delta = 0.0
        for j in range(0,len(phi_list)):
            delta += np.sin(standAngleDiff(phi_list[j],phi_list[i]))*Kdv
        vel_list[i] += delta
    return vel_list

min_index = -5.0
max_index = 5.0
x1 = np.linspace(min_index, max_index, 40)
y1 = np.linspace(min_index, max_index, 40)
x1, y1 = np.meshgrid(x1, y1)
#point_x = np.reshape(x1,(40*40,1))
#point_y = np.reshape(y1,(40*40,1))
point_x = x1.flatten()
point_y = y1.flatten()
v0 = 1.0
tx = 1.0
ty = 0.0
rd = 2.0
if __name__ == '__main__':
    plt.figure(figsize=(6,6))
    vx_d,vy_d,phi_d,diff_phi_d = lpvf(v0,point_x,point_y,tx,ty,rd)
    plt.quiver(point_x,point_y,vx_d,vy_d,angles='xy',color='0.7')
    plt.xlim([min_index,max_index])
    plt.ylim([min_index,max_index])
    x=[]#np.linspace(-3, 3, 40)
    y=[]#[1.0]*40
    px = np.random.uniform(min_index,max_index)
    py = np.random.uniform(min_index,max_index)
    yaw = 0
    w = 0
    x.append(px)
    y.append(py)
    for i in range(0,1000):
        w = lpvf_yaw_control(px,py,yaw,v0,w,tx,ty,rd)
        px,py,yaw = diff_drive_iter(px,py,yaw,v0,w,step=0.05)
        x.append(px)
        y.append(py)
    plt.plot(x,y,color='b')
    #plt.legend()
    #plt.draw()
    plt.show()