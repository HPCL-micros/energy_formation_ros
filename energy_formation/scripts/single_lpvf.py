#!/usr/bin/env python
import rospy
from rosplane_msgs.msg import Controller_Commands
from rosplane_msgs.msg import State
import numpy as np
import matplotlib.pyplot as plt



rd = 30.0

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


    #print w
    
class UAVhandle:
    def __init__(self,number=0):
        self.pub = rospy.Publisher('/uav'+str(number)+'/controller_commands', Controller_Commands, queue_size=10)
        self.sub = rospy.Subscriber('/uav'+str(number)+'/truth', State, self.stateCB)
        self.px,self.py,self.v,self.w,self.psi=0.1,0.1,0,0,0
        
    def stateCB(self,truth_msg):
        
        self.px = truth_msg.position[0]
        self.py = -truth_msg.position[1]
        self.v = truth_msg.Va
        self.w = truth_msg.r
        self.psi = -truth_msg.psi
        
def phi_sync_control(phi_list,v0,Kdv):
    vel_list = np.array([v0]*len(phi_list))
    for i in range(0,len(phi_list)):
        delta = 0.0
        for j in range(0,len(phi_list)):
            delta += np.sin(standAngleDiff(phi_list[j],phi_list[i]))*Kdv
        vel_list[i] += delta
    return vel_list

if __name__ == '__main__':
    rospy.init_node('single_lvpf', anonymous=True)
    uav_number = rospy.get_param("~uav_number", 2)
    uav_handles = []
    targets = [[0,0],[30,0]]
    phi_list = [0.0]*uav_number
    r_list = [0.0]*uav_number
    for i in range(0,uav_number):
        uav_handles.append(UAVhandle(i))
    #pub = rospy.Publisher('/fixedwing/controller_commands', Controller_Commands, queue_size=10)
    #sub = rospy.Subscriber("/fixedwing/truth", State, stateCB)
    #sub_reset = rospy.Subscriber("/env_worker/reset", EnvAction, resetCB)
    #w_t = 1.0
    #step = 1.0
    #psi_t = 0
    #x1 = np.linspace(-10, 90, 100)
    #y1 = np.linspace(-10, 90, 100)
    #x1, y1 = np.meshgrid(x1, y1)
    #point_x = x1.flatten()
    #point_y = y1.flatten()
    #plt.figure(figsize=(6,6))
    #vx_d,vy_d,phi_d,diff_phi_d = lpvf(12,point_x,point_y,0.0,0.0,rd)
    #plt.quiver(point_x,point_y,vx_d,vy_d,angles='xy',color='0.7')
    #plt.xlim([-10,90])
    #plt.ylim([-10,90])
    #plt.show()
    for i in range(0,6000):
        uav_v = phi_sync_control(phi_list=phi_list,v0=12.0,Kdv=5.0)
        for j in range(0,uav_number):
            phi_list[j] = np.arctan2(uav_handles[j].py-targets[j][1],uav_handles[j].px-targets[j][0])
            r_list[j] = np.sqrt((uav_handles[j].px-targets[j][0])**2+(uav_handles[j].py-targets[j][1])**2)
            _,_,phi_d,diff = lpvf(12.0,uav_handles[j].px,uav_handles[j].py,targets[j][0],targets[j][1],rd)
            psi_t = -phi_d
            #psi_t = psi+w_t*step
            #if i%50 == 0:
            #    psi_t +=np.pi/2.0
            if psi_t>np.pi:
                psi_t -= 2*np.pi
            if psi_t<-np.pi:
                psi_t += 2*np.pi    
            msg = Controller_Commands()
            msg.Va_c = uav_v[j]
            msg.h_c = 30.0
            msg.chi_c = psi_t
            uav_handles[j].pub.publish(msg)
        print phi_list
        print r_list
        #print phi_d,uav_handle.psi,uav_handle.w,np.sqrt((uav_handle.px-0.0)**2+(uav_handle.py-0.0)**2)
        rospy.sleep(0.1)
    rospy.spin()

