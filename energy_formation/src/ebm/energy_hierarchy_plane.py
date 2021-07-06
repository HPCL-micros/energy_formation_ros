#!/usr/bin/env python
import os
from functools import reduce
from operator import mul
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import ObjectEnv as oe
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from EBMloader import EnergyLoader
import lpvf
from tensorflow.python.platform import flags
import rospy
from rosplane_msgs.msg import Controller_Commands
from rosplane_msgs.msg import State
class UAVhandle:
    def __init__(self, number=0):
        self.rcv = False
        self.pub = rospy.Publisher('/uav' + str(number) + '/controller_commands', Controller_Commands, queue_size=10)
        self.sub = rospy.Subscriber('/uav' + str(number) + '/truth', State, self.stateCB)
        self.px, self.py, self.v, self.w, self.psi = 0.1, 0.1, 0, 0, 0

    def stateCB(self, truth_msg):
        self.px = truth_msg.position[0]
        self.py = -truth_msg.position[1]
        self.v = truth_msg.Va
        self.w = truth_msg.r
        self.psi = -truth_msg.psi
        self.rcv = True

min_index = -5.0
max_index = 5.0
use_anchor = False
shuffle_points = False
squire_size = 4.0
triangle_size = 2.0
velocity = 0.05
v0 = 1.2
rd = 3.0
#TODO: not implemented. waiting for localization.
if __name__ == '__main__':
    rospy.init_node('energy_hierarchy_plane', anonymous=True)
    logfile = open('/home/czx/fixed_wing_uavs.txt', 'w')
    if not use_anchor:
        uav_handles = []
        phi_list = [0.0] * 12
        r_list = [0.0] * 12
        targets = [[0., 0.]] * 12
        for i in range(0, 12):
            uav_handles.append(UAVhandle(i))
        ready = False
        while not ready:
            ready = True
            for i in range(0,12):
                ready = ready and uav_handles[i].rcv
            rospy.sleep(0.02)

        squire_energy = EnergyLoader('model/fixed_squire4_ns',False)
        triangle_energy = EnergyLoader('model/fixed_triangle2_ns',False)
        x0 = oe.generate_fixed_triangle(1000, shuffle_points, min_index, max_index,c=triangle_size)
        x0 = np.reshape(x0, (1000, 3 * 2))
        _, _, e_x0_tri = triangle_energy.optimize(x0)
        x0 = oe.generate_fixed_squire(1000, shuffle_points, min_index, max_index, c=squire_size)
        x0 = np.reshape(x0, (1000, 4 * 2))
        _, _, e_x0_squ = squire_energy.optimize(x0)
        e_squ_mean = np.mean(e_x0_squ)
        e_tri_mean = np.mean(e_x0_tri)
        print("energy mean: squ="+str(np.mean(e_x0_squ))+",tri="+str(np.mean(e_x0_tri)))
        squire_x = np.random.uniform(min_index, max_index, size=(4 * 2)) # squire points
        x = np.random.uniform(min_index, max_index, size=(4, 2 * 2)) # triangle points

        for i in range(0,4):
            squire_x[2*i] =uav_handles[i].px/10.0+0.5
            squire_x[2*i+1] = uav_handles[i].py/10.0
        for i in range(0,4):
            x[i][0] = uav_handles[2*i+4].px/10.0+0.5
            x[i][1] = uav_handles[2*i + 4].py / 10.0
            x[i][2] = uav_handles[2*i+1 + 4].px / 10.0 + 0.5
            x[i][3] = uav_handles[2*i+1 + 4].py / 10.0


        x_squeezing = np.reshape(x, (4 * 2 * 2))
        x_squeezing = np.concatenate([x_squeezing, squire_x], axis=0)
        uav_x = np.array(x_squeezing) - 0.5
        uav_yaw = np.array([0.0] * 12)
        uav_w = np.array([0.0] * 12)
        uav_v = np.array([v0] * 12)
        uav_phi = np.array([-0.75 * np.pi] * 12)

        for i in range(0,5000):
            squire_x = squire_x.tolist()
            squire_x_new, squire_x_grads, e_x = squire_energy.optimize([squire_x])
            squire_x = np.squeeze(squire_x)
            squire_x_grads = np.squeeze(squire_x_grads)
            vel = squire_x_grads / np.linalg.norm(squire_x_grads) * velocity
            current_energy = np.mean(e_x)
            #print(current_energy)
            #print(e_x0_squ)
            if current_energy < 10.0 * e_squ_mean:
                vel = vel * current_energy / (10.0 * e_squ_mean)
            squire_x = squire_x - vel
            tri = np.reshape(squire_x,(4,2))
            for j in range(0,4):
                tri_x = np.concatenate([tri[j],x[j]])
                triangle_x, triangle_x_grads, e_x = triangle_energy.optimize([tri_x])
                triangle_x = np.squeeze(triangle_x)
                triangle_x_grads = np.squeeze(triangle_x_grads)
                vel = triangle_x_grads / np.linalg.norm(triangle_x_grads) * velocity
                current_energy = np.mean(e_x)
                if current_energy < 10.0 * e_tri_mean:
                    vel = vel * current_energy / (10.0 * e_tri_mean)
                tri_x = tri_x - vel
                for k in range(2,6):
                    #x[j][k-2]=triangle_x[k]
                    x[j][k - 2] = tri_x[k]
            x_squeezing = np.reshape(x, (4 * 2 * 2))
            x_squeezing = np.concatenate([x_squeezing, squire_x], axis=0)

            for j in range(0, 12):
                u = lpvf.lpvf_yaw_control(px=uav_x[2 * j], py=uav_x[2 * j + 1], yaw=uav_yaw[j], v=v0, w=uav_w[j],
                                          tx=x_squeezing[2 * j], ty=x_squeezing[2 * j + 1], rd=rd)
                # v_list = lpvf.yaw_sync_control(uav_phi,v0,Kdv=0.05)
                px, py, yaw = lpvf.diff_drive_iter(px=uav_x[2 * j], py=uav_x[2 * j + 1], yaw=uav_yaw[j], v=uav_v[j],
                                                   w=uav_w[j], step=0.05)
                uav_x[2 * j] = px
                uav_x[2 * j + 1] = py
                uav_yaw[j] = yaw
                uav_w[j] = u
                uav_phi[j] = np.arctan2(uav_x[2 * j + 1] - x_squeezing[2 * j + 1], uav_x[2 * j] - x_squeezing[2 * j])
            uav_v = lpvf.phi_sync_control(phi_list=uav_phi, v0=v0, Kdv=0.05)

            plane_v = lpvf.phi_sync_control(phi_list=phi_list, v0=9.0, Kdv=5.0)
            for j in range(0, 12):
                targets[j][0] = x_squeezing[2 * j] * 10.0
                targets[j][1] = x_squeezing[2 * j + 1] * 10.0
                phi_list[j] = np.arctan2(uav_handles[j].py - targets[j][1], uav_handles[j].px - targets[j][0])
                r_list[j] = np.sqrt(
                    (uav_handles[j].px - targets[j][0]) ** 2 + (uav_handles[j].py - targets[j][1]) ** 2)
                _, _, phi_d, diff = lpvf.lpvf(9.0, uav_handles[j].px, uav_handles[j].py, targets[j][0], targets[j][1],
                                              25.0)
                psi_t = -phi_d
                # psi_t = psi+w_t*step
                # if i%50 == 0:
                #    psi_t +=np.pi/2.0
                if psi_t > np.pi:
                    psi_t -= 2 * np.pi
                if psi_t < -np.pi:
                    psi_t += 2 * np.pi
                msg = Controller_Commands()
                msg.Va_c = plane_v[j]
                msg.h_c = 30.0
                msg.chi_c = psi_t
                uav_handles[j].pub.publish(msg)
            print ("phi_list=" + str(phi_list))
            print ("r_list=" + str(r_list))

            s_error_t, s_error_a = oe.squire_error(40, uav_handles[8].px, uav_handles[8].py, uav_handles[9].px, uav_handles[9].py, uav_handles[10].px, uav_handles[10].py,
                                                   uav_handles[11].px, uav_handles[11].py)
            t1_error_t, t1_error_a = oe.triangle_error(20, uav_handles[8].px, uav_handles[8].py, uav_handles[0].px, uav_handles[0].py, uav_handles[1].px, uav_handles[1].py)
            t2_error_t, t2_error_a = oe.triangle_error(20, uav_handles[9].px, uav_handles[9].py, uav_handles[2].px, uav_handles[2].py, uav_handles[3].px, uav_handles[3].py)
            t3_error_t, t3_error_a = oe.triangle_error(20, uav_handles[10].px, uav_handles[10].py, uav_handles[4].px, uav_handles[4].py, uav_handles[5].px,
                                                       uav_handles[5].py)
            t4_error_t, t4_error_a = oe.triangle_error(20, uav_handles[11].px, uav_handles[11].py, uav_handles[6].px, uav_handles[6].py, uav_handles[7].px,
                                                       uav_handles[7].py)

            t1_e_t, t1_e_a = oe.triangle_error(20, uav_handles[8].px, uav_handles[8].py, uav_handles[0].px, uav_handles[0].py, uav_handles[1].px, uav_handles[1].py)
            t2_e_t, t2_e_a = oe.triangle_error(20, uav_handles[8].px + 40, uav_handles[8].py, uav_handles[2].px, uav_handles[2].py, uav_handles[3].px, uav_handles[3].py)
            t3_e_t, t3_e_a = oe.triangle_error(20, uav_handles[8].px, uav_handles[8].py + 40, uav_handles[4].px, uav_handles[4].py, uav_handles[5].px, uav_handles[5].py)
            t4_e_t, t4_e_a = oe.triangle_error(20, uav_handles[8].px + 40, uav_handles[8].py + 40, uav_handles[6].px, uav_handles[6].py, uav_handles[7].px, uav_handles[7].py)

            all_e_t = s_error_t + t1_e_t + t2_e_t + t3_e_t + t4_e_t
            all_e_a = all_e_t / 11.0

            alltri_e_a = (t1_e_a + t2_e_a + t3_e_a + t4_e_a) / 4.0
            intertri_e_a = (t1_error_a + t2_error_a + t3_error_a + t4_error_a) / 4.0
            logfile.write(str(i)+' '+str(rospy.get_time()) + ' ' + str(all_e_a) + ' ' + str(s_error_a) + ' ' + str(alltri_e_a) + ' ' + str(
                intertri_e_a) + '\n')

            rospy.sleep(0.02)
        exit(1)