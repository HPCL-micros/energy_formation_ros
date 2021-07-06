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

from tensorflow.python.platform import flags


class EnergyLoader:
    def __init__(self,name,anchor=True):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(name+'.meta')
        self.sess=tf.Session(graph=self.graph,config=tf.ConfigProto(gpu_options=gpu_options))
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess,name)
        self.x_pos = self.graph.get_tensor_by_name('x_pos:0')
        self.x_next = self.graph.get_tensor_by_name('x_next:0')
        self.anchor = anchor
        if anchor:
            self.x_anchor = self.graph.get_tensor_by_name('x_anchor:0')
        self.grads = self.graph.get_tensor_by_name('x_grad/ebm_2/dense/MatMul_grad/MatMul:0')
        self.energy_pos = self.graph.get_tensor_by_name('ebm_4/energy:0')

    def optimize(self,states,anchor=None):
        if self.anchor:
            x, x_grads, e_x = self.sess.run([self.x_next, self.grads, self.energy_pos], feed_dict={self.x_pos: states, self.x_anchor: anchor})
            return x,x_grads,e_x
        else:
            x, x_grads, e_x = self.sess.run([self.x_next, self.grads, self.energy_pos],
                                            feed_dict={self.x_pos: states})
            return x, x_grads, e_x


min_index = -5.0
max_index = 5.0
if __name__ == '__main__':
    squire_energy = EnergyLoader('model/fixed_squire',False)
    triangle_energy = EnergyLoader('model/fixed_triangle',False)
    #test squire
    x = np.random.uniform(min_index, max_index, size=(4 * 2)).tolist()
    #test_anchor = np.random.uniform(min_index, max_index / 2.0, size=(1 * 2)).tolist()
    env = oe.ObjectEnv(min_index, max_index)
    env.clear()
    env.set_from_x_short(x)
    env.show()
    plt.waitforbuttonpress()
    for i in range(0, 500):
        #x, x_grads, e_x =squire_energy.optimize([x],[test_anchor]) #sess.run([x_next, grads, energy_pos], feed_dict={x_pos: [x], x_anchor: [test_anchor]})
        x, x_grads, e_x = squire_energy.optimize([x])
        print("energy = " + str(e_x))
        #print("energy = " + str(e_x) + " anchor = " + str(test_anchor))
        # print (x)
        # print (x_grads)
        x = np.squeeze(x)
        env.clear()
        env.set_from_x_short(x)
        env.show()
        # plt.waitforbuttonpress()
        plt.pause(0.02)
    #test triangle
    x = np.random.uniform(min_index, max_index, size=(3 * 2)).tolist()
    #test_anchor = np.random.uniform(min_index, max_index / 2.0, size=(1 * 2)).tolist()
    env.clear()
    env.set_from_x_short(x)
    env.show()
    plt.waitforbuttonpress()
    for i in range(0, 500):
        #x, x_grads, e_x = triangle_energy.optimize([x],[test_anchor])#sess.run([x_next, grads, energy_pos], feed_dict={x_pos: [x], x_anchor: [test_anchor]})
        x, x_grads, e_x = triangle_energy.optimize([x])
        print("energy = " + str(e_x))
        #print("energy = " + str(e_x) + " anchor = " + str(test_anchor))
        # print (x)
        # print (x_grads)
        x = np.squeeze(x)
        env.clear()
        env.set_from_x_short(x)
        env.show()
        # plt.waitforbuttonpress()
        plt.pause(0.02)