import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import subprocess
import os
from pathlib import Path

from tqdm import tqdm

import gudhi as gd
from gudhi.point_cloud.timedelay import TimeDelayEmbedding


# %%
# Define classes

class Joints:
    def __init__(self, i, x, y, xborder, yborder):
        self.x = x
        self.y = y
        self.xborder = xborder
        self.yborder = yborder
        self.name = i
        self.origin = (x, y)

    def change_xy(self, x, y):
        self.x = x
        self.y = y


class Knee_Joints:
    def __init__(self, number_steps=10):
        self.origin = Joints(0, 0, 0, [0, 0], [0, 0])

        # each point is connected to its
        # adjacent list elements
        self.points = [self.origin]
        self.history = []
        self.time = 0
        self.n_steps = number_steps

    def add_point_relative(self,
                           xshift=0,
                           yshift=-1,
                           xborder=[-1, 1],
                           yborder=[0, 0]):
        joint = Joints(len(self.points), 
                       self.points[-1].x + xshift,
                       self.points[-1].y + yshift,
                       xborder,
                       yborder)
        self.points.append(joint)

    def add_point(self, x, y,
                  xborder=[-1, 1],
                  yborder=[0, 0]):
        joint = Joints(len(self.points),
                       x,
                       y,
                       xborder,
                       yborder)
        self.points.append(joint)

    def move_joint(self, name=None,
                   method='sinoidal',
                   noise_size=0,
                   bandwidth=0.1,
                   time_updater=1):
        # self.history.append({joint.name:
        #                      [joint.origin, (joint.x, joint.y)]
        #                      for joint in self.points})
        self.time += time_updater

        if name is None:
            name = [self.points[i] for i in range(1, len(self.points))]
        else:
            name = [self.points[i] for i in name]
            # now we ignore xshift and yshift

        #
        # xshift = [
        #           + (joint.xborder[1]-joint.xborder[0])
        #           * np.sin(2* np./pi * self.time / self.n_steps)
        #           for joint in name]
        # yshift = [joint.origin[1]
        #           + np.random.uniform(joint.yborder[0], joint.yborder[1])
        #           for joint in name]

        # always have the sinoidal wave
        xshift = [(joint.xborder[1]-joint.xborder[0])/2
                  * np.sin(2 * np.pi * self.time / self.n_steps)
                  + (joint.xborder[1]+joint.xborder[0])/2
                  for joint in name]

        yshift = [(joint.yborder[1]-joint.yborder[0])/2
                  * np.sin(2 * np.pi * self.time / self.n_steps)
                  + (joint.yborder[1]+joint.yborder[0])/2 + joint.y
                  for joint in name]

        if method == 'sinoidal':
            # xshift = [(joint.xborder[1]-joint.xborder[0])/2
            #           * np.sin(2 * np.pi * self.time / self.n_steps)
            #           + (joint.xborder[1]+joint.xborder[0])/2
            #           for joint in name]

            # yshift = [(joint.yborder[1]-joint.yborder[0])/2
            #           * np.sin(2 * np.pi * self.time / self.n_steps)
            #           + (joint.yborder[1]+joint.yborder[0])/2 + joint.y
            #           for joint in name]
            pass

        elif method == 'sinoidal_noise':
            # make a lot of noise on all points
            # xshift = [min(max((joint.xborder[1]-joint.xborder[0])/2
            #           * np.sin(2 * np.pi * self.time / self.n_steps)
            #           + (joint.xborder[1]+joint.xborder[0])/2
            #           + np.random.uniform(low=-noise_size,
            #                               high=noise_size),
            #           joint.xborder[0]), joint.xborder[1])
            #           for joint in name]

            # yshift = [(joint.yborder[1]-joint.yborder[0])/2
            #           * np.sin(2 * np.pi * self.time / self.n_steps)
            #           + (joint.yborder[1]+joint.yborder[0])/2 + joint.y
            #           for joint in name]

            xshift = [min(max(xshift[i]
                      + np.random.uniform(low=-noise_size,
                                          high=noise_size),
                      name[i].xborder[0]), name[i].xborder[1])
                      for i in range(len(name))]

        elif method == 'last_point':
            t_step = self.time / self.n_steps
            bandwidth = np.array([-bandwidth, bandwidth])

            # check if t_step is either in the interval
            # [0.25-0.02, 0.25+0.02] or
            if t_step > 0.25 + bandwidth[0] and t_step < 0.25 + bandwidth[1]:
                stop = 0.25 + bandwidth[0]
            elif t_step > 0.75 + bandwidth[0] and t_step < 0.75 + bandwidth[1]:
                stop = 0.75 + bandwidth[0]
            else:
                stop = t_step

            xshift[-1] = ((name[-1].xborder[1]-name[-1].xborder[0])/2
                          * np.sin(2 * np.pi * stop)
                          + (name[-1].xborder[1]+name[-1].xborder[0])/2
                          + name[-1].x)
            xshift[-1] = min(max(xshift[-1]
                                 + np.random.uniform(low=-noise_size,
                                                     high=noise_size),
                             name[-1].xborder[0]),
                             name[-1].xborder[1])

        elif method == 'skip_overextension':
            t_step = self.time / self.n_steps
            bandwidth = np.array([-bandwidth, bandwidth])

            # check if t_step is either in the interval
            # [0.25-0.02, 0.25+0.02] or
            if t_step > 0.25 + bandwidth[0] and t_step < 0.25 + bandwidth[1]:
                self.time += np.ceil(self.n_steps
                                     * (0.25 + bandwidth[1])
                                     - self.time)
            elif t_step > 0.75 + bandwidth[0] and t_step < 0.75 + bandwidth[1]:
                self.time += np.ceil(self.n_steps
                                     * (0.75 + bandwidth[1])
                                     - self.time)

            xshift = [(joint.xborder[1]-joint.xborder[0])/2
                      * np.sin(2 * np.pi * self.time / self.n_steps)
                      + (joint.xborder[1]+joint.xborder[0])/2
                      for joint in name]
            xshift[-1] = min(max(xshift[-1]
                                 + np.random.uniform(low=-noise_size,
                                                     high=noise_size),
                                 name[-1].xborder[0]),
                             name[-1].xborder[1])

            yshift = [(joint.yborder[1]-joint.yborder[0])/2
                      * np.sin(2 * np.pi * self.time / self.n_steps)
                      + (joint.yborder[1]+joint.yborder[0])/2 + joint.y
                      for joint in name]

        else:
            # TODO add better message
            raise ValueError('Method should be one of ...')

        for i, joint in enumerate(name):
            joint.change_xy(xshift[i], yshift[i])

        self.history.append({joint.name: (joint.x, joint.y)
                            for joint in self.points})

    def save_current(self):
        # self.history.append({joint.name:
        #                      [joint.origin, (joint.x, joint.y)]
        #                      for joint in self.points})
        if self.time == len(self.history) or self.time == len(self.history)-1:
            self.history.append({joint.name: (joint.x, joint.y)
                                for joint in self.points})

    def get_history_array(self):
        points = []
        for pts in self.history:
            keys = list(pts.keys())
            pts_arr = [[pts[key][0], pts[key][1]]
                       for key in keys]
            points.append(pts_arr)
        return(np.array(points))

    def get_bounding_box(self):
        bbox = np.array([[joint.xborder[0]+joint.origin[0],
                          joint.xborder[1]+joint.origin[0],
                          joint.yborder[0]+joint.origin[1],
                          joint.yborder[1]+joint.origin[1]]
                         for joint in self.points])
        return([np.min(bbox[:, 0]), np.max(bbox[:, 1]),
                np.min(bbox[:, 2]), np.max(bbox[:, 3])])

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        G = nx.Graph()

        G.add_nodes_from([x.name for x in self.points])
        G.add_edges_from([(i, i+1)
                          for i in range(len(self.points)-1)])

        pos = [(joint.x, joint.y) for joint in self.points]
        nx.draw(G, pos, ax=ax)
        return (fig, ax)

# %%
# define helper functions for plotting


def get_rows_cols(n_subfigs):
    if n_subfigs > 3:
        cols = np.ceil(np.sqrt(n_subfigs))
        rows = (n_subfigs+1) // cols
    else:
        cols = n_subfigs
        rows = 1
    return(int(rows), int(cols))


def get_ax_list(ax, rows, cols, n_subfigs):
    if rows == cols and rows == 1:
        ax_list = [ax]
    elif rows == 1:
        ax_list = [ax[i] for i in range(n_subfigs)]
    else:
        ax_list = [ax[np.unravel_index(i, [rows, cols])]
                   for i in range(n_subfigs)]
    return (ax_list)


def create_fig(n_subfigs, row_factor=4, col_factor=4, **kwargs):
    if isinstance(n_subfigs, int):
        n = n_subfigs
    else:
        n = len(n_subfigs)
    rows, cols = get_rows_cols(n)
    fig, ax = plt.subplots(rows, cols,
                           figsize=(col_factor*cols,
                                    row_factor*rows), **kwargs)
    ax_list = get_ax_list(ax, rows, cols, rows*cols)
    return(fig, ax_list)
