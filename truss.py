#!/usr/bin/env python

"""
Class to represent a bridge Truss and optimize it.
See readme.md for more details
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import imageio

class Truss:
    def __init__(self, nodes : np.ndarray, edges : np.ndarray) -> None:
        """Nodes: Nx2 np array of X,Y coord pairs.
        Edges: Mx2 array of (i,j) edge pairs"""
        self.n_nodes = len(nodes)
        self.n_edges = len(edges)
        assert nodes.shape == (self.n_nodes, 2)
        assert edges.shape == (self.n_edges, 2)

        # nodes are stored as list of tensors so that some can require gradients, but others not
        self.node_tensors = [torch.tensor(node, requires_grad=True) for node in nodes]
        self.edges = edges
        self.loads = []
        self.anchors : list[tuple] = []
        self.forces = None
        self._validate_edges()
    
    def _validate_edges(self):
        """Validates that all edges make sense and have no duplicates"""
        edge_set = set()
        for i, j in self.edges:
            assert 0 <= i < self.n_nodes, f"index error: {i}"
            assert 0 <= j < self.n_nodes, f"index error for {j}"
            edge = frozenset((i, j))
            assert edge not in edge_set, f"Duplicate edge {(i,j)}!"
            edge_set.add(edge)
    
    def add_load(self, idx, fx, fy):
        """Adds a force acting on the given node"""
        assert 0 <= idx < self.n_nodes
        self.loads.append((idx, fx, fy))
        self.node_tensors[idx].requires_grad = False

    def add_anchor(self, idx : int, x : bool = True, y : bool = True):
        """Sets node at idx to be an anchor in possibly the x and y dimensions.
        An anchor node does not need to reach equilibrium, because any
        excess force will be taken in by the ground. It should not be
        included in the system of equations!"""
        assert 0 <= idx < self.n_nodes
        self.anchors.append((idx, x, y))
        self.node_tensors[idx].requires_grad = False
    
    def draw(self) -> None:
        """Draws the graph and the forces on it"""
        _, ax = plt.subplots()
        plt.axis('equal')

        # get values out of tensor list
        xs = np.array([n[0].detach() for n in self.node_tensors])
        ys = np.array([n[1].detach() for n in self.node_tensors])
        
        # draw edges
        if self.forces is None:
            for i, j in self.edges:
                plt.plot(xs[[i, j]], ys[[i, j]] , 'k-', zorder=-1)
        else:
            for (i, j), f in zip(self.edges, self.forces):
                color = 'r' if f > 0 else 'b'
                plt.plot(
                    xs[[i, j]], ys[[i, j]], 
                    f'{color}-',
                    linewidth=abs(f)+0.1,
                    zorder=-1)

        # draw loads
        for idx, fx, fy in self.loads:
            fx /= 10
            fy /= 10
            plt.arrow(xs[idx] - fx, ys[idx] - fy, fx, fy,  # subtract force to put tip at node
                head_width = 0.2,
                width = 0.05,
                color='green',
                length_includes_head=True)
            
        # draw anchors
        for idx, x_constrained, y_constrained in self.anchors:
            constrained_size = 0.1
            unconstrained_size = 0.3
            width = constrained_size if x_constrained else unconstrained_size
            height = constrained_size if y_constrained else unconstrained_size
            center = (xs[idx] - width / 2, ys[idx] - height / 2)
            ax.add_patch(Rectangle(center, width, height, edgecolor="green", facecolor="green"))

        # draw nodes
        plt.scatter(xs, ys)

    def _get_unit_xy(self, dx, dy):
        """Return the unit components of x, y normalized by total vector length"""
        hypot = torch.sqrt(dx**2 + dy**2)
        return dx / hypot, dy / hypot

    def calculate_forces(self):
        """Computes the force on each beam!
        Construct a matrix A representing system of equations, and solve.
        A @ Force_Vec = Load_vec
        Positive forces are compression, negative are tension.
        """

        # create design matrix
        A = torch.zeros((2 * self.n_nodes, self.n_edges))
        for edge_i, (n1_i, n2_i) in enumerate(self.edges):

            n1x, n1y = self.node_tensors[n1_i]
            n2x, n2y = self.node_tensors[n2_i]
            ux, uy = self._get_unit_xy(n1x - n2x, n1y - n2y)

            # towards N1
            A[n1_i * 2][edge_i], A[n1_i * 2 + 1][edge_i] = ux, uy

            # towards N2
            A[n2_i * 2][edge_i], A[n2_i * 2 + 1][edge_i] = -ux, -uy

        # construct load vector
        load_vec = torch.zeros((2 * self.n_nodes, 1))  # stored as [x1,y1,x2,y2...]
        for idx, fx, fy in self.loads:
            load_vec[idx * 2] += fx
            load_vec[idx * 2 + 1] += fy
        
        # handle anchor points: anchor points can have any force, so they can simply be
        # removed from the system of equations!
        # Zeroing instead of removing to preserve indexing...
        for node_idx, x, y in self.anchors:
            if x:
                A[node_idx * 2, :] = 0.0
                load_vec[node_idx * 2] = 0.0  # loads on anchor points are irrelevant
            if y:
                A[node_idx * 2 + 1, :] = 0.0
                load_vec[node_idx * 2 + 1] = 0.0  # loads on anchor points are irrelevant
        
        # use gelsd solver to handle non-full rank matrices
        self.forces, residuals, rank, s = torch.linalg.lstsq(A, -load_vec, rcond=None, driver='gelsd')
        
        # print(f"{A=}")
        # print(f"{self.forces=}")
        # print(f"{residuals=}")
        # print(f"{rank=}")
        # print(f"{s=}")
        # print(f"{A@self.forces=}")

        residual_threshold = 0.1
        if len(residuals) > 0 and residuals.item() > residual_threshold:
            raise ValueError("Nodes could not reach equilibrium!")

        return self.forces
    
    def get_cost(self):
        """Calculates the cost of the truss from the lengths and forces.
        Tension weight = length * force / 1
        compression weight = length * force / (3 / (length + 3))
        """

        self.calculate_forces()

        lengths = torch.zeros_like(self.forces)
        for i, (n1, n2) in enumerate(self.edges):
            lengths[i] = torch.linalg.norm(self.node_tensors[n1] - self.node_tensors[n2])
    
        weights = torch.zeros_like(lengths)

        # beams in tension
        weights = lengths * torch.abs(self.forces)

        # compression
        fall_off = 3.0
        weights[self.forces > 0] *= (lengths[self.forces > 0] + fall_off) / fall_off

        return torch.sum(weights)
    
    def get_params(self):
        """Returns a list of all nodes that require a gradient"""
        return [n for n in self.node_tensors if n.requires_grad]
    
    def optimize(self, n_frames=200, render=True, gif_path="output/animation.gif"):
        """Optimizes the truss and saves a gif of the process"""

        frame_folder = 'output/frames/'

        optimizer = torch.optim.Adam(params=self.get_params(), lr=0.005, betas=(0.0, 0.0))

        for i in range(n_frames):
            cost = self.get_cost()

            if render:
                plt.figure()
                self.draw()
                plt.savefig(f"{frame_folder}frame_{i}.png")
                display.display(plt.gcf())
                display.clear_output(wait=True)
                plt.close()

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        if render:
            # Compile frames into a GIF
            frames = []
            for i in range(n_frames):
                frames.append(imageio.imread(f"{frame_folder}frame_{i}.png"))
            imageio.mimsave(gif_path, frames, format='GIF', fps=30, loop=0)
        
        return cost