# import tools
from ..graph import tools

import matplotlib 
matplotlib.use('Agg') 

class Graph:
    def __init__(self, labeling_mode='spatial', layout='MCFS-22'):

        self.get_edge(layout)
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_edge(self, layout):
        if layout == 'MCFS-22' or layout == 'MCFS-130':
            self.num_node = 25
            self.self_link = [(i, i) for i in range(self.num_node)]
            inward_ori_index = [(2,9), (1,2), (16,1), (18,16), (17,1), (19,17), (6,2),
                                (7,6), (8,7), (3,2), (4,3), (5,4), (10,9),
                                (11, 10), (12, 11), (25, 12), (23, 12), (24, 23), (13,9),
                                (14, 13), (15, 14), (22, 15), (20, 15), (21, 20)]
            self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            self.outward  = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        elif layout == 'PKU-subject' or layout == 'PKU-view':
            self.num_node = 25
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.inward = [(12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), 
                 (18, 17), (19, 18), (1, 0), (20, 1), (2, 20), (3, 2), (4,20),
                 (5,4), (6,5), (7,6), (21,7), (22,6), (8,20), (9,8), (10, 9),
                 (11,10), (24,10), (23,11)]
            self.outward  = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        elif layout == 'LARA':
            self.num_node = 19
            self.self_link  = [(i, i) for i in range(self.num_node)]
            self.inward  = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9), (11, 9), (12,10), (13,12), (14,13), (15,9), (16,15), (17,16), (18,17)]
            self.outward  = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()


    fig = plt.figure()
    ax = fig.add_subplot(111)

    # im = ax.imshow((A[0]+A[1]+A[2])*(A[0]+A[1]+A[2])*(A[0]+A[1]+A[2])*(A[0]+A[1]+A[2]), cmap=plt.cm.hot_r)
    im = ax.imshow(A[1], cmap=plt.cm.hot_r)

    plt.colorbar(im)
    #show
    plt.show()
    # for i in A:
    #     plt.imshow(i, cmap='gray')
    #     print(i)
    #     plt.show()
    # plt.savefig("./10.jpg")
    # # print(A)


