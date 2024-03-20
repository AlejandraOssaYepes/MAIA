from minisom import MiniSom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D

def PlotSom(data, t, epochs):

    som = MiniSom(10, 10, data.shape[1], sigma=1.5, learning_rate=.7, activation_distance='euclidean',
              topology='hexagonal', neighborhood_function='gaussian', random_seed=10)

    som.train(data, epochs, verbose=True)

    xx, yy = som.get_euclidean_coordinates()
    umatrix = som.distance_map()
    weights = som.get_weights()

    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)

    ax.set_aspect('equal')

    # iteratively add hexagons
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            hex = RegularPolygon((xx[(i, j)], wy), 
                                numVertices=6, 
                                radius=.95 / np.sqrt(3),
                                facecolor=cm.Blues(umatrix[i, j]), 
                                alpha=.5, 
                                edgecolor='gray')
            ax.add_patch(hex)

    markers = [ '^',  '.',  'o', '+']

    colors = ['C0', 'C1', 'C2', 'C3']
    for cnt, x in enumerate(data):
        w = som.winner(x)
        wx, wy = som.convert_map_to_euclidean(w) 
        wy = wy * np.sqrt(3) / 2
        plt.plot(wx, wy, 
                markers[t[cnt]-1], 
                markerfacecolor='None',
                markeredgecolor=colors[t[cnt]-1], 
                markersize=15, 
                markeredgewidth=2)



    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, 
                                orientation='vertical', alpha=.4)

    cb1.ax.get_yaxis().labelpad = 12
    plt.gcf().add_axes(ax_cb)

    legend_elements = [Line2D([0], [0], marker='^', color='C0', label='ODS 3',
                    markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=1),
                    Line2D([0], [0], marker='.', color='C1', label='ODS 4',
                    markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=1),
                    Line2D([0], [0], marker='o', color='C2', label='ODS 5',
                    markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=1),
                    Line2D([0], [0], marker='+', color='C3', label='ODS 16',
                    markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=1)]

    ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.11), loc='upper left', 
            borderaxespad=0., ncol=4, fontsize=12)

    plt.show()
