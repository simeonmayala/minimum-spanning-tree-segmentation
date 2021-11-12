

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import shortest_path, dijkstra
from scipy.sparse.csgraph import breadth_first_tree
from scipy.sparse import csr_matrix
from sknetwork.path import shortest_path
from skimage import color
import numpy as np
from skimage.filters import gaussian

def prepare_image(image_original, sigma):
   image_original = image_original.astype('uint8')
   input_image = image_original[:,:,:]
   img_grayscale = color.rgb2gray(input_image)
   image = gaussian(img_grayscale, sigma = sigma)
   return input_image, image

def mst_construction(image):
    segments = np.arange(np.prod(image.shape), dtype=np.int).reshape(image.shape)
    
    # compute edges weights
    down_cost = np.abs((image[1:]-image[:-1]).ravel())
    right_cost = np.abs((image[:,1:]-image[:,:-1]).ravel())
    costs = np.concatenate((down_cost, right_cost)) + 1
    
    #Compute edges
    row_down  = segments[:-1].ravel()
    col_down  = segments[1: ].ravel()
    row_right  = segments[:,:-1].ravel()
    col_right  = segments[:,1: ].ravel()
    row = np.concatenate((row_down, row_right))
    col = np.concatenate((col_down, col_right))
    
    # construct a graph
    create_sparse_gr = csr_matrix((costs, (row, col)), shape = (np.prod(segments.shape), np.prod(segments.shape)))
    # construct a minimum spanning tree
    mst = minimum_spanning_tree(create_sparse_gr)
    mst = mst + mst.T
    return mst

def BFS_path(mst, node1, node2):
    BFS_tree = breadth_first_tree(mst, node1, directed=False)
    path = shortest_path(BFS_tree, sources = node1, targets=node2)
    return path

def split_components(mst, node1, node2):
    
    path_finder = np.array(BFS_path(mst, node1, node2))
    if len(path_finder)==0:
        return connected_components(mst)[1]
    sources = path_finder[:-1]
    targets = path_finder[1:]
    longest_ind_edge = np.argmax(mst[sources].T[targets].diagonal())
    n1 = sources[longest_ind_edge]
    n2 = targets[longest_ind_edge]

    mst[n1,n2]=0
    mst[n2,n1]=0
    mst.eliminate_zeros()
 
    return connected_components(mst)[1]

def nodes_image(image):
    img_grayscale = color.rgb2gray(image)
    height, width =  img_grayscale.shape
    template_image =  np.arange(width*height, dtype=np.intp).reshape(height, width)
    return template_image

def segmented_and_labels(new_labels, image):
    new_labels_reshaped = new_labels.reshape(image.shape)
    visualization_image = image.copy()
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if new_labels_reshaped[row, col] ==0:
                visualization_image[row, col] = 0
                
    return visualization_image, new_labels_reshaped
