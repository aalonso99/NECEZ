from sklearn.neighbors import KNeighborsRegressor
import torch
import numpy as np
import collections
import pickle

### TODO: 
# - The queue must be modified so it allows to upgrade the priority of an element after querying it

class DND(object):
    def __init__(self,
    			 vector_dim,
    			 k=50,
                 max_size = 1000,
                 kdtree_rebuild = 10,
                 leaf_size=30,
                 memory_object=None,
                 n_jobs=-1):
        
        # Dimension of the latent vectors stored in the memory
        self.vector_dim = vector_dim
        
        # Number of neighbors to use in Q value regression
        self.k = k
        self.neighbors = None
        
        # Maximum number of elements to be stored in the dictionary
        self.max_size = max_size
        
        # Number used to create new indices for the dictionary elements
        # IT IS NOT THE NUMBER OF ELEMENTS CURRENTLY IN MEMORY
        self.counter = 0
        
        # Table of the dictionary. Its items have the form (i : (representation, q_value)), where:
        #	i is the reference to each value recorded
        #	representation is the state observed, stored as a latent vector
        #	q_value is the Q-value associated to such observation
        self.memory_table = {}

        # Object in charge of the operations with memory. It is a object of the Memory class in
        # memory.py. Used to save/read the raw observations in/from disk.
        self.memory_object = memory_object
        
        # Priority Queue that manages the elements that should be discarded next
        self.priority_queue = collections.deque()
        
        # Index used for the KNN algorithm
        self.Q_regressor = KNeighborsRegressor(n_neighbors=k, 
        									   weights='distance', 
        									   algorithm='kd_tree', 
        									   leaf_size=leaf_size, 
        									   p=2, 
        									   metric='minkowski', 
        									   metric_params=None, 
        									   n_jobs=n_jobs)
									     
        # The index needs to be retrained to appropietaly quantize the vector space
        # These variables allow to know when to do the retraining
        self.kdtree_rebuild = kdtree_rebuild
        self.kdtree_rebuild_counter = 0
        self.never_built = True
        self.available = False


    def save(self, path):
        with open(path,'wb') as f:
            pickle.dump((self.memory_table,self.priority_queue,self.max_size), f)


    def load(self, path):
        with open(path,'rb') as f:
            self.memory_table, self.priority_queue, self.max_size = pickle.load(f)
		
        if len(self.memory_table) > 0:
            self.rebuild_kdtree()


    def rebuild_kdtree(self):
		
        representations = []
        q_values = []
        for r,q in self.memory_table.values():
            representations.append(r)
            q_values.append(q)

        self.Q_regressor.fit( np.array(representations), np.array(q_values) )
        
        self.never_built = False
        self.kdtree_rebuild_counter = 0
        
        if len(self.memory_table) >= self.k:
            self.available = True
            

    def add(self, representations, q_values, observations=None, memory_object=None):
        assert len(representations) == len(q_values)
        if observations:
            assert len(representations) == len(observations)
        #assert len(representations.shape) == 2

        new_ids = np.arange(self.counter, self.counter+len(q_values))
        self.counter += len(q_values)
        
        for i, r, q in zip(new_ids, representations, q_values):
            self.memory_table[i] = (r, q)
            self.priority_queue.append(i)   # Adds element to the right of the priority queue

        if observations:
            if memory_object is not None:
                memory_object.save_observations.remote(observations, new_ids)
            elif self.memory_object is not None:
                self.mememory_objectmory.save_observations.remote(observations, new_ids)
            else:
                raise Exception("There are observations to save in memory, but no memory object provided to DND")

        current_memory_size = len(self.memory_table)
        if current_memory_size > self.max_size:
            old_ids = []
            for _ in range(current_memory_size - self.max_size):
                old_ids.append(self.priority_queue.popleft())
                
            for old_id in old_ids:
                del self.memory_table[old_id]

            if memory_object is not None:
                memory_object.delete_observations.remote(old_ids)
            elif self.memory_object is not None:
                self.memory_object.delete_observations.remote(old_ids)
            else:
                raise Exception("There are observations to remove from memory, but no memory object provided to DND")

            # Removing elements from the kdtree is also a reason to rebuild it
            self.kdtree_rebuild_counter += len(old_ids)
        
        self.kdtree_rebuild_counter += len(representations)
        
        if self.never_built or self.kdtree_rebuild_counter >= self.kdtree_rebuild:
            self.rebuild_kdtree()
        	

    def query_knn(self, representations, k=None):

        if k == None:
            n_neighbours = self.k
        else:
            n_neighbours = k

        # print("Calculando KNN. k={}".format(n_neighbours))

        # Indices returned by the kneighbours method are in the range [0, n_neighbours] and do not 
        # correspond with our references in the memory
        dists, indices = self.Q_regressor.kneighbors(representations, n_neighbors=n_neighbours)
        
        # Transform the indices into our dictionary keys assuming they are indexed in the 
        # tree in the same ordered as the data passed to the tree building function
        indices = np.asarray(list(self.memory_table.keys()))[indices]

        # When an element in the memory is queried its priority is reset so only unused values
        # are removed when the memory is full
        for i in indices.flatten():
            self.priority_queue.remove(i)
            self.priority_queue.append(i)

        self.neighbors = indices
        
        return dists, indices
        
    def query_q_value(self, representations):
        Q_values = self.Q_regressor.predict(representations)
        return Q_values

if __name__ == "__main__":
    pass
