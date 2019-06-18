import pandas as pd
import numpy as np

import io, os, sys
import glob

class data_clean():
    
    """
    centered the point clouds and make a total train/test set
    """
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with the ModelNet.
            max_num (int): largest number of point clouds 

        """
        self.root_dir = root_dir
#         self.max_num = max_num
        
        
        

    def find_max_num(self):
        
        global max_num
        
        num_points = []
#         class_list =  sorted(os.listdir(self.root_dir))


        for i in range(len(class_list)):

            train_files = sorted(glob.glob(self.root_dir + class_list[i] + '/train/*.off'))
            print("findin_max_num in train ", class_list[i])
            
            for filename in train_files:
                    data_set = pd.read_csv(filename, skiprows = [0,1],
                            header = None, names = ['x', 'y', 'z', 'c'], sep = " ")
                    data_set = data_set[data_set.isnull().any(axis = 1)]
                    data_set = data_set.iloc[:, :-1]
                    

                    n_pointers = len(data_set)
                    num_points.append(n_pointers)

        for i in range(len(class_list)):
            
            test_files = sorted(glob.glob(self.root_dir + class_list[i] + '/test/*.off'))
            
            print("findin_max_num in test ", class_list[i])

            for filename in test_files:
                    data_set = pd.read_csv(filename, skiprows = [0,1],
                            header = None, names = ['x', 'y', 'z', 'c'], sep = " ")
                    data_set = data_set[data_set.isnull().any(axis = 1)]
                    data_set = data_set.iloc[:, :-1]
                    
                    n_pointers = len(data_set)
                    num_points.append(n_pointers)
       
    
        num_points_np = np.array(num_points)        
        max_num = int(np.mean(num_points_np) + 2 * np.std(num_points_np))

        
        return max_num
        
        

    def centered_point_clouds(self, files):

        train_set = np.array([])
                
        for filename in files:
            data_set = pd.read_csv(filename, skiprows = [0,1],
                    header = None, names = ['x', 'y', 'z', 'c'], sep = " ")
            data_set = data_set[data_set.isnull().any(axis = 1)]
            data_set = data_set.iloc[:, :-1]
            data_set['class'] = i
            data_set = data_set.to_numpy() 

            max_min = (np.abs(np.max(data_set, axis = 0)) - np.abs(np.min(data_set, axis = 0))) / 2


            Canonical_ST = np.zeros([1,4])

            Canonical_ST[0,0] = max_min[0]
            Canonical_ST[0,1] = max_min[1]
            Canonical_ST[0,2] = max_min[2]

            centered_point_clouds = data_set - Canonical_ST
            
            df_repeated = pd.DataFrame(centered_point_clouds)
            

            centered_point_clouds = self.match_up(df_repeated)
            
            
            if len(centered_point_clouds) != max_num:
                print(len(centered_point_clouds))
            
            train_set = np.append(train_set, centered_point_clouds)
            
        train_set = train_set.reshape(-1, max_num, 4)
        
        
        return train_set
    
    
    def match_up(self, centered_point_clouds_df):
                
        while len(centered_point_clouds_df) < max_num:
            
            centered_point_clouds_df = pd.concat([centered_point_clouds_df] * 50, ignore_index=True)
            
        
        centered_point_clouds = centered_point_clouds_df.iloc[:max_num, :].to_numpy()
        
        return centered_point_clouds
    
    
    
    def total_set(self):
        
        global class_list
        global i 

        total_train_set = np.array([])
        total_test_set = np.array([])
        
        class_list =  sorted(os.listdir(self.root_dir))


#         class_list = os.listdir(self.root_dir) # to find how many classes are, of course 40.....
        max_num  = self.find_max_num()


        for i in range(len(class_list)):
            
            print("working on class: ",  class_list[i])
            train_files = glob.glob(self.root_dir + class_list[i] + '/train/*.off')
            test_files = glob.glob(self.root_dir + class_list[i] + '/test/*.off')
            

            train = self.centered_point_clouds(train_files)
            test = self.centered_point_clouds(test_files)
            
            total_train_set = np.append(total_train_set, train)      
            total_test_set = np.append(total_test_set, test)
        
        total_train_set = total_train_set.reshape(-1, max_num, 4)
        total_test_set = total_test_set.reshape(-1, max_num, 4)
        
        return total_train_set, total_test_set

if __name__ == "__main__":
    data = data_clean('./ModelNet10/')
    total_train_set, total_test_set = data.total_set()
    np.save('total_train_set.npy', total_train_set)
    np.save('total_test_set.npy', total_test_set)