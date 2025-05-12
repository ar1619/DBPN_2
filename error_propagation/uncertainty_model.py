import numpy as np
import os
import xarray as xr
import random
def main(num_samples, folder_path, list_of_sites, size=32):
    """
    Generate low-resolution data for testing.
    
    Args:
        num_samples (int): Number of low-resolution samples to generate.
        low_res_shape (int): Shape of the low-resolution data.
        
    Returns:
        Nothing. The function saves the low-resolution data to files.
    """
    list_low_res = os.listdir(folder_path)
    for item in list_of_sites:
        # take random num_samples samples from list_low_res that start with item
        list_item = [x for x in list_low_res if x.startswith(item)]
        list_item = random.sample(list_item, num_samples)
        for item_list in list_item:
            data = np.load(os.path.join(folder_path, item_list))
            low_res_data = data[:32, :]
            # save low_res_data to file
            np.save(f'Input/uncertainty/sites/'+item_list, low_res_data)
            print(f'Saved low_res_data to Input/uncertainty/sites/'+item_list)

list_of_sp = ['br', 'bu', 'ci', 'db', 'df', 'et', 'eu', 'gm', 'hf',
                 'iz', 'jf', 'js', 'ka', 'll', 'lr', 'ni', 'or', 'pa',
                 'pr', 'ra', 'rj', 'so', 'sp', 'wg']
folder_path = '../RDS/OCO-2/centered_arrays/'
main(100, folder_path, list_of_sp, 32)