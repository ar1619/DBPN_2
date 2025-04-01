import numpy as np
import os
import xarray as xr

def main(num_samples, size=32):
    """
    Generate low-resolution data for testing.
    
    Args:
        num_samples (int): Number of low-resolution samples to generate.
        low_res_shape (int): Shape of the low-resolution data.
        
    Returns:
        Nothing. The function saves the low-resolution data to files.
    """
    list_low_res = os.listdir('../RDS/OCO-2/low_res/')
    # randomly extract num_samples from the list
    list_low_res_samples = np.random.choice(list_low_res, num_samples, replace=False)
    for i in range(num_samples):
        file = xr.open_dataset('../RDS/OCO-2/low_res/'+list_low_res_samples[i])
        data = file['XCO2'].values[0,...]
        # randomly extract subarray of shape (32,32)
        x = np.random.randint(0, data.shape[0]-32)
        y = np.random.randint(0, data.shape[1]-32)
        low_res_data = data[x:x+32, y:y+32]
        # save low_res_data to file
        np.save(f'Input/uncertainty/sample_{i}.npy', low_res_data)
        print(f'Saved low_res_data to Input/uncertainty/sample_{i}.npy')

main(1000, 32)