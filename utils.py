"""Utility functions for MRNN.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using 
           Multi-Directional Recurrent Neural Networks," 
           in IEEE Transactions on Biomedical Engineering, 
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
--------------------------------------------------
(1) MinMaxScaler
(2) Imputation performance
"""

# Necessary packages
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def MinMaxScaler1(data):
  """Normalization tool: Min Max Scaler.
  
  Args:
    - data: raw input data
    
  Returns:
    - normalized_data: minmax normalized data
    - norm_parameters: normalization parameters for rescaling if needed
  """  
  min_val = np.min(data, axis = 0)
  data = data - min_val
  max_val = np.max(data, axis = 0) + 1e-8
  # max_val = max_val - min_val
  normalized_data = data / max_val
  # normalized_data = normalized_data * (max_val - min_val) + min_val
  
  norm_parameters = {'min_val': min_val, 'max_val': max_val}

  return normalized_data, norm_parameters


def imputation_performance (ori_x, imputed_x, metric_name):
  """Performance metrics for imputation.
  
  Args:
    - ori_x: original complete data (without missing values)
    - imputed_x: imputed data from incomplete data
    - m: observation indicator
    - metric_name: mae, mse, or rmse
    
  Returns:
    - performance: imputation performance in terms or mae, mse, or rmse
  """
  
  assert metric_name in ['mae','mse','rmse']
  # ori_x[:, np.newaxis]
  # imputed_x = np.expand_dims(imputed_x, axis=0) #

  if len(ori_x.shape) >= 3 :
    no, seq_len, dim = ori_x.shape
    ori_x = np.reshape(ori_x, [no * seq_len, dim])
    imputed_x = np.reshape(imputed_x, [no * seq_len, dim])

  # Only compute the imputation performance if m = 0 (missing)
  if metric_name == 'mae':
    performance = mean_absolute_error(ori_x, imputed_x)
  elif metric_name == 'mse':
    performance = mean_squared_error(ori_x, imputed_x)
  elif metric_name == 'rmse':
    performance = np.sqrt(mean_squared_error(ori_x, imputed_x))
    
  return performance