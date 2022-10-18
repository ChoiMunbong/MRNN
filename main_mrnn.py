"""
Main function for MRNN
Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using 
           Multi-Directional Recurrent Neural Networks," 
           in IEEE Transactions on Biomedical Engineering, 
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
--------------------------------------------------
(1) Load the data
(2) Train MRNN model
(3) Impute missing data
(4) Evaluate the imputation performance
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings

import impyute
import pandas as pd
import tensorflow as tf
from lstm import rnn
import sklearn.preprocessing
from impyute.imputation.cs import mice, mean, fast_knn
from impyute.imputation.ts import moving_window
warnings.filterwarnings("ignore")
import numpy as np
import shutil
import os

from data_loader import data_loader
from mrnn import mrnn
from utils import imputation_performance


def main(args):
  """MRNN main function.
  
  Args:
    - file_name: dataset file name
    - seq_len: sequence length of time-series data
    - missing_rate: the rate of introduced missingness
    - h_dim: hidden state dimensions
    - batch_size: the number of samples in mini batch
    - iteration: the number of iteration
    - learning_rate: learning rate of model training
    - metric_name: imputation performance metric (mse, mae, rmse)
    
  Returns:
    - output:
      - x: original data with missing
      - ori_x: original data without missing
      - m: mask matrix
      - t: time matrix
      - imputed_x: imputed data
      - performance: imputation performance
  """  
  
  ## Load data
  x, m, t, ori_x, scaler_sklearn = data_loader(args.file_name,
                               args.seq_len, 
                               args.missing_rate)
  ## Train M-RNN
  # Remove 'tmp/mrnn_imputation' directory if exist
  if os.path.exists('tmp/mrnn_imputation'):
    shutil.rmtree('tmp/mrnn_imputation')
  
  # mrnn model parameters
  model_parameters = {'h_dim': args.h_dim,
                      'batch_size': args.batch_size,
                      'iteration': args.iteration, 
                      'learning_rate': args.learning_rate}

  lstm = rnn(x, model_parameters)
  lstm.fit(x,m,t)
  imputed_x_lstm = lstm.transform(x,m,t)
  performance_matrix = ('mae', 'mse', 'rmse')
  for metrix_name in performance_matrix :
    performance = imputation_performance(ori_x, imputed_x_lstm, metrix_name)
    print(f"{metrix_name} : {performance}")
  performance = 1
  output2 = {'x': x, 'ori_x': ori_x, 'm': m, 't': t, 'imputed_x': imputed_x_lstm,
            'performance': performance}

  # # Fit mrnn_model
  # mrnn_model = mrnn(x, model_parameters)
  # mrnn_model.fit(x, m, t)
  # # Impute missing data
  # imputed_x = mrnn_model.transform(x, m, t)
  #
  #
  # # Evaluate the imputation performance
  # performance = imputation_performance (ori_x, imputed_x, args.metric_name)
  #
  # # Report the result
  # print(args.metric_name + ': ' + str(np.round(performance, 4)))
  #
  # # Return the output
  # output = {'x': x, 'ori_x': ori_x, 'm': m, 't': t, 'imputed_x': imputed_x,
  #           'performance': performance}

  if os.path.exists('tmp/mrnn_imputation'):
    shutil.rmtree('tmp/mrnn_imputation')

  return output2, scaler_sklearn


##
if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--file_name',
      default='data/test_case2.csv',
      # default='data/test_case.csv',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length of time-series data',
      default=24,
      type=int)
  parser.add_argument(
      '--missing_rate',
      help='the rate of introduced missingness',
      default=0.2,
      type=float)
  parser.add_argument(
      '--h_dim',
      help='hidden state dimensions',
      default=10,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini batch',
      default=96*7,
      type=int)
  parser.add_argument(
      '--iteration',
      help='the number of iteration',
      default=2000,
      type=int)
  parser.add_argument(
      '--learning_rate',
      help='learning rate of model training',
      default=0.01,
      type=float)
  parser.add_argument(
      '--metric_name',
      help='imputation performance metric',
      default='mae',
      type=str)

  args = parser.parse_args()


  # Call main function
  output, scaler_sklearn = main(args)


  print(output)
  # print(output2)
  print(output['imputed_x'])

  col = ['YMS, HM, Used']

  lis = np.array([0,0,0])

  print(lis)

  for i in range(len(output['imputed_x'])) :
      if i == len(output['imputed_x']) - 1:
          for j in range(args.seq_len):
            lis = np.concatenate([lis, output['imputed_x'][i][j]])
      # print(output['ori_x'][i][0])
      else :
        lis = np.concatenate([lis, output['imputed_x'][i][0]])



  lis = lis.reshape(-1, 3)
  lis = lis[:len(lis)-2]
  lis = scaler_sklearn.inverse_transform(lis)
  lis = np.flipud(lis)
  print(lis)
  np.savetxt("result.csv", lis, delimiter=",", header='YMS, HM, Used', fmt='%d, %f, %f', comments='', encoding='cp949')


  # df.to_csv('result.csv', ',', index=False)
  # df = pd.DataFrame(output, encodings = 'cp949')
  # print(df['imputed_x'])


   ###############################^using mRNN###############################################
  lis2 = np.loadtxt('result.csv', delimiter=',', skiprows=1)
  for i in range(round(len(lis2)*0.5)) :
      lis2[np.random.randint(len(lis2)), 2] = np.nan

  print(lis2)
  temp_df = lis2

  imputed_mice = max(temp_df)
  performance = imputation_performance(ori_x=lis, imputed_x=imputed_mice, metric_name= args.metric_name)

  # Report the result

  print(args.metric_name + ': ' + str(np.round(performance, 4)))
