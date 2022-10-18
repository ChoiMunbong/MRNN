"""MRNN core functions.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using 
           Multi-Directional Recurrent Neural Networks," 
           in IEEE Transactions on Biomedical Engineering, 
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
---------------------------------------------------
(1) Train RNN part
(2) Test RNN part
(3) Train FC part
(4) Test FC part
"""
sclar = None
# Necessary Packages
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
from model_utils import biGRUCell, initial_point_interpolation

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

class rnn ():
  """MRNN class with core functions.
  
  Attributes:
    - x: incomplete data
    - model_parameters:
      - h_dim: hidden state dimensions
      - batch_size: the number of samples in mini-batch
      - iteration: the number of iteration
      - learning_rate: learning rate of model training
  """      
  
  def __init__(self, x, model_parameters):
      
    # Set Parameters
    self.no, self.seq_len, self.dim = x.shape

    self.h_dim = model_parameters['h_dim']   
    self.batch_size = model_parameters['batch_size']  
    self.iteration = model_parameters['iteration']     
    self.learning_rate = model_parameters['learning_rate'] 
        
    
  def rnn_train (self, x, m, t, f):  
    """Train RNN for each feature.
    
    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix
      - f: feature index
    """
    tf.compat.v1.reset_default_graph()
     
    with tf.compat.v1.Session() as sess:        
        
      # input place holders
      target = tf.compat.v1.placeholder(tf.float32, [self.seq_len, None, 1])
      mask = tf.compat.v1.placeholder(tf.float32, [self.seq_len, None, 1])
                
      # Build rnn object
      rnn = biGRUCell(3, self.h_dim, 1)    
      outputs = rnn.get_outputs()
      loss = tf.sqrt(tf.reduce_mean(tf.square(mask*outputs - mask*target)))
      optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
      train = optimizer.minimize(loss)  
      
      sess.run(tf.compat.v1.global_variables_initializer())    
      
      # Training
      for i in range(self.iteration):  
        # Batch selection
        batch_idx = np.random.permutation(x.shape[0])[:self.batch_size]
        
        temp_input = np.dstack((x[:,:,f], m[:,:,f], t[:,:,f]))            
        temp_input_reverse = np.flip(temp_input, 1)
              
        forward_input = np.zeros([self.batch_size, self.seq_len, 3]) 
        forward_input[:,1:,:] = temp_input[batch_idx,
                                           :(self.seq_len-1), :] 
              
        backward_input = np.zeros([self.batch_size, self.seq_len, 3]) 
        backward_input[:,1:,:] = temp_input_reverse[batch_idx,
                                                    :(self.seq_len-1),:] 
              
        _, step_loss = \
        sess.run([train, loss], 
                 feed_dict=
                 {mask: np.transpose(np.dstack(m[batch_idx,:,f]),[1, 2, 0]), 
                  target: np.transpose(np.dstack(x[batch_idx,:,f]),[1, 2, 0]),
                  rnn._inputs: forward_input, 
                  rnn._inputs_rev: backward_input})        
                      
      # Save model
      inputs = {'forward_input': rnn._inputs, 
                'backward_input': rnn._inputs_rev}
      outputs = {'imputation': outputs}
        
      save_file_name = 'tmp/mrnn_imputation/rnn_feature_' + str(f+1) + '/'
      tf.compat.v1.saved_model.simple_save(sess, save_file_name, 
                                           inputs, outputs)

      save_file_name2 = 'model/mrnn_imputation/rnn_feature_' + str(f + 1) + '/'
      tf.compat.v1.saved_model.simple_save(sess, save_file_name2,
                                           inputs, outputs)


  def rnn_predict (self, x, m, t):
    """Impute missing data using RNN block.
    
    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix
    
    Returns:
      - imputed_x: imputed data by rnn block
    """
    # Output Initialization
    imputed_x = np.zeros([self.no, self.seq_len, self.dim])
      
    # For each feature
    for f in range(self.dim):      
      
      temp_input = np.dstack((x[:,:,f], m[:,:,f], t[:,:,f]))            
      temp_input_reverse = np.flip(temp_input, 1)
            
      forward_input = np.zeros([self.no, self.seq_len, 3]) 
      forward_input[:,1:,:] = temp_input[:,:(self.seq_len-1), :] 
            
      backward_input = np.zeros([self.no, self.seq_len, 3]) 
      backward_input[:,1:,:] = temp_input_reverse[:,:(self.seq_len-1),:] 
            
      save_file_name = 'tmp/mrnn_imputation/rnn_feature_' + str(f+1)  + '/'
      
      # Load saved model
      graph = tf.Graph()

      with graph.as_default():
        with tf.compat.v1.Session() as sess:          
          sess.run(tf.compat.v1.global_variables_initializer())
          tf.compat.v1.saved_model.loader.load(sess, 
                                               [tf.compat.v1.saved_model.SERVING], 
                                               save_file_name)
          fw_input = graph.get_tensor_by_name('inputs:0')
          bw_input = graph.get_tensor_by_name('inputs_rev:0')
          output = graph.get_tensor_by_name('map/TensorArrayStack/TensorArrayGatherV3:0')
      
          imputed_data = sess.run(output, 
                                  feed_dict={fw_input: forward_input, 
                                             bw_input: backward_input})
        
          imputed_x[:, :, f] = (1-m[:,:,f]) * np.transpose(np.squeeze(imputed_data)) + \
                               m[:,:,f] * x[:,:,f]
      
    # Initial poitn interpolation for better performance
    imputed_x = initial_point_interpolation (x, m, t, imputed_x)
                    
    return imputed_x

  def fit(self, x, m, t):
    """Train the entire MRNN.
    
    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix
    """
    # Train RNN part
    for f in range(self.dim):
      self.rnn_train(x, m, t, f)
      print('Finish ' + str(f+1) + '-th feature training with RNN for imputation')
    # Train FC part  
    print('Finish M-RNN training with both RNN and FC for imputation')
    

  def transform(self, x, m, t):
    """Impute missing data using the entire MRNN.
    
    Args:
      - x: incomplete data
      - m: mask matrix
      - t: time matrix
      
    Returns:
      - imputed_x: imputed data
    """
    # Impute with both RNN and FC part
    imputed_x = self.rnn_predict(x, m, t)
    
    return imputed_x