Getting Started
===============


Basic usage
-----------

TorchCTR has a set of built-in datasets and models.


Quick example
~~~~~~~~~~~~~~


.. literalinclude:: ../../examples/basic_usage.py
   :caption: From file ``examples/basic_usage.py``
   :name: basic_usage.py


The results should be:

.. parsed-literal::

   | Warning | Didn't specify the func for dense field, so we will use default log
   | Warning | Didn't specify the func for target column, so we will use raw data
   | WARNING | embed_dim should be specified, otherwise we'll use default value 4
   | building parameters ...
   | building trainer ...
   | Didn't find dashboard
   | Start training ...
   | Training 1/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 67.87it/s, auc=0.567, loss=0.946]
   | Validating: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 27.05it/s, auc=0.410, loss=1.53]
   | Training 2/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 84.69it/s, auc=0.725, loss=0.703]
   | Validating: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 22.27it/s, auc=0.490, loss=1.11]
   | Training 3/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 76.40it/s, auc=0.813, loss=0.521]
   | Validating: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 28.83it/s, auc=0.602, loss=0.873]
   | Training 4/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 78.02it/s, auc=0.875, loss=0.412]
   | Validating: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 27.46it/s, auc=0.685, loss=0.766]
   | Training 5/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 83.62it/s, auc=0.875, loss=0.388]
   | Validating: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 26.10it/s, auc=0.721, loss=0.73]
   | Training 6/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 88.59it/s, auc=0.867, loss=0.397]
   | Validating: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 28.79it/s, auc=0.750, loss=0.694]
   | Training 7/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 81.10it/s, auc=0.875, loss=0.386]
   | Validating: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 25.55it/s, auc=0.772, loss=0.638]
   | Training 8/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 77.11it/s, auc=0.871, loss=0.371]
   | Validating: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 24.10it/s, auc=0.782, loss=0.59]
   | Training 9/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 79.48it/s, auc=0.879, loss=0.368]
   | Validating: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 27.65it/s, auc=0.791, loss=0.562]
   | Training 10/10: 100%|██████████████████████████████████████████████| 7/7 [00:00<00:00, 83.20it/s, auc=0.883, loss=0.37]
   | Validating: 100%|█████████████████████████████████████████████████| 1/1 [00:00<00:00, 27.10it/s, auc=0.789, loss=0.552]
   | Didn't find dir, so we will create it

the method will offer to download the Titanic data if it has not already been downloaded, and it'll be saved in `.torchctr` folder in your home directory.

Build model with custom trainer hyper-parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basicly there are two hyper-parameters setting we can customirize, the first one is model hyper-parameter, such as the vector dimension in Factorization Machine model, the neural network's hidden dimension in Google Wide & Deep model, which shoule be set with model's introduction. The second one is the trainer hyper-parameter, which could be set with your own environment.

.. code-block:: python

   hyper_parameters = {
      "batch_size": 128,
      "device": "cpu",
      "learning_rate": 0.01, 
      "weight_decay": 1e-6,
      "epochs": 30,
      "metrics": ["auc", "acc"]
      }
