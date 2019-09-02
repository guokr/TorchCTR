Visulization
=============

TorchCTR has a simple but enough dashboard for log and metrics visulization. This dashboard should be setup as a server.

.. literalinclude:: ../../examples/server.py
   :caption: Simple dashboard server ``examples/server.py``
   :name: server.py


You'll get the prompt like this:

.. parsed-literal::

   * Serving Flask app "torchctr.dashboard.dashboard" (lazy loading)
   * Environment: production
     WARNING: Do not use the development server in a production environment.
     Use a production WSGI server instead.
   * Debug mode: off
   * Running on http://0.0.0.0:8080/ (Press CTRL+C to quit)

Back to the training part, and tell trainer the server address:

.. code-block:: python

   trainer = Trainer(model, dataset)
   trainer.train(dashboard_address="localhost:8080")

Then trainer will send all the metrics to the server for visulization, sure you can check them in your browser.
