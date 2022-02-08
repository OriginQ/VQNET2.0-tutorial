Net API
========================

.. currentmodule:: pyvqnet.static.vqnet

Net function
---------------

Model
^^^^^^^^^^^^

.. autofunction:: Model


build
^^^^^^^^^^^^

.. autofunction:: build


save
^^^^^^^^^^^^

.. autofunction:: save


load
^^^^^^^^^^^^

.. autofunction:: load

.. note::
    you should write the code of layer construction and use ``vqnet.build(net)`` before ``load()``


summary
^^^^^^^^^^^^

.. autofunction:: summary


fit 
^^^^^^^^^^^^

.. autofunction:: fit


evaluate
^^^^^^^^^^^^

.. autofunction:: evaluate


predict
^^^^^^^^^^^^
.. autofunction:: predict

setLayerTrainable
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: setLayerTrainable

reset_loss
^^^^^^^^^^^^
.. autofunction:: reset_loss

train_batch
^^^^^^^^^^^^
.. autofunction:: train_batch

eval_batch
^^^^^^^^^^^^
.. autofunction:: eval_batch

get_losses
^^^^^^^^^^^^
.. autofunction:: get_losses

get_metrics
^^^^^^^^^^^^^
.. autofunction:: get_metrics