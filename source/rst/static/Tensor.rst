Tensor module
=============

.. currentmodule:: pyvqnet.static.tensor
.. autoclass:: Tensor

Creation Methods
------------------------------


eye
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.eye

fromarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.fromarray


full
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.full


linspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.linspace

logspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.logspace


ones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.ones

randn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.randn

randu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.randu


range
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.range


zeros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.zeros


Math Methods
-----------------------------


abs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.abs


acos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.acos


add
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.add


asin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.asin


atan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.atan


ceil
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.ceil


clamp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.clamp


clampmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.clampmax


clampmin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.clampmin


cos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.cos


cosh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.cosh


div
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.div


exp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.exp


floor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.floor


logn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.logn

max
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.max

min
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.min

mod
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.mod


mult
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.mult

mult2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.mult2D


neg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.neg



normalize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.normalize


reciprocal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.reciprocal


round
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.round

rsqrt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. automethod:: Tensor.rsqrt


sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.sigmoid

sign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. automethod:: Tensor.sign


sin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.sin


sinh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.sinh


sqr
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.sqr


sqrt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.sqrt

sub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. automethod:: Tensor.sub


tan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.tan


tanh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.tanh


trunc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.trunc

Logical operations
-----------------------------

logical_and
^^^^^^^^^^^^^^

.. automethod:: Tensor.logical_and

logical_or
^^^^^^^^^^^^^^

.. automethod:: Tensor.logical_or

logical_not
^^^^^^^^^^^^^^

.. automethod:: Tensor.logical_not

logical_xor
^^^^^^^^^^^^^^

.. automethod:: Tensor.logical_xor


Array contents
-----------------------------

isfinite
^^^^^^^^^^^^^^

.. automethod:: Tensor.isfinite

isinf
^^^^^^^^^^^^^^

.. automethod:: Tensor.isinf

isnan
^^^^^^^^^^^^^^

.. automethod:: Tensor.isnan

isneginf
^^^^^^^^^^^^^^

.. automethod:: Tensor.isneginf

isposinf
^^^^^^^^^^^^^^

.. automethod:: Tensor.isposinf


Comparison
-----------------------------

greater
^^^^^^^^^^^^^^

.. automethod:: Tensor.greater

greater_equal
^^^^^^^^^^^^^^

.. automethod:: Tensor.greater_equal

less
^^^^^^^^^^^^^^

.. automethod:: Tensor.less

less_equal
^^^^^^^^^^^^^^

.. automethod:: Tensor.less_equal

equal
^^^^^^^^^^^^^^

.. automethod:: Tensor.equal

not_equal
^^^^^^^^^^^^^^

.. automethod:: Tensor.not_equal


nonzero
^^^^^^^^^^^^^^

.. automethod:: Tensor.nonzero


Truth value testing
-----------------------------

all
^^^^^^^^^^^^^^

.. automethod:: Tensor.all

any
^^^^^^^^^^^^^^

.. automethod:: Tensor.any

Reductions
-----------------------------

argmax
^^^^^^^^^^^^^^

.. automethod:: Tensor.argmax

argmin
^^^^^^^^^^^^^^

.. automethod:: Tensor.argmin

median
^^^^^^^^^^^^^^

.. automethod:: Tensor.median

prod
^^^^^^^^^^^^^^

.. automethod:: Tensor.prod

std
^^^^^^^^^^^^^^

.. automethod:: Tensor.std

sum
^^^^^^^^^^^^^^

.. automethod:: Tensor.sum

var
^^^^^^^^^^^^^^

.. automethod:: Tensor.var


Sorting
-----------------------------


sort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.sort



argsort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.argsort

Value operations
-----------------------------

fill_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.fill_


fill_rand_uniform_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.fill_rand_uniform_


fill_rand_signed_uniform_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.fill_rand_signed_uniform_



fill_rand_normal_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.fill_rand_normal_


Changing array shape
-----------------------------


flatten
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.flatten


reshape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.reshape



Changing number of dimensions
-----------------------------


squeeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.squeeze

unsqueeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.unsqueeze


Transpose-like operations
-----------------------------


permute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.permute

swapaxis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.swapaxis


Multiple Tensors manipulation
------------------------------


concat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.concat


stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.stack


tile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.tile



Other Methods
-----------------------------



diag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.diag


getdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.getdata

getShape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.getShape


info
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.info


print
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.print



trace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Tensor.trace

