QTensor 模块
==============

VQNet量子机器学习所使用的数据结构QTensor的python接口文档。QTensor支持常用的多维矩阵的操作包括创建函数，数学函数，逻辑函数，矩阵变换等。


.. currentmodule:: pyvqnet.tensor.tensor
.. autoclass:: QTensor


QTensor's 函数与属性
----------------------------------


__init__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.__init__


ndim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: QTensor.ndim

shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: QTensor.shape

size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: QTensor.size


zero_grad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.zero_grad


backward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.backward

to_numpy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.to_numpy

item
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.item

argmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.argmax


argmin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.argmin

fill\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.fill_


all
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.all

any
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.any


fill_rand_binary\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.fill_rand_binary_

fill_rand_signed_uniform\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.fill_rand_signed_uniform_

fill_rand_uniform\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.fill_rand_uniform_

fill_rand_normal\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.fill_rand_normal_

QTensor.transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.transpose


transpose\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.transpose_


QTensor.reshape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.reshape


reshape\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.reshape_


getdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: QTensor.getdata



创建函数
-----------------------------


ones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ones

ones_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ones_like

full
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: full

full_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: full_like


zeros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: zeros

zeros_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: zeros_like

arange
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: arange

linspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: linspace

logspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: logspace

eye
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: eye

diag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: diag

randu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: randu

randn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: randn


数学函数
-----------------------------


floor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: floor

ceil
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ceil

round
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: round

sort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sort

argsort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: argsort

add
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: add

sub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sub

mul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: mul

divide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: divide

sums
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sums

mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: mean

median
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: median

std
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: std

var
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: var

matmul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: matmul

reciprocal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: reciprocal

sign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sign

neg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: neg

trace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: trace

exp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: exp

acos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: acos

asin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: asin

atan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: atan

sin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sin

cos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: cos

tan 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: tan

tanh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: tanh

sinh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sinh

cosh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: cosh

power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: power

abs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: abs

log
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: log

sqrt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqrt

square
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: square

逻辑函数
--------------------------

maximum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maximum

minimum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: minimum

min
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: min

max
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: max

clip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: clip

where
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: where

nonzero
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nonzero

isfinite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isfinite

isinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isinf

isnan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isnan

isneginf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isneginf

isposinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isposinf

logical_and
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: logical_and

logical_or
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: logical_or

logical_not
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: logical_not

logical_xor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: logical_xor

greater
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: greater

greater_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: greater_equal

less
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: less

less_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: less_equal

equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: equal

not_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: not_equal

矩阵操作
--------------------------

select
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: select

concatenate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: concatenate

stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: stack

permute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: permute

transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: transpose

tile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: tile

squeeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: squeeze

unsqueeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: unsqueeze

swapaxis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: swapaxis

flatten
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: flatten


实用函数
-----------------------------


to_tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: to_tensor