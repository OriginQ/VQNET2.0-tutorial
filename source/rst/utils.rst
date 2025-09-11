实用函数
######################



随机种子生成
*******************************************

set_random_seed
==============================

.. py:function:: pyvqnet.utils.set_random_seed(seed)

    设定全局随机种子。

    :param seed: 随机数种子。

    .. note::
            当指定固定随机数种子时,随机分布将依据随机种子产生固定的伪随机分布。
            影响包括: `tensor.randu` , `tensor.randn` ,含参经典神经网络以及量子计算层的参数初始化。

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet.utils import get_random_seed, set_random_seed

        set_random_seed(256)


        rn = tensor.randn([2, 3])
        print(rn)
        rn = tensor.randn([2, 3])
        print(rn)
        rn = tensor.randu([2, 3])
        print(rn)
        rn = tensor.randu([2, 3])
        print(rn)

        print("########################################################")
        from pyvqnet.nn.parameter import Parameter
        from pyvqnet.utils.initializer import he_normal, he_uniform, xavier_normal, xavier_uniform, uniform, quantum_uniform, normal
        print(Parameter(shape=[2, 3], initializer=he_normal))
        print(Parameter(shape=[2, 3], initializer=he_uniform))
        print(Parameter(shape=[2, 3], initializer=xavier_normal))
        print(Parameter(shape=[2, 3], initializer=xavier_uniform))
        print(Parameter(shape=[2, 3], initializer=uniform))
        print(Parameter(shape=[2, 3], initializer=quantum_uniform))
        print(Parameter(shape=[2, 3], initializer=normal))
        # [
        # [-1.2093765, 1.1265280, 0.0796480],
        #  [0.2420146, 1.2623813, 0.2844022]
        # ]
        # [
        # [-1.2093765, 1.1265280, 0.0796480],
        #  [0.2420146, 1.2623813, 0.2844022]
        # ]
        # [
        # [0.3151870, 0.6721524, 0.0416874],
        #  [0.8232620, 0.6537889, 0.9672953]
        # ]
        # [
        # [0.3151870, 0.6721524, 0.0416874],
        #  [0.8232620, 0.6537889, 0.9672953]
        # ]
        # ########################################################
        # [
        # [-0.9874518, 0.9198063, 0.0650323],
        #  [0.1976041, 1.0307300, 0.2322134]
        # ]
        # [
        # [-0.2134037, 0.1987845, -0.5292138],
        #  [0.3732708, 0.1775801, 0.5395861]
        # ]
        # [
        # [-0.7648768, 0.7124789, 0.0503738],
        #  [0.1530635, 0.7984000, 0.1798717]
        # ]
        # [
        # [-0.4049051, 0.3771670, -1.0041126],
        #  [0.7082316, 0.3369346, 1.0237927]
        # ]
        # [
        # [0.3151870, 0.6721524, 0.0416874],
        #  [0.8232620, 0.6537889, 0.9672953]
        # ]
        # [
        # [1.9803783, 4.2232580, 0.2619299],
        #  [5.1727076, 4.1078768, 6.0776958]
        # ]
        # [
        # [-1.2093765, 1.1265280, 0.0796480],
        #  [0.2420146, 1.2623813, 0.2844022]
        # ]

get_random_seed
==============================

.. py:function:: pyvqnet.utils.get_random_seed()

    获取当前随机数种子。

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet.utils import get_random_seed, set_random_seed

        set_random_seed(256)
        print(get_random_seed())
        #256


