.. _vqnet_dist:

VQNet的分布式计算模块
*********************************************************

分布式计算​​是指通过多台设备（如GPU/CPU节点）协同完成神经网络的训练或推理任务，利用并行处理加速计算并扩展模型规模。
其核心是通过​​分布式接口​​（如MPI、NCCL、gRP）协调设备间的通信与同步

VQNet的分布式计算模块模块使用mpi启动多进程并行计算, 使用nccl进行GPU之间通信。该功能仅在linux操作系统下能够使用。



环境部署
=================================

以下介绍VQNet分别基于CPU、GPU分布式计算所需的Linux系统下环境的部署.该部分必须MPI的支持, 以下介绍MPI的环境部署。

MPI安装
^^^^^^^^^^^^^^^^^^^^^^

MPI为CPU间通信的常用库, **VQNet中CPU的分布式计算功能则基于MPI进行实现**,以下将介绍如何在Linux系统中对MPI进行安装(目前基于CPU的分布式计算功能仅在Linux上实现)。

本软件当前编译依赖的是mpicxx==4.1.2,您可以通过 ``conda`` 或其他方式安装。 

.. code-block::
    
    conda install conda-forge::mpich-mpicxx==4.1.2


此外我们还必须安装 **mpi4py** 库。通过pip install完成mpi4py的安装即可, 若是出现以下类似错误

.. image:: ./images/mpi_bug.png
    :align: center

|

为mpi4py与python版本之间不兼容的问题, 可以通过以下方法解决

.. code-block::

    # 通过下列代码暂存当前python环境的编译器
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld ld.bak && popd

    # 再次安装
    pip install mpi4py

    # 还原
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld.bak ld && popd

NCCL安装
^^^^^^^^^^^^^^^^^^^^^^

NCCL为GPU间通信的常用库, **VQNet中GPU的分布式计算功能则基于NCCL进行实现**,本软件默认在安装时候同时安装NCCL的动态链接库, 一般不需要安装NCCL。
如果要安装NCCL,可以按照以下介绍如何在Linux系统中对NCCL进行安装(目前基于GPU的分布式计算功能仅在Linux上实现).


从github上将NCCL的仓库拉到本地:

.. code-block::

    git clone https://github.com/NVIDIA/nccl.git

进入nccl根目录并编译

.. code-block::
    
    cd nccl
    make -j src.build

如果cuda没有安装到默认的路径即/usr/local/cuda, 则需要定义CUDA的路径, 使用以下代码来编译

.. code-block::

    make src.build CUDA_HOME=<path to cuda install>

并且可以根据BUILDDIR指定安装目录, 指令如下

.. code-block::
    
    make src.build CUDA_HOME=<path to cuda install> BUILDDIR=/usr/local/nccl

安装完成后在.bashrc文件中添加配置

.. code-block::
    
    vim ~/.bashrc

    # 在最下面加入
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/lib
    export PATH=$PATH:/usr/local/nccl/bin

保存后, 执行

.. code-block::
    
    source ~/.bashrc

可以通过nccl-test进行验证

.. code-block::
    
    git clone https://github.com/NVIDIA/nccl-tests.git
    cd nccl-tests
    make -j12 CUDA_HOME=/usr/local/cuda
    ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 1

节点间通信环境部署
^^^^^^^^^^^^^^^^^^^^^^

在多节点上实现分布式计算,首先 **需要保证多节点上mpich环境的一致,python环境一致** ,其次,需要设置 **节点间的免密通信** 。

假设需要设置node0(主节点)、node1、node2三个节点的免密通信。

.. code-block::

    # 在每个节点上执行
    ssh-keygen
    
    # 之后一直回车,在.ssh文件夹下生成一个公钥(id_rsa.pub)一个私钥(id_rsa)
    # 将其另外两个节点的公钥都添加到第一个节点的authorized_keys文件中,
    # 再将第一个节点authorized_keys文件传到另外两个节点便可以实现节点间的免密通信
    # 在子节点node1上执行
    cat ~/.ssh/id_dsa.pub >> node0:~/.ssh/authorized_keys

    # 在子节点node2上执行
    cat ~/.ssh/id_dsa.pub >> node0:~/.ssh/authorized_keys
    
    # 先删除node1、node2中的authorized_keys文件后,在node0上将authorized_keys文件拷贝到另外两个节点上
    scp ~/.ssh/authorized_keys  node1:~/.ssh/authorized_keys
    scp ~/.ssh/authorized_keys  node2:~/.ssh/authorized_keys

    # 保证三个不同节点生成的公钥都在authorized_keys文件中,即可实现节点间的免密通信

可选的, 最好还设置一个共享目录,使得改变共享目录下的文件时,不同节点中文件也会进行更改,预防多节点运行模型时不同节点中的文件不同步的问题。
使用nfs-utils和rpcbind实现共享目录。

.. code-block::

    # 安装软件包
    yum -y install nfs* rpcbind  

    # 编辑主节点上配置文件
    vim /etc/exports  
    /data/mpi *(rw,sync,no_all_squash,no_subtree_check)

    # 主节点上启动服务
    systemctl start rpcbind
    systemctl start nfs

    # 在所有子结点node1,node2上mount要共享的目录
    mount node1:/data/mpi/ /data/mpi
    mount node2:/data/mpi/ /data/mpi

分布式启动
=================================

使用分布式计算接口,通过 ``vqnetrun`` 命令启动, 接下来介绍 ``vqnetrun`` 的各个参数.

n, np
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口中可以通过 ``-n``, ``-np`` 参数控制启动的进程数,执行样例如下:

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("mpi") # init mpi controller
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun -n 2 python test.py
        # vqnetrun -np 2 python test.py

H, hosts
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口中可以通过 ``-H``, ``--hosts`` 指定节点以及进程分配来跨节点执行(在跨节点运行时必须将节点的环境配置成功, 在相同的环境,相同的路径下执行),执行样例如下:

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # init mpi controller
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 -H node0:1,node2:1 python test.py
        # vqnetrun -np 4 --hosts node0:1,node2:1 python test.py


.. _hostfile:

hostfile, f, hostfile
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口中可以通过指定hosts文件来指定节点以及进程分配来跨节点(在跨节点运行时必须将节点的环境配置成功, 在相同的环境,相同的路径下执行), 命令行参数为 ``-hostfile``, ``-f``, ``--hostfile``.

文件内每行的格式必须为:<hostname> slots=<slots> 如；

node0 slots=1

node2 slots=1

执行样例如下

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # init mpi controller
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 -f hosts python test.py
        # vqnetrun -np 4 -hostfile hosts python test.py
        # vqnetrun -np 4 --hostfile hosts python test.py


output-filename
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口中可以通过命令行参数 ``--output-filename`` 来将输出结果保存到指定文件.

执行样例如下

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # init mpi controller
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --output-filename output  python test.py


verbose
^^^^^^^^^^^^^^^^^^^^^^
``vqnetrun`` 接口中可以通过命令行参数 ``--verbose`` 来对节点间的通信进行检测,并额外输出检测结果。

执行样例如下

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # init mpi controller
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --verbose python test.py


start-timeout
^^^^^^^^^^^^^^^^^^^^^^
``vqnetrun`` 接口中可以通过命令行参数 ``--start-timeout`` 来指定超时前执行所有检查并启动进程。默认值为 30 秒。

执行样例如下

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # init mpi controller
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --start-timeout 10 python test.py


h
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口中可以通过该标志, 输出vqnetrun支持的所有参数以及参数的详细介绍。

执行代码如下

    .. code-block::

        # vqnetrun -h


CommController
=================================

.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)

    CommController用于控制在cpu、gpu下数据通信的控制器, 通过设置参数 `backend` 来生成cpu(mpi)、gpu(nccl)的控制器。(目前分布式计算的功能仅支持linux操作系系统下使用)

    :param backend: 用于生成cpu或者gpu的数据通信控制器。
    :param rank: 该参数仅在非pyvqnet后端下有用, 默认值为: None。
    :param world_size: 该参数仅在非pyvqnet后端下有用, 默认值为: None。

    :return:
        CommController 实例。

    Examples::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl") # init nccl controller

        # Comm_OP = CommController("mpi") # init mpi controller

 
    .. py:method:: getRank()
        
        用于获得当前进程的进程号。


        :return: 返回当前进程的进程号。

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # init nccl controller
            
            Comm_OP.getRank()


    .. py:method:: getSize()
    
        用于获得总共启动的进程数。


        :return: 返回总共进程的数量。

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # init nccl controller
            
            Comm_OP.getSize()
            # vqnetrun -n 2 python test.py 
            # 2

 
    .. py:method:: getLocalRank()
        
        用于获得当前机器上的当前进程号。


        :return: 当前机器上的当前进程号。

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # init nccl controller
            
            Comm_OP.getLocalRank()
            # vqnetrun -n 2 python test.py 

 


 
    .. py:method:: barrier()
        
        同步。

        :return: 同步操作。

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.barrier()


    .. py:method:: get_device_num()
        
        用于获得当前节点上的显卡数量, (仅支持gpu下使用)。

        :return: 返回当前节点上显卡数量。
        
        Examples::


            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.get_device_num()
            # python test.py


    .. py:method:: allreduce(tensor, c_op = "avg")
        
        支持对数据作allreduce通信。

        :param tensor: 输入数据.
        :param c_op: 计算方式.

        Examples::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")

            Comm_OP.allreduce(num, "sum")
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py

 
    .. py:method:: reduce(tensor, root = 0, c_op = "avg")
        
        支持对数据作reduce通信。

        :param tensor: 输入数据。
        :param root: 指定数据返回的节点。
        :param c_op: 计算方式。

        Examples::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")
            
            Comm_OP.reduce(num, 1)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py

 
    .. py:method:: broadcast(tensor, root = 0)
        
        将指定进程root上的数据广播到所有进程上。

        :param tensor: 输入数据。
        :param root: 指定的节点。

        Examples::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")
            
            Comm_OP.broadcast(num, 1)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py

 
    .. py:method:: allgather(tensor)
        
        将所有进程上数据allgather到一起。

        :param tensor: 输入数据。

        Examples::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")

            num = Comm_OP.allgather(num)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py


    .. py:method:: send(tensor, dest)
        
        p2p通信接口。

        :param tensor: 输入数据.
        :param dest: 目的进程.

        Examples::

            from pyvqnet.distributed import CommController,get_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            recv = tensor.zeros_like(num)

            if get_rank() == 0:
                Comm_OP.send(num, 1)
            elif get_rank() == 1:
                Comm_OP.recv(recv, 0)
            print(f"rank {Comm_OP.getRank()}  {num}")
            print(f"rank {Comm_OP.getRank()}  {recv}")
            
            # vqnetrun -n 2 python test.py

 
    .. py:method:: recv(tensor, source)
        
        p2p通信接口。

        :param tensor: 输入数据.
        :param source: 接受进程.

        Examples::

            from pyvqnet.distributed import CommController,get_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            recv = tensor.zeros_like(num)

            if get_rank() == 0:
                Comm_OP.send(num, 1)
            elif get_rank() == 1:
                Comm_OP.recv(recv, 0)
            print(f"rank {Comm_OP.getRank()}  {num}")
            print(f"rank {Comm_OP.getRank()}  {recv}")
            
            # vqnetrun -n 2 python test.py

    .. py:method:: split_group(rankL)
        
        根据入参设置的进程号列表用于划分多个通信组。

        :param rankL: 进程组序号列表。

        :return: 当后端为 `nccl` 返回的是进程组序号元组，当后端为 `mpi` 返回一个列表，其长度等于分组个数；每个元素是二元组 (comm, rank)，其中 comm 为该分组的 MPI 通信器，rank 为组内序号。

        Examples::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            groups = Comm_OP.split_group([[0, 1],[2,3]])
            print(groups)
            #[[<mpi4py.MPI.Intracomm object at 0x7f53691f3230>, [0, 3]], [<mpi4py.MPI.Intracomm object at 0x7f53691f3010>, [2, 1]]]

            # mpirun -n 4 python test.py
        
    .. py:method:: allreduce_group(tensor, c_op = "avg", group = None)
        
        组内allreduce通信接口。

        :param tensor: 输入数据.
        :param c_op: 计算方法.
        :param group: 当使用mpi后端时候，输入由 `init_group` 或 `split_group` 生成的组对应通信组，当使用nccl后端时候输入`split_group` 生成的组序号。


        Examples::

            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_group([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"allreduce_group before rank {get_rank()}: {complex_data}")

            Comm_OP.allreduce_group(complex_data, c_op="sum",group = groups[0])
            print(f"allreduce_group after rank {get_rank()}: {complex_data}")
            # vqnetrun -n 2 python test.py

    .. py:method:: reduce_group(tensor, root = 0, c_op = "avg", group = None)
        
        组内reduce通信接口。

        :param tensor: 输入数据.
        :param root: 指定进程号.
        :param c_op: 计算方法.
        :param group: 当使用mpi后端时候，输入由 `init_group` 或 `split_group` 生成的组对应通信组，当使用nccl后端时候输入`split_group` 生成的组序号。


        Examples::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_group([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"reduce_group before rank {get_rank()}: {complex_data}")

            Comm_OP.reduce_group(complex_data, c_op="sum",group = groups[0])
            print(f"reduce_group after rank {get_rank()}: {complex_data}")
            # vqnetrun -n 2 python test.py

 
    .. py:method:: broadcast_group(tensor, root = 0, group = None)
        
        组内broadcast通信接口。

        :param tensor: 输入数据.
        :param root: 指定从哪个进程号广播， 默认为0.
        :param group: 当使用mpi后端时候，输入由 `init_group` 或 `split_group` 生成的组对应通信组，当使用nccl后端时候输入`split_group` 生成的组序号。


        Examples::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_group([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"broadcast_group before rank {get_rank()}: {complex_data}")

            Comm_OP.broadcast_group(complex_data,group = groups[0])
            Comm_OP.barrier()
            print(f"broadcast_group after rank {get_rank()}: {complex_data}")
            # vqnetrun -n 2 python test.py

 
    .. py:method:: allgather_group(tensor, group = None)
        
        组内allgather通信接口。

        :param tensor: 输入数据.
        :param group: 当使用mpi后端时候，输入由 `init_group` 或 `split_group` 生成的组对应通信组，当使用nccl后端时候输入`split_group` 生成的组序号。


        Examples::
            
            from pyvqnet.distributed import CommController,get_rank,init_group
            from pyvqnet.tensor import tensor

            Comm_OP = CommController("mpi")
            group = init_group([[0,1]])
            #mpi init group internally
            # A list of lists, where each sublist contains a communicator and the corresponding rank list.
            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1))
            print(f" before rank {get_rank()}: {complex_data}")
            for comm_ in group:
                if Comm_OP.getRank() in comm_[1]:
                    complex_data = Comm_OP.all_gather_group(complex_data, comm_[0])
                    print(f"after rank {get_rank()}: {complex_data}")
            # mpirun -n 2 python test.py

            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            Comm_OP = CommController("nccl")
            groups = Comm_OP.split_group([[0, 1]])
            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())
            print(f" before rank {get_rank()}: {complex_data}")
            complex_data = Comm_OP.all_gather_group(complex_data, group = groups[0])
            print(f"after rank {get_rank()}: {complex_data}")
            # mpirun -n 2 python test.py


 

split_data
=================================

在多进程中,使用 ``split_data`` 根据进程数对数据进行切分,返回相应进程上数据。

.. py:function:: pyvqnet.distributed.datasplit.split_data(x_train, y_train, shuffle=False)

    设置分布式计算参数。

    :param x_train: `np.array` - 训练数据.
    :param y_train: `np.array` -  训练数据标签.
    :param shuffle: `bool` - 是否打乱后再进行切分,默认值是False.

    :return: 切分后的训练数据和标签。

    Example::

        from pyvqnet.distributed import split_data
        import numpy as np

        x_train = np.random.randint(255, size = (100, 5))
        y_train = np.random.randint(2, size = (100, 1))

        x_train, y_train= split_data(x_train, y_train)

get_local_rank
=================================


.. py:function:: pyvqnet.distributed.ControlComm.get_local_rank()

    得到当前节点上进程号。例如你在第2个节点的第3个进程,每个节点5个进程,则返回2。

    :return: 当前机器上的当前进程号。

    Example::

        from pyvqnet.distributed.ControlComm import get_local_rank

        print(get_local_rank())
        # vqnetrun -n 2 python test.py

get_rank
=================================

.. py:function:: pyvqnet.distributed.ControlComm.get_rank()

    用于获得当前进程的全局进程号。例如你在第2个节点的第3个进程,每个节点5个进程,则返回7。

    :return: 当前进程的进程号。

    Example::

        from pyvqnet.distributed.ControlComm import get_rank

        print(get_rank())
        # vqnetrun -n 2 python test.py

init_group
=================================


.. py:function:: pyvqnet.distributed.ControlComm.init_group(rank_lists)

    根据给出的进程数列表来对基于 `mpi` 后端的进程组进行初始化。

    .. warning::

        该接口只支持分布式后端为 `mpi` 。

    :param rank_lists: 通信进程组列表.
    :return: 返回一个列表，其长度等于分组个数；每个元素是二元组 (comm, rank)，其中 comm 为该分组的 MPI 通信器，rank 为组内序号。

    Example::

        from pyvqnet.distributed import *

        Comm_OP = CommController("mpi")
        num = tensor.to_tensor(np.random.rand(1, 5))
        print(f"rank {Comm_OP.getRank()}  {num}")
        
        group_l = init_group([[0,2], [1]])

        for comm_ in group_l:
            if Comm_OP.getRank() in comm_[1]:
                Comm_OP.allreduce_group(num, "sum", group = comm_[0])
                print(f"rank {Comm_OP.getRank()}  {num} after")
        
        # vqnetrun -n 3 python test.py


PipelineParallelTrainingWrapper
=================================
.. py:class:: pyvqnet.distributed.pp.PipelineParallelTrainingWrapper(args,join_layers,trainset)
    
    Pipeline Parallel Training Wrapper 实现了 1F1B训练。仅在 Linux 平台上,且具有 GPU 的情况下可用。
    更多算法细节可以在(https://www.deepspeed.ai/tutorials/pipeline/)找到。

    :param args: 参数字典。参见示例。
    :param join_layers: Sequential 模块的列表。
    :param trainset: 数据集。

    :return:
        PipelineParallelTrainingWrapper 实例。

    以下使用 CIFAR10数据库 `CIFAR10_Dataset`,在2块GPU上训练AlexNet上的分类任务。
    本例子中分成两个流水线并行进程 `pipeline_parallel_size` = 2。
    批处理大小为 `train_batch_size` = 64, 单GPU 上为 `train_micro_batch_size_per_gpu` = 32。
    其他配置参数可见 `args`。
    此外,每个进程需要在 `__main__` 函数中配置环境变量的 `LOCAL_RANK`。
    
    .. code-block::

        os.environ["LOCAL_RANK"] = str(dist.get_local_rank())

    调用 `train_batch` 进行训练。

    Examples::

        import os
        import pyvqnet

        from pyvqnet.nn import Module,Sequential,CrossEntropyLoss
        from pyvqnet.nn import Linear
        from pyvqnet.nn import Conv2D
        from pyvqnet.nn import activation as F
        from pyvqnet.nn import MaxPool2D
        from pyvqnet.nn import CrossEntropyLoss

        from pyvqnet.tensor import tensor
        from pyvqnet.distributed.pp import PipelineParallelTrainingWrapper
        from pyvqnet.distributed.configs import comm as dist
        from pyvqnet.distributed import *


        pipeline_parallel_size = 2

        num_steps = 1000

        def cifar_trainset_vqnet(local_rank, dl_path='./cifar10-data'):
            transform = pyvqnet.data.TransformCompose([
                pyvqnet.data.TransformResize(256),
                pyvqnet.data.TransformCenterCrop(224),
                pyvqnet.data.TransformToTensor(),
                pyvqnet.data.TransformNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            trainset = pyvqnet.data.CIFAR10_Dataset(root=dl_path,
                                                    mode="train",
                                                    transform=transform,layout="HWC")

            return trainset

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.features = Sequential( 
                Conv2D(input_channels=3, output_channels=8, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),
                MaxPool2D([2, 2], [2, 2]),

                Conv2D(input_channels=8, output_channels=16, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),
                MaxPool2D([2, 2], [2, 2]),

                Conv2D(input_channels=16, output_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),

                Conv2D(input_channels=32, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),

                Conv2D(input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),
                MaxPool2D([3, 3], [2, 2]),)
                
                self.cls = Sequential( 
                Linear(64 * 27 * 27, 512),
                F.ReLu(),

                Linear(512, 256),
                F.ReLu(),
                Linear(256, 10) )

            def forward(self, x):
                x = self.features(x)
                x = tensor.flatten(x,1)
                x = self.cls(x)

                return x
            
        def join_layers(vision_model):
            layers = [
                *vision_model.features,
                lambda x: tensor.flatten(x, 1),
                *vision_model.cls,
            ]
            return layers


        if __name__ == "__main__":


            args = {
            "backend":'nccl',  
            "train_batch_size" : 64,
            "train_micro_batch_size_per_gpu" : 32,
            "epochs":5,
        "optimizer": {
            "type": "Adam",
            "params": {
            "lr": 0.001
            }}, 
            "local_rank":dist.get_local_rank(), 
            "pipeline_parallel_size":pipeline_parallel_size, "seed":42, "steps":num_steps,
            "loss":CrossEntropyLoss(),
            }
            os.environ["LOCAL_RANK"] = str(dist.get_local_rank())
            trainset = cifar_trainset_vqnet(args["local_rank"])
            w = PipelineParallelTrainingWrapper(args,join_layers(Model()),trainset)

            all_loss = {}

            for i in range(args["epochs"]):
                w.train_batch()
                
            all_loss = w.loss_dict


ZeroModelInitial
=================================
.. py:class:: pyvqnet.distributed.ZeroModelInitial(args,model,optimizer)
    
    Zero1 api接口, 目前仅用于linux平台下基于GPU并行计算。

    :param args: 参数字典。参见示例。
    :param model: 输入模型。
    :param optimizer: 优化器。

    :return:
        Zero1 Engine.

    以下使用 MNIST 数据库, 在2块GPU上训练一个MLP模型上的分类任务。

    批处理大小为 `train_batch_size` = 64, `zero_optimization` 的阶段 `stage` 设置为1.
    若Optimizer为None, 则采用 `args` 中 `optimizer` 的设置.
    其他配置参数可见 `args`。
    此外,每个进程需要在配置环境变量的 `LOCAL_RANK`。
    
    .. code-block::

        os.environ["LOCAL_RANK"] = str(dist.get_local_rank())

    Examples::

        from pyvqnet.distributed import *
        from pyvqnet import *
        from time import time
        import pyvqnet.optim as optim
        import pyvqnet.nn as nn
        import pyvqnet
        import sys
        import pyvqnet 
        import numpy as np
        import os
        import struct

        def load_mnist(dataset="training_data",
                    digits=np.arange(2),
                    path="./"):
            """
            load mnist data
            """
            from array import array as pyarray
            if dataset == "training_data":
                fname_image = os.path.join(path, "train-images.idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace(
                    "\\", "/")
            elif dataset == "testing_data":
                fname_image = os.path.join(path, "t10k-images.idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace(
                    "\\", "/")
            else:
                raise ValueError("dataset must be 'training_data' or 'testing_data'")

            flbl = open(fname_label, "rb")
            _, size = struct.unpack(">II", flbl.read(8))

            lbl = pyarray("b", flbl.read())
            flbl.close()

            fimg = open(fname_image, "rb")
            _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = pyarray("B", fimg.read())
            fimg.close()

            ind = [k for k in range(size) if lbl[k] in digits]
            num = len(ind)
            images = np.zeros((num, rows, cols),dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels


        train_images_np, train_labels_np = load_mnist(dataset="training_data", digits=np.arange(10),path="../data/MNIST_data/")
        train_images_np = train_images_np / 255.

        test_images_np, test_labels_np = load_mnist(dataset="testing_data", digits=np.arange(10),path="../data/MNIST_data/")
        test_images_np = test_images_np / 255.

        local_rank = pyvqnet.distributed.get_rank()

        from pyvqnet.distributed import ZeroModelInitial

        class MNISTClassifier(nn.Module):
            
            def __init__(self):
                super(MNISTClassifier, self).__init__()
                self.fc1 = nn.Linear(28*28, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, 64)
                self.fc5 = nn.Linear(64, 10)
                self.ac = nn.activation.ReLu()
                
            def forward(self, x:pyvqnet.QTensor):
                
                x = x.reshape([-1, 28*28])  
                x = self.ac(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                x = self.fc4(x)
                x = self.fc5(x)
                return x
        
        model = MNISTClassifier()

        model.to(local_rank + 1000)
            
        Comm_op = CommController("nccl")
        Comm_op.broadcast_model_params(model, 0)

        batch_size = 64

        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.Adam(model.parameters(), lr=0.001) 

        args_ = {
                "train_batch_size": batch_size, # 等效的总batch
                "optimizer": {
                    "type": "adam",
                    "params": {
                    "lr": 0.001,
                    }
                },
                "zero_optimization": {
                    "stage": 1, 
                }    
            }

        os.environ["LOCAL_RANK"] = str(get_local_rank())
        model = ZeroModelInitial(args=args_, model=model, optimizer=optimizer) 

        def compute_acc(outputs, labels, correct, total):
            predicted = pyvqnet.tensor.argmax(outputs, dim=1, keepdims=True)
            total += labels.size
            correct += pyvqnet.tensor.sums(predicted == labels).item()
            return correct, total

        train_acc = 0
        test_acc = 0
        epochs = 5
        loss = 0
        time1 = time()

        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            step = 0
            
            num_batches = (train_images_np.shape[0] + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                
                data_ = tensor.QTensor(train_images_np[i*batch_size: (i+1) * batch_size,:], dtype = kfloat32)
                labels = tensor.QTensor(train_labels_np[i*batch_size: (i+1) * batch_size,:], dtype = kint64)
                    
                data_ = data_.to(local_rank + 1000)
                labels = labels.to(local_rank + 1000)
                
                outputs = model(data_)
                loss = criterion(labels, outputs)
                
                model.backward(loss) # 基于返回的model做backward、step 
                model.step() 

                correct, total = compute_acc(outputs, labels, correct, total)
                step += 1
                if step % 50 == 0:
                    print(f"Train : rank {get_rank()} Epoch [{epoch+1}/{epochs}], step {step} Loss: {loss.item():.4f} acc {100 * correct / total}")
                    sys.stdout.flush()
                    
            train_acc = 100 * correct / total
            
        time2 = time()
        print(f'Accuracy of the model on the 10000 Train images: {train_acc}% time cost {time2 - time1}')

ColumnParallelLinear
=================================
.. py:class:: pyvqnet.distributed.ColumnParallelLinear(input_size,output_size,weight_initializer,bias_initializer,use_bias,dtype,name,tp_comm)
    
    张量并行计算,列并行线性层
    
    线性层定义为 Y = XA + b。
    其二维并行为 A = [A_1,...,A_p]。

    :param input_size: 矩阵 A 的第一个维度。
    :param output_size: 矩阵 A 的第二个维度。
    :param weight_initializer: `callable` 默认为 `normal`。
    :param bias_initializer: `callable` 默认为0。
    :param use_bias: `bool` - 默认为 True。
    :param dtype: 默认 `None`,使用默认数据类型。
    :param name: 模块名称,默认为“”。
    :param tp_comm:  通讯控制器。


    以下使用 MNIST 数据库, 在2块GPU上训练一个MLP模型上的分类任务。

    使用时与经典的Linear层的使用相似

    多进程使用时基于 `vqnetrun -n 2 python test.py` 的方式进行

    Examples::

        import pyvqnet.distributed
        import pyvqnet.optim as optim
        import pyvqnet.nn as nn
        import pyvqnet
        import sys
        from pyvqnet.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from pyvqnet.distributed import *
        from time import time

        import pyvqnet 
        import numpy as np
        import os
        from pyvqnet import *
        import pytest

        Comm_OP = CommController("nccl")

        import struct
        def load_mnist(dataset="training_data",
                    digits=np.arange(2),
                    path="./"):
            """
            load mnist data
            """
            from array import array as pyarray
            # download_mnist(path)
            if dataset == "training_data":
                fname_image = os.path.join(path, "train-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "train-labels-idx1-ubyte").replace(
                    "\\", "/")
            elif dataset == "testing_data":
                fname_image = os.path.join(path, "t10k-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "t10k-labels-idx1-ubyte").replace(
                    "\\", "/")
            else:
                raise ValueError("dataset must be 'training_data' or 'testing_data'")

            flbl = open(fname_label, "rb")
            _, size = struct.unpack(">II", flbl.read(8))

            lbl = pyarray("b", flbl.read())
            flbl.close()

            fimg = open(fname_image, "rb")
            _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = pyarray("B", fimg.read())
            fimg.close()

            ind = [k for k in range(size) if lbl[k] in digits]
            num = len(ind)
            images = np.zeros((num, rows, cols),dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels

        train_images_np, train_labels_np = load_mnist(dataset="training_data", digits=np.arange(10),path="./data/MNIST/raw/")
        train_images_np = train_images_np / 255.

        test_images_np, test_labels_np = load_mnist(dataset="testing_data", digits=np.arange(10),path="./data/MNIST/raw/")
        test_images_np = test_images_np / 255.

        local_rank = pyvqnet.distributed.get_rank()

        class MNISTClassifier(nn.Module):
            def __init__(self):
                super(MNISTClassifier, self).__init__()
                self.fc1 = RowParallelLinear(28*28, 512, tp_comm = Comm_OP)
                self.fc2 = ColumnParallelLinear(512, 256, tp_comm = Comm_OP)
                self.fc3 = RowParallelLinear(256, 128, tp_comm = Comm_OP)
                self.fc4 = ColumnParallelLinear(128, 64, tp_comm = Comm_OP)
                self.fc5 = RowParallelLinear(64, 10, tp_comm = Comm_OP)  
                self.ac = nn.activation.ReLu()
                
            def forward(self, x:pyvqnet.QTensor):
                
                x = x.reshape([-1, 28*28])  
                x = self.ac(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                x = self.fc4(x)
                x = self.fc5(x)
                return x
            
        
        model = MNISTClassifier()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.to(local_rank + 1000)

        Comm_OP.broadcast_model_params(model, 0)

        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        def compute_acc(outputs, labels, correct, total):
            predicted = pyvqnet.tensor.argmax(outputs, dim=1, keepdims=True)
            total += labels.size
            correct += pyvqnet.tensor.sums(predicted == labels).item()
            return correct, total

        train_acc = 0
        test_acc = 0
        epochs = 5
        loss = 0

        time1 = time()
        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            step = 0
            
            batch_size = 64
            num_batches = (train_images_np.shape[0] + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                data_ = tensor.QTensor(train_images_np[i*batch_size: (i+1) * batch_size,:], dtype = kfloat32)
                labels = tensor.QTensor(train_labels_np[i*batch_size: (i+1) * batch_size,:], dtype = kint64)

                data_ = data_.to(local_rank + 1000)
                labels = labels.to(local_rank + 1000)

                optimizer.zero_grad()

                outputs = model(data_)
                loss = criterion(labels, outputs)

                loss.backward()
                optimizer.step()

                correct, total = compute_acc(outputs, labels, correct, total)
                step += 1
                if step % 50 == 0:
                    print(f"Train : rank {get_rank()} Epoch [{epoch+1}/{epochs}], step {step} Loss: {loss.item():.4f} acc {100 * correct / total}")
                    sys.stdout.flush()

            train_acc = 100 * correct / total
        time2 = time()

        print(f'Accuracy of the model on the 10000 Train images: {train_acc}% time cost {time2 - time1}')


RowParallelLinear
=================================
.. py:class:: pyvqnet.distributed.RowParallelLinear(input_size,output_size,weight_initializer,bias_initializer,use_bias,dtype,name,tp_comm)
    
    张量并行计算,行并行线性层。

    线性层的定义为 Y = XA + b。A 沿其一维并行,X 沿其二维并行。
    A = transpose([A_1 ... A_p]) X = [X_1, ..., X_p]。

    :param input_size: 矩阵 A 的第一个维度。
    :param output_size: 矩阵 A 的第二个维度。
    :param weight_initializer: `callable` 默认为 `normal`。
    :param bias_initializer: `callable` 默认为0。
    :param use_bias: `bool` - 默认为 True。
    :param dtype: 默认 `None`,使用默认数据类型。
    :param name: 模块名称。
    :param tp_comm: 通讯控制器。

    以下使用 MNIST 数据库, 在2块GPU上训练一个MLP模型上的分类任务。
    使用时与经典的Linear层的使用相似

    多进程使用时基于 `vqnetrun -n 2 python test.py` 的方式进行

    Examples::

        import pyvqnet.distributed
        import pyvqnet.optim as optim
        import pyvqnet.nn as nn
        import pyvqnet
        import sys
        from pyvqnet.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from pyvqnet.distributed import *
        from time import time

        import pyvqnet 
        import numpy as np
        import os
        from pyvqnet import *
        import pytest

        Comm_OP = CommController("nccl")

        import struct
        def load_mnist(dataset="training_data",
                    digits=np.arange(2),
                    path="./"):
            """
            load mnist data
            """
            from array import array as pyarray
            # download_mnist(path)
            if dataset == "training_data":
                fname_image = os.path.join(path, "train-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "train-labels-idx1-ubyte").replace(
                    "\\", "/")
            elif dataset == "testing_data":
                fname_image = os.path.join(path, "t10k-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "t10k-labels-idx1-ubyte").replace(
                    "\\", "/")
            else:
                raise ValueError("dataset must be 'training_data' or 'testing_data'")

            flbl = open(fname_label, "rb")
            _, size = struct.unpack(">II", flbl.read(8))

            lbl = pyarray("b", flbl.read())
            flbl.close()

            fimg = open(fname_image, "rb")
            _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = pyarray("B", fimg.read())
            fimg.close()

            ind = [k for k in range(size) if lbl[k] in digits]
            num = len(ind)
            images = np.zeros((num, rows, cols),dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels

        train_images_np, train_labels_np = load_mnist(dataset="training_data", digits=np.arange(10),path="./data/MNIST/raw/")
        train_images_np = train_images_np / 255.

        test_images_np, test_labels_np = load_mnist(dataset="testing_data", digits=np.arange(10),path="./data/MNIST/raw/")
        test_images_np = test_images_np / 255.

        local_rank = pyvqnet.distributed.get_rank()

        class MNISTClassifier(nn.Module):
            def __init__(self):
                super(MNISTClassifier, self).__init__()
                self.fc1 = RowParallelLinear(28*28, 512, tp_comm = Comm_OP)
                self.fc2 = ColumnParallelLinear(512, 256, tp_comm = Comm_OP)
                self.fc3 = RowParallelLinear(256, 128, tp_comm = Comm_OP)
                self.fc4 = ColumnParallelLinear(128, 64, tp_comm = Comm_OP)
                self.fc5 = RowParallelLinear(64, 10, tp_comm = Comm_OP)  
                self.ac = nn.activation.ReLu()
                
                
            def forward(self, x:pyvqnet.QTensor):
                
                x = x.reshape([-1, 28*28])  
                x = self.ac(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                x = self.fc4(x)
                x = self.fc5(x)
                return x
            
        model = MNISTClassifier()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.to(local_rank + 1000)
        Comm_OP.broadcast_model_params(model, 0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        def compute_acc(outputs, labels, correct, total):
            predicted = pyvqnet.tensor.argmax(outputs, dim=1, keepdims=True)
            total += labels.size
            correct += pyvqnet.tensor.sums(predicted == labels).item()
            return correct, total

        train_acc = 0
        test_acc = 0
        epochs = 5
        loss = 0

        time1 = time()
        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            step = 0
            
            batch_size = 64
            num_batches = (train_images_np.shape[0] + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                data_ = tensor.QTensor(train_images_np[i*batch_size: (i+1) * batch_size,:], dtype = kfloat32)
                labels = tensor.QTensor(train_labels_np[i*batch_size: (i+1) * batch_size,:], dtype = kint64)

                data_ = data_.to(local_rank + 1000)
                labels = labels.to(local_rank + 1000)

                optimizer.zero_grad()

                outputs = model(data_)
                loss = criterion(labels, outputs)

                loss.backward()
                optimizer.step()

                correct, total = compute_acc(outputs, labels, correct, total)
                step += 1
                if step % 50 == 0:
                    print(f"Train : rank {get_rank()} Epoch [{epoch+1}/{epochs}], step {step} Loss: {loss.item():.4f} acc {100 * correct / total}")
                    sys.stdout.flush()

            train_acc = 100 * correct / total
        time2 = time()

        print(f'Accuracy of the model on the 10000 Train images: {train_acc}% time cost {time2 - time1}')

比特重排序
=================================

量子比特重排序技术是比特并行中的技术，其核心是通过改变比特并行过程中量子逻辑门的排列顺序，减少比特并行中需要执行比特变换的次数，以下是基于比特并行构建大比特量子线路时需要的模块。参照论文 `Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation <https://export.arxiv.org/abs/2410.04252>`__ 。
以下接口需要通过 `mpi` 启动多个进程进行计算。

DistributeQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistributeQMachine(num_wires,dtype,grad_mode)

    用于比特并行中的变分量子计算的模拟类，包含每个节点包含的部分比特上的量子态。通过MPI,每个节点都申请一个该类，进行分布式的量子变分线路模拟，N的值必须等于2的分布式并行的比特个数 `global_qubit` 的幂次方，可通过 `set_qr_config` 进行配置。

    :param num_wires: 完整量子线路的比特的数量。
    :param dtype: 计算数据的数据类型。默认是 pyvqnet.kcomplex64，对应的参数精度是 pyvqnet.kfloat32。
    :param grad_mode: 在 ``DistQuantumLayerAdjoint`` 进行反传时设置为 adjoint。

    .. note::

        输入的比特数是整个量子线路所需要的比特数量，通过DistributeQMachine会根据全局比特数构建量子模拟器, 其比特数量为 ``nums_wires - global_qubit``，
        反传必须基于 ``DistQuantumLayerAdjoint``。

    .. warning::

        该接口只支持在Linux下运行；

        必须对 ``DistributeQMachine`` 中比特并行中参数进行配置， 如样例中所示，包括：
        
        .. code-block::

            qm.set_just_defined(True)
            qm.set_save_op_history_flag(True) # open save op
            qm.set_qr_config({'qubit': 总比特个数, 'global_qubit': 分布式比特个数})

    Examples::

        from pyvqnet.distributed import get_rank
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import rx, ry, cnot, MeasureAll,rz
        import pyvqnet
        from pyvqnet.distributed.qubits_reorder import DistributeQMachine,DistQuantumLayerAdjoint
        pyvqnet.utils.set_random_seed(123)


        qubit = 10
        batch_size = 5

        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = DistributeQMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
                
                self.qm.set_just_defined(True)
                self.qm.set_save_op_history_flag(True) # open save op
                self.qm.set_qr_config({"qubit": num_wires, # open qubit reordered, set config
                                        "global_qubit": 1}) # global_qubit == log2(nproc)
                
                self.params = pyvqnet.nn.Parameter([qubit])

                self.measure = MeasureAll(obs={
                    "X5":1.0
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                for i in range(qubit):
                    rx(q_machine=self.qm, params=self.params[i], wires=i)
                    ry(q_machine=self.qm, params=self.params[3], wires=i)
                    rz(q_machine=self.qm, params=self.params[4], wires=i)
                cnot(q_machine=self.qm,  wires=[0, 1])
                rlt = self.measure(q_machine=self.qm)
                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]], requires_grad=True).toGPU(pyvqnet.DEV_GPU_0+get_rank())

        input_x = tensor.broadcast_to(input_x, [2, 3])

        input_x.requires_grad = True

        quantum_model = QModel(num_wires=qubit,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint").toGPU(pyvqnet.DEV_GPU_0+get_rank())

        adjoint_model = DistQuantumLayerAdjoint(quantum_model)
        adjoint_model.train()

        batch_y = adjoint_model(input_x)
        batch_y.backward()

        print(batch_y)
        # mpirun -n 2 python test.py


DistQuantumLayerAdjoint
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistQuantumLayerAdjoint(vqc_module,name)

    使用伴随矩阵方式对比特并行计算中的参数进行梯度计算的DistQuantumLayer层

    :param vqc_module: 输入的蕴含 ``DistributeQMachine`` 模块。
    :param name: 模块名称。

    .. note::

        输入的vqc_module模块必须包含 ``DistributeQMachine``， 基于 ``DistributeQMachine`` 进行比特并行下的adjoint反传梯度计算。

    .. warning::

        该接口只支持在Linux下运行；
        
    Examples::

        from pyvqnet.distributed import get_rank
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import rx, ry, cnot, MeasureAll,rz
        import pyvqnet
        from pyvqnet.distributed.qubits_reorder import DistributeQMachine,DistQuantumLayerAdjoint
        pyvqnet.utils.set_random_seed(123)


        qubit = 10
        batch_size = 5

        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = DistributeQMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
                
                self.qm.set_just_defined(True)
                self.qm.set_save_op_history_flag(True) # open save op
                self.qm.set_qr_config({"qubit": num_wires, # open qubit reordered, set config
                                            "global_qubit": 1}) # global_qubit == log2(nproc)
                
                self.params = pyvqnet.nn.Parameter([qubit])

                self.measure = MeasureAll(obs={
                    "X5":1.0
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                for i in range(qubit):
                    rx(q_machine=self.qm, params=self.params[i], wires=i)
                    ry(q_machine=self.qm, params=self.params[3], wires=i)
                    rz(q_machine=self.qm, params=self.params[4], wires=i)
                cnot(q_machine=self.qm,  wires=[0, 1])
                rlt = self.measure(q_machine=self.qm)
                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]], requires_grad=True).toGPU(pyvqnet.DEV_GPU_0+get_rank())

        input_x = tensor.broadcast_to(input_x, [2, 3])

        input_x.requires_grad = True

        quantum_model = QModel(num_wires=qubit,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint").toGPU(pyvqnet.DEV_GPU_0+get_rank())

        adjoint_model = DistQuantumLayerAdjoint(quantum_model)
        adjoint_model.train()

        batch_y = adjoint_model(input_x)
        batch_y.backward()

        print(batch_y)
        # mpirun -n 2 python test.py