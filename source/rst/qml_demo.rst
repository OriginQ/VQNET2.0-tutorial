量子机器学习示例
#################################

我们这里使用VQNet实现了多个量子机器学习示例。

带参量子线路在分类任务的应用
*******************************

1. QVC示例
=========================

这个例子使用VQNet实现了论文 `Circuit-centric quantum classifiers <https://arxiv.org/pdf/1804.00633.pdf>`_ 中可变量子线路进行二分类任务。
该例子用来判断一个二进制数是奇数还是偶数。通过将二进制数编码到量子比特上，通过优化线路中的可变参数，使得该线路z方向测量值可以指示该输入为奇数还是偶数。

量子线路
-------------------
变分量子线路通常定义一个子线路，这是一种基本的电路架构，可以通过重复层构建复杂变分电路。
我们的电路层由多个旋转逻辑门以及将每个量子位与其相邻的量子位纠缠在一起的 ``CNOT`` 逻辑门组成。
我们还需要一个线路将经典数据编码到量子态上，使得线路测量的输出与输入有关联。
本例中，我们把二进制输入编码到对应顺序的量子比特上。例如输入数据1101被编码到4个量子比特。

.. math::

    x = 0101 \rightarrow|\psi\rangle=|0101\rangle

.. image:: ./images/qvc_circuit.png
   :width: 600 px
   :align: center

|

.. code-block::

    import pyqpanda as pq

    def qvc_circuits(input,weights,qlist,clist,machine):

        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits)-1):
                cir.insert(pq.CNOT(nqubits[i],nqubits[i+1]))
            cir.insert(pq.CNOT(nqubits[len(nqubits)-1],nqubits[0]))
            return cir

        def build_circult(weights, xx, nqubits):
            
            def Rot(weights_j, qubits):
                circult = pq.QCircuit()
                circult.insert(pq.RZ(qubits, weights_j[0]))
                circult.insert(pq.RY(qubits, weights_j[1]))
                circult.insert(pq.RZ(qubits, weights_j[2]))
                return circult
            def basisstate():
                circult = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i] == 1:
                        circult.insert(pq.X(nqubits[i]))
                return circult

            circult = pq.QCircuit()
            circult.insert(basisstate())

            for i in range(weights.shape[0]):
                
                weights_i = weights[i,:,:]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circult.insert(Rot(weights_j,nqubits[j]))
                cnots = get_cnot(nqubits)  
                circult.insert(cnots) 

            circult.insert(pq.Z(nqubits[0]))
            
            prog = pq.QProg() 
            prog.insert(circult)
            return prog

        weights = weights.reshape([2,4,3])
        prog = build_circult(weights,input,qlist)  
        prob = machine.prob_run_dict(prog, qlist[0], -1)
        prob = list(prob.values())
        
        return prob

模型构建
-------------------
我们已经定义了可变量子线路 ``qvc_circuits`` 。我们希望将其用于我们VQNet的自动微分逻辑中，并使用VQNet的优化算法进行模型训练。我们定义了一个 Model 类，该类继承于抽象类 ``Module``。
Model中使用 :ref:`QuantumLayer` 类这个可进行自动微分的量子计算层。``qvc_circuits`` 为我们希望运行的量子线路，24 为所有需要训练的量子线路参数的个数，"cpu" 表示这里使用 pyQPanda 的 全振幅模拟器，4表示需要申请4个量子比特。
在 ``forward()`` 函数中，用户定义了模型前向运行的逻辑。

.. code-block::

    from pyvqnet.nn.module import Module
    from pyvqnet.optim.sgd import SGD
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.data import data_generator as dataloader
    import pyqpanda as pq
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyqpanda import *
    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits,24,"cpu",4)

        def forward(self, x):
            return self.qvc(x)


模型训练和测试
----------------------
我们使用预先生成的随机二进制数以及其奇数偶数标签。其中数据如下：

.. code-block::

    import numpy as np
    import os
    qvc_train_data = [0,1,0,0,1,
    0, 1, 0, 1, 0,
    0, 1, 1, 0, 0,
    0, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 1, 0, 0,
    1, 0, 1, 1, 1,
    1, 1, 0, 0, 0,
    1, 1, 0, 1, 1,
    1, 1, 1, 0, 1,
    1, 1, 1, 1, 0]
    qvc_test_data= [0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    0, 0, 1, 0, 1,
    0, 0, 1, 1, 0]

    def get_data(dataset_str):
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)
            
        else:
            datasets = np.array(qvc_test_data)
            
        datasets = datasets.reshape([-1,5])
        data = datasets[:,:-1]
        label = datasets[:,-1].astype(int)
        label = np.eye(2)[label].reshape(-1,2)
        return data, label

接着就可以按照一般神经网络训练的模式进行模型前传，损失函数计算，反向运算，优化器运算，直到迭代次数达到预设值。
其所使用的训练数据是上述生成的qvc_train_data，测试数据为qvc_test_data。

.. code-block::

    def get_accuary(result,label):
        result,label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
        return score

    #示例化Model类
    model = Model()
    #定义优化器，此处需要传入model.parameters()表示模型中所有待训练参数，lr为学习率
    optimizer = SGD(model.parameters(),lr =0.1)
    #训练时候可以修改批处理的样本数
    batch_size = 3
    #训练最大迭代次数
    epoch = 20
    #模型损失函数
    loss = CategoricalCrossEntropy()

    model.train()
    datas,labels = get_data("train")

    for i in range(epoch):
        count=0
        sum_loss = 0
        accuary = 0
        t = 0
        for data,label in dataloader(datas,labels,batch_size,False):
            optimizer.zero_grad()
            data,label = QTensor(data), QTensor(label)

            result = model(data)

            loss_b = loss(label,result)
            loss_b.backward()
            optimizer._step()
            sum_loss += loss_b.item()
            count+=batch_size
            accuary += get_accuary(result,label)
            t = t + 1

        print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}")

    model.eval()
    count = 0
    test_data,test_label = get_data("test")
    test_batch_size = 1
    accuary = 0
    sum_loss = 0
    for testd,testl in dataloader(test_data,test_label,test_batch_size):
        testd = QTensor(testd)
        test_result = model(testd)
        test_loss = loss(testl,test_result)
        sum_loss += test_loss
        count+=test_batch_size
        accuary += get_accuary(test_result,testl)
    print(f"test:--------------->loss:{sum_loss/count} #####accuray:{accuary/count}")

.. code-block::

    epoch:0, #### loss:0.20194714764753977 #####accuray:0.6666666666666666
    epoch:1, #### loss:0.19724808633327484 #####accuray:0.8333333333333334
    epoch:2, #### loss:0.19266503552595773 #####accuray:1.0
    epoch:3, #### loss:0.18812804917494455 #####accuray:1.0
    epoch:4, #### loss:0.1835678368806839 #####accuray:1.0
    epoch:5, #### loss:0.1789149840672811 #####accuray:1.0
    epoch:6, #### loss:0.17410411685705185 #####accuray:1.0
    epoch:7, #### loss:0.16908332953850427 #####accuray:1.0
    epoch:8, #### loss:0.16382796317338943 #####accuray:1.0
    epoch:9, #### loss:0.15835540741682053 #####accuray:1.0
    epoch:10, #### loss:0.15273457020521164 #####accuray:1.0
    epoch:11, #### loss:0.14708336691061655 #####accuray:1.0
    epoch:12, #### loss:0.14155150949954987 #####accuray:1.0
    epoch:13, #### loss:0.1362930883963903 #####accuray:1.0
    epoch:14, #### loss:0.1314386005202929 #####accuray:1.0
    epoch:15, #### loss:0.12707658857107162 #####accuray:1.0
    epoch:16, #### loss:0.123248390853405 #####accuray:1.0
    epoch:17, #### loss:0.11995399743318558 #####accuray:1.0
    epoch:18, #### loss:0.1171633576353391 #####accuray:1.0
    epoch:19, #### loss:0.11482855677604675 #####accuray:1.0
    [0.3412148654]
    test:--------------->loss:QTensor(0.3412148654, requires_grad=True) #####accuray:1.0

模型在测试数据上准确率变化情况：

.. image:: ./images/qvc_accuracy.png
   :width: 600 px
   :align: center

|

2. Data Re-uploading模型
==================================
在神经网络中，每一个神经元都接受来自上层所有神经元的信息（图a）。与之相对的，单比特量子分类器接受上一个的信息处理单元和输入（图b）。
通俗地来说，对于传统的量子线路来说，当数据上传完成，可以直接通过若干幺正变换 :math:`U(\theta_1,\theta_2,\theta_3)` 直接得到结果。
但是在量子数据重上传（Quantum Data Re-upLoading，QDRL）任务中，数据在幺正变换之前需要进行重新上传操作。

                                            .. centered:: QDRL与经典神经网络原理图对比

.. image:: ./images/qdrl.png
   :width: 600 px
   :align: center

|

.. code-block::

    """
    Parameterized quantum circuit for Quantum Data Re-upLoading
    """

    import sys
    sys.path.insert(0, "../")
    import numpy as np
    from pyvqnet.nn.linear import Linear
    from pyvqnet.qnn.qdrl.vqnet_model import vmodel
    from pyvqnet.optim import sgd
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.nn.module import Module
    import matplotlib.pyplot as plt
    import matplotlib
    from pyvqnet.data import data_generator as get_minibatch_data
    try:
        matplotlib.use("TkAgg")
    except:  #pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    num_layers = 3
    params = np.random.uniform(size=(num_layers, 3))


    class Model(Module):
        def __init__(self):

            super(Model, self).__init__()
            self.pqc = vmodel(params.shape)
            self.fc2 = Linear(2, 2)

        def forward(self, x):
            x = self.pqc(x)
            return x


    def circle(samples: int, reps=np.sqrt(1 / 2)):
        data_x, data_y = [], []
        for _ in range(samples):
            x = np.random.rand(2)
            y = [0, 1]
            if np.linalg.norm(x) < reps:
                y = [1, 0]
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)


    def plot_data(x, y, fig=None, ax=None):

        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        reds = y == 0
        blues = y == 1
        ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
        ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")


    def get_score(pred, label):
        pred, label = np.array(pred.data), np.array(label.data)
        score = np.sum(np.argmax(pred, axis=1) == np.argmax(label, 1))
        return score


    model = Model()
    optimizer = sgd.SGD(model.parameters(), lr=1)


    def train():
        """
        Main function for train qdrl model
        """
        batch_size = 5
        model.train()
        x_train, y_train = circle(500)
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))  # 500*3

        epoch = 10
        print("start training...........")
        for i in range(epoch):
            accuracy = 0
            count = 0
            loss = 0
            for data, label in get_minibatch_data(x_train, y_train, batch_size):
                optimizer.zero_grad()

                data, label = QTensor(data), QTensor(label)

                output = model(data)

                loss_fun = CategoricalCrossEntropy()
                losss = loss_fun(label, output)

                losss.backward()

                optimizer._step()
                accuracy += get_score(output, label)

                loss += losss.item()
                # print(f"epoch:{i}, train_accuracy:{accuracy}")
                # print(f"epoch:{i}, train_loss:{losss}")
                count += batch_size

            print(f"epoch:{i}, train_accuracy_for_each_batch:{accuracy/count}")
            print(f"epoch:{i}, train_loss_for_each_batch:{loss/count}")


    def test():
        batch_size = 5
        model.eval()
        print("start eval...................")
        x_test, y_test = circle(500)
        test_accuracy = 0
        count = 0
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

        for test_data, test_label in get_minibatch_data(x_test, y_test,
                                                        batch_size):

            test_data, test_label = QTensor(test_data), QTensor(test_label)
            output = model(test_data)
            test_accuracy += get_score(output, test_label)
            count += batch_size
        print(f"test_accuracy:{test_accuracy/count}")


    if __name__ == "__main__":
        train()
        test()

QDRL在测试数据上准确率变化情况：

.. image:: ./images/qdrl_accuracy.png
   :width: 600 px
   :align: center

|

3. VSQL: Variational Shadow Quantum Learning for Classification模型
=======================================================================
使用可变量子线路构建2分类模型，在与相似参数精度的神经网络对比分类精度，两者精度相近。而量子线路的参数量远小于经典神经网络。
算法基于论文：`Variational Shadow Quantum Learning for Classification Model <https://arxiv.org/abs/2012.08288>`_  复现。

VSQL量子整体模型如下：

.. image:: ./images/vsql_model.PNG
   :width: 600 px
   :align: center

|

VSQL中各个量子比特上的局部量子线路图如下：

.. image:: ./images/vsql_0.png
.. image:: ./images/vsql_1.png
.. image:: ./images/vsql_2.png
.. image:: ./images/vsql_3.png
.. image:: ./images/vsql_4.png
.. image:: ./images/vsql_5.png
.. image:: ./images/vsql_6.png
.. image:: ./images/vsql_7.png
.. image:: ./images/vsql_8.png

|

.. code-block::

    """
    Parameterized quantum circuit for VSQL

    """

        import sys
        sys.path.insert(0, "../")
        import os
        import os.path
        import struct
        import gzip
        from pyvqnet.nn.module import Module
        from pyvqnet.nn.loss import CategoricalCrossEntropy
        from pyvqnet.optim.adam import Adam
        from pyvqnet.data.data import data_generator
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.measure import expval
        from pyvqnet.qnn.quantumlayer import QuantumLayer
        from pyvqnet.qnn.template import AmplitudeEmbeddingCircuit
        from pyvqnet.nn.linear import Linear
        import numpy as np
        import pyqpanda as pq
        import matplotlib.pyplot as plt
        import matplotlib
        try:
            matplotlib.use("TkAgg")
        except:  #pylint:disable=bare-except
            print("Can not use matplot TkAgg")
            pass

        try:
            import urllib.request
        except ImportError:
            raise ImportError("You should use Python 3.x")

        url_base = "http://yann.lecun.com/exdb/mnist/"
        key_file = {
            "train_img": "train-images-idx3-ubyte.gz",
            "train_label": "train-labels-idx1-ubyte.gz",
            "test_img": "t10k-images-idx3-ubyte.gz",
            "test_label": "t10k-labels-idx1-ubyte.gz"
        }


        def _download(dataset_dir, file_name):
            """
            Download function for mnist dataset file
            """
            file_path = dataset_dir + "/" + file_name

            if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as file:
                    file_path_ungz = file_path[:-3].replace("\\", "/")
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz, "wb").write(file.read())
                return

            print("Downloading " + file_name + " ... ")
            urllib.request.urlretrieve(url_base + file_name, file_path)
            if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as file:
                    file_path_ungz = file_path[:-3].replace("\\", "/")
                    file_path_ungz = file_path_ungz.replace("-idx", ".idx")
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz, "wb").write(file.read())
            print("Done")


        def download_mnist(dataset_dir):
            for v in key_file.values():
                _download(dataset_dir, v)


        if not os.path.exists("./result"):
            os.makedirs("./result")
        else:
            pass


        def circuits_of_vsql(x, weights, qlist, clist, machine):  #pylint:disable=unused-argument
            """
            VSQL model of quantum circuits
            """
            weights = weights.reshape([depth + 1, 3, n_qsc])

            def subcir(weights, qlist, depth, n_qsc, n_start):  #pylint:disable=redefined-outer-name
                cir = pq.QCircuit()

                for i in range(n_qsc):
                    cir.insert(pq.RX(qlist[n_start + i], weights[0][0][i]))
                    cir.insert(pq.RY(qlist[n_start + i], weights[0][1][i]))
                    cir.insert(pq.RX(qlist[n_start + i], weights[0][2][i]))
                for repeat in range(1, depth + 1):
                    for i in range(n_qsc - 1):
                        cir.insert(pq.CNOT(qlist[n_start + i], qlist[n_start + i + 1]))
                    cir.insert(pq.CNOT(qlist[n_start + n_qsc - 1], qlist[n_start]))
                    for i in range(n_qsc):
                        cir.insert(pq.RY(qlist[n_start + i], weights[repeat][1][i]))

                return cir

            def get_pauli_str(n_start, n_qsc):  #pylint:disable=redefined-outer-name
                pauli_str = ",".join("X" + str(i)
                                    for i in range(n_start, n_start + n_qsc))
                return {pauli_str: 1.0}

            f_i = []
            origin_in = AmplitudeEmbeddingCircuit(x, qlist)
            for st in range(n - n_qsc + 1):
                psd = get_pauli_str(st, n_qsc)
                cir = pq.QCircuit()
                cir.insert(origin_in)
                cir.insert(subcir(weights, qlist, depth, n_qsc, st))
                prog = pq.QProg()
                prog.insert(cir)

                f_ij = expval(machine, prog, psd, qlist)
                f_i.append(f_ij)
            f_i = np.array(f_i)
            return f_i


        #GLOBAL VAR
        n = 10
        n_qsc = 2
        depth = 1


        class QModel(Module):
            """
            Model of VSQL
            """
            def __init__(self):
                super().__init__()
                self.vq = QuantumLayer(circuits_of_vsql, (depth + 1) * 3 * n_qsc,
                                    "cpu", 10)
                self.fc = Linear(n - n_qsc + 1, 2)

            def forward(self, x):
                x = self.vq(x)
                x = self.fc(x)

                return x


        class Model(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(input_channels=28 * 28, output_channels=2)

            def forward(self, x):

                x = tensor.flatten(x, 1)
                x = self.fc1(x)
                return x


        def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):
            """
            load mnist data
            """
            from array import array as pyarray
            download_mnist(path)
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
            images = np.zeros((num, rows, cols), dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels


        def run_vsql():
            """
            VQSL MODEL
            """
            digits = [0, 1]
            x_train, y_train = load_mnist("training_data", digits)
            x_train = x_train / 255
            y_train = y_train.reshape(-1, 1)
            y_train = np.eye(len(digits))[y_train].reshape(-1, len(digits)).astype(
                np.int64)
            x_test, y_test = load_mnist("testing_data", digits)
            x_test = x_test / 255
            y_test = y_test.reshape(-1, 1)
            y_test = np.eye(len(digits))[y_test].reshape(-1,
                                                        len(digits)).astype(np.int64)

            x_train_list = []
            x_test_list = []
            for i in range(x_train.shape[0]):
                x_train_list.append(
                    np.pad(x_train[i, :, :].flatten(), (0, 240),
                        constant_values=(0, 0)))
            x_train = np.array(x_train_list)

            for i in range(x_test.shape[0]):
                x_test_list.append(
                    np.pad(x_test[i, :, :].flatten(), (0, 240),
                        constant_values=(0, 0)))

            x_test = np.array(x_test_list)

            x_train = x_train[:500]
            y_train = y_train[:500]

            x_test = x_test[:100]
            y_test = y_test[:100]
            print("model start")
            model = QModel()

            optimizer = Adam(model.parameters(), lr=0.1)

            model.train()
            result_file = open("./result/vqslrlt.txt", "w")
            for epoch in range(1, 3):

                model.train()
                full_loss = 0
                n_loss = 0
                n_eval = 0
                batch_size = 1
                correct = 0
                for x, y in data_generator(x_train,
                                        y_train,
                                        batch_size=batch_size,
                                        shuffle=True):
                    optimizer.zero_grad()
                    try:
                        x = x.reshape(batch_size, 1024)
                    except:  #pylint:disable=bare-except
                        x = x.reshape(-1, 1024)

                    output = model(x)
                    cceloss = CategoricalCrossEntropy()
                    loss = cceloss(y, output)
                    loss.backward()
                    optimizer._step()

                    full_loss += loss.item()
                    n_loss += batch_size
                    np_output = np.array(output.data, copy=False)
                    mask = np_output.argmax(1) == y.argmax(1)
                    correct += sum(mask)
                    print(f" n_loss {n_loss} Train Accuracy: {correct/n_loss} ")
                print(f"Train Accuracy: {correct/n_loss} ")
                print(f"Epoch: {epoch}, Loss: {full_loss / n_loss}")
                result_file.write(f"{epoch}\t{full_loss / n_loss}\t{correct/n_loss}\t")

                # Evaluation
                model.eval()
                print("eval")
                correct = 0
                full_loss = 0
                n_loss = 0
                n_eval = 0
                batch_size = 1
                for x, y in data_generator(x_test,
                                        y_test,
                                        batch_size=batch_size,
                                        shuffle=True):
                    x = x.reshape(1, 1024)
                    output = model(x)

                    cceloss = CategoricalCrossEntropy()
                    loss = cceloss(y, output)
                    full_loss += loss.item()

                    np_output = np.array(output.data, copy=False)
                    mask = np_output.argmax(1) == y.argmax(1)
                    correct += sum(mask)
                    n_eval += 1
                    n_loss += 1

                print(f"Eval Accuracy: {correct/n_eval}")
                result_file.write(f"{full_loss / n_loss}\t{correct/n_eval}\n")

            result_file.close()
            del model
            print("\ndone vqsl\n")


        if __name__ == "__main__":

            run_vsql()


VSQL在测试数据上准确率变化情况：

.. image:: ./images/vsql_cacc.PNG
   :width: 600 px
   :align: center

.. image:: ./images/vsql_closs.PNG
   :width: 600 px
   :align: center

.. image:: ./images/vsql_qacc.PNG
   :width: 600 px
   :align: center

.. image:: ./images/vsql_qloss.PNG
   :width: 600 px
   :align: center

|

4.Quanvolution进行图像分类
===============================

在此示例中，我们实现了量子卷积神经网络，这是一种最初在论文 `Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits <https://arxiv.org/abs/1904.04767>`_ 中介绍的方法。

类似经典卷积，Quanvolution有以下步骤：
输入图像的一小块区域，在我们的例子中是 2×2方形经典数据，嵌入到量子电路中。
在此示例中，这是通过将参数化旋转逻辑门应用于在基态中初始化的量子位来实现的。此处的卷积核由参考文献中提出的随机电路生成变分线路。
最后测量量子系统，获得经典期望值列表。 
类似于经典的卷积层，每个期望值都映射到单个输出像素的不同通道。
在不同区域重复相同的过程，可以扫描完整的输入图像，生成一个输出对象，该对象将被构造为多通道图像。
为了进行分类任务，本例在Quanvolution获取测量值后，使用经典全连接层 ``Linear`` 进行分类任务。
与经典卷积的主要区别在于，Quanvolution可以生成高度复杂的内核，其计算至少在原则上是经典难处理的。

.. image:: ./images/quanvo.png
   :width: 600 px
   :align: center

|


Mnist数据集定义

.. code-block::

    import os
    import os.path
    import struct
    import gzip
    import sys
    sys.path.insert(0, "../")
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import NLL_Loss
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.measure import expval
    from pyvqnet.nn.linear import Linear
    import numpy as np
    from pyvqnet.qnn.qcnn import Quanvolution
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  #pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = "http://yann.lecun.com/exdb/mnist/"
    key_file = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_label": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_label": "t10k-labels-idx1-ubyte.gz"
    }


    def _download(dataset_dir, file_name):
        """
        Download function for mnist dataset file
        """
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                file_path_ungz = file_path_ungz.replace("-idx", ".idx")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
        print("Done")


    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir, v)


    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass


    def load_mnist(dataset="training_data", digits=np.arange(10), path="./"):
        """
        load mnist data
        """
        from array import array as pyarray
        download_mnist(path)
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
        images = np.zeros((num, rows, cols))

        labels = np.zeros((num, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                    cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

模型定义与运行函数定义

.. code-block::

    class QModel(Module):

        def __init__(self):
            super().__init__()
            self.vq = Quanvolution([4, 2], (2, 2))
            self.fc = Linear(4 * 14 * 14, 10)

        def forward(self, x):
            x = self.vq(x)
            x = tensor.flatten(x, 1)
            x = self.fc(x)
            x = tensor.log_softmax(x)
            return x



    def run_quanvolution():

        digit = 10
        x_train, y_train = load_mnist("training_data", digits=np.arange(digit))
        x_train = x_train / 255

        y_train = y_train.flatten()

        x_test, y_test = load_mnist("testing_data", digits=np.arange(digit))

        x_test = x_test / 255
        y_test = y_test.flatten()

        x_train = x_train[:500]
        y_train = y_train[:500]

        x_test = x_test[:100]
        y_test = y_test[:100]

        print("model start")
        model = QModel()

        optimizer = Adam(model.parameters(), lr=5e-3)

        model.train()
        result_file = open("quanvolution.txt", "w")

        cceloss = NLL_Loss()
        N_EPOCH = 15

        for epoch in range(1, N_EPOCH):

            model.train()
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 10
            correct = 0
            for x, y in data_generator(x_train,
                                    y_train,
                                    batch_size=batch_size,
                                    shuffle=True):
                optimizer.zero_grad()
                try:
                    x = x.reshape(batch_size, 1, 28, 28)
                except:  #pylint:disable=bare-except
                    x = x.reshape(-1, 1, 28, 28)

                output = model(x)

                loss = cceloss(y, output)
                print(f"loss {loss}")
                loss.backward()
                optimizer._step()

                full_loss += loss.item()
                n_loss += batch_size
                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y

                correct += sum(mask)
                print(f"correct {correct}")
            print(f"Train Accuracy: {correct/n_loss}%")
            print(f"Epoch: {epoch}, Loss: {full_loss / n_loss}")
            result_file.write(f"{epoch}\t{full_loss / n_loss}\t{correct/n_loss}\t")

            # Evaluation
            model.eval()
            print("eval")
            correct = 0
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            for x, y in data_generator(x_test,
                                    y_test,
                                    batch_size=batch_size,
                                    shuffle=True):
                x = x.reshape(-1, 1, 28, 28)
                output = model(x)

                loss = cceloss(y, output)
                full_loss += loss.item()

                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y
                correct += sum(mask)
                n_eval += 1
                n_loss += 1

            print(f"Eval Accuracy: {correct/n_eval}")
            result_file.write(f"{full_loss / n_loss}\t{correct/n_eval}\n")

        result_file.close()
        del model
        print("\ndone\n")


    if __name__ == "__main__":

        run_quanvolution()

训练集、验证集损失，训练集、验证集分类准确率随Epoch 变换情况。

.. code-block::

    # epoch train_loss      train_accuracy eval_loss    eval_accuracy
    # 1	0.2488900272846222	0.232	1.7297331787645818	0.39
    # 2	0.12281704187393189	0.646	1.201728610806167	0.61
    # 3	0.08001763761043548	0.772	0.8947569639235735	0.73
    # 4	0.06211201059818268	0.83	0.777864265316166	0.74
    # 5	0.052190632969141004	0.858	0.7291000287979841	0.76
    # 6	0.04542196464538574	0.87	0.6764470228599384	0.8
    # 7	0.04029472427070141	0.896	0.6153804161818698	0.79
    # 8	0.03600500610470772	0.902	0.5644993982824963	0.81
    # 9	0.03230033944547176	0.916	0.528938240573043	0.81
    # 10	0.02912954458594322	0.93	0.5058713140769396	0.83
    # 11	0.026443827204406262	0.936	0.49064547760412097	0.83
    # 12	0.024144304402172564	0.942	0.4800815625616815	0.82
    # 13	0.022141477409750223	0.952	0.4724775951183983	0.83
    # 14	0.020372112181037665	0.956	0.46692863543197743	0.83

量子自编码器模型
******************************

1.量子自编码器
======================

经典的自动编码器是一种神经网络，可以在高维空间学习数据的高效低维表示。自动编码器的任务是，给定一个输入x，将x映射到一个低维点y，这样x就可以从y中恢复。
可以选择底层自动编码器网络的结构，以便在较小的维度上表示数据，从而有效地压缩输入。受这一想法的启发，量子自动编码器的模型来对量子数据执行类似的任务。
量子自动编码器被训练来压缩量子态的特定数据集，而经典的压缩算法无法使用。量子自动编码器的参数采用经典优化算法进行训练。
我们展示了一个简单的可编程线路的例子，它可以被训练成一个高效的自动编码器。我们在量子模拟的背景下应用我们的模型来压缩哈伯德模型和分子哈密顿量的基态。
该例子参考自 `Quantum autoencoders for efficient compression of quantum data <https://arxiv.org/pdf/1612.02806.pdf>`_ .

QAE量子线路：

.. image:: ./images/QAE_Quantum_Cir.png
   :width: 600 px
   :align: center

|

.. code-block::

    """
    Quantum AutoEncoder demo

    

    """

    import os
    import sys
    sys.path.insert(0,'../')
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import  fidelityLoss
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.qnn.qae.qae import QAElayer
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    try:
        import urllib.request
    except ImportError:
        raise ImportError('You should use Python 3.x')
    import os.path
    import gzip

    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }

    def _download(dataset_dir,file_name):
        file_path = dataset_dir + "/" + file_name
        
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as f:
                file_path_ungz = file_path[:-3].replace('\\', '/')
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz,"wb").write(f.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as f:
                    file_path_ungz = file_path[:-3].replace('\\', '/')
                    file_path_ungz = file_path_ungz.replace('-idx', '.idx')
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz,"wb").write(f.read())
        print("Done")
        
    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir,v)


    class Model(Module):

        def __init__(self, trash_num: int = 2, total_num: int = 7):
            super().__init__()
            self.pqc = QAElayer(trash_num, total_num)

        def forward(self, x):
            
            x = self.pqc(x)
            return x

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         # 下载数据
        import os, struct
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
        elif dataset == "testing_data":
            fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))

        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        N = len(ind)
        images = np.zeros((N, rows, cols))
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def run2():
        ##load dataset

        x_train, y_train = load_mnist("training_data")                      # 下载训练数据
        x_train = x_train / 255                                             # 将数据进行归一化处理[0,1]

        x_test, y_test = load_mnist("testing_data")

        x_test = x_test / 255

        x_train = x_train.reshape([-1, 1, 28, 28])
        x_test = x_test.reshape([-1, 1, 28, 28])
        x_train = x_train[:100, :, :, :]
        x_train = np.resize(x_train, [x_train.shape[0], 1, 2, 2])

        x_test = x_test[:10, :, :, :]
        x_test = np.resize(x_test, [x_test.shape[0], 1, 2, 2])
        encode_qubits = 4
        latent_qubits = 2
        trash_qubits = encode_qubits - latent_qubits
        total_qubits = 1 + trash_qubits + encode_qubits
        print("model start")
        model = Model(trash_qubits, total_qubits)

        optimizer = Adam(model.parameters(), lr=0.005)                        
        model.train()
        F1 = open("rlt.txt", "w")
        loss_list = []
        loss_list_test = []
        fidelity_train = []
        fidelity_val = []

        for epoch in range(1, 10):
            running_fidelity_train = 0
            running_fidelity_val = 0
            print(f"epoch {epoch}")
            model.train()
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            correct = 0
            iter = 0
            if epoch %5 ==1:
                optimizer.lr  = optimizer.lr *0.5
            for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True): #shuffle batch rather than data

                x = x.reshape((-1, encode_qubits))
                x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]), x), 1)
                optimizer.zero_grad()
                output = model(x)
                iter += 1
                np_out = np.array(output.data)
                floss = fidelityLoss()
                loss = floss(output)
                loss_data = np.array(loss.data)
                loss.backward()

                running_fidelity_train += np_out[0]
                optimizer._step()
                full_loss += loss_data[0]
                n_loss += batch_size
                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y.argmax(1)

                correct += sum(mask)

            loss_output = full_loss / n_loss
            print(f"Epoch: {epoch}, Loss: {loss_output}")
            loss_list.append(loss_output)


            # Evaluation
            model.eval()
            correct = 0
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            for x, y in data_generator(x_test, y_test, batch_size=batch_size, shuffle=True):
                x = x.reshape((-1, encode_qubits))
                x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]),x),1)
                output = model(x)

                floss = fidelityLoss()
                loss = floss(output)
                loss_data = np.array(loss.data)
                full_loss += loss_data[0]
                running_fidelity_val += np.array(output.data)[0]

                n_eval += 1
                n_loss += 1

            loss_output = full_loss / n_loss
            print(f"Epoch: {epoch}, Loss: {loss_output}")
            loss_list_test.append(loss_output)

            fidelity_train.append(running_fidelity_train / 64)
            fidelity_val.append(running_fidelity_val / 64)

        figure_path = os.path.join(os.getcwd(), 'QAE-rate1.png')
        plt.plot(loss_list, color="blue", label="train")
        plt.plot(loss_list_test, color="red", label="validation")
        plt.title('QAE')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(figure_path)
        plt.show()

        F1.write(f"done\n")
        F1.close()
        del model

    if __name__ == '__main__':
        run2()

运行上述代码得到的QAE误差值,该loss为1/保真度，趋向于1表示保真度接近1。

.. image:: ./images/qae_train_loss.png
   :width: 600 px
   :align: center

|

量子线路结构学习
******************************
1.量子线路结构学习
=====================
在量子线路结构中，最经常使用的带参数的量子门就是RZ、RY、RX门，但是在什么情况下使用什么门是一个十分值得研究的问题，一种方法就是随机选择，但是这种情况很有可能达不到最好的效果。
Quantum circuit structure learning任务的核心目标就是找到最优的带参量子门组合。
这里的做法是这一组最优的量子逻辑门要使得目标函数（loss function）取得最小值。

.. code-block::

    """
    Quantum Circuits Strcture Learning Demo

    """

    import sys
    sys.path.insert(0,"../")

    import copy
    import pyqpanda as pq
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn.measure import expval
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  #pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    machine = pq.CPUQVM()
    machine.init_qvm()
    nqbits = machine.qAlloc_many(2)

    def gen(param, generators, qbits, circuit):
        if generators == "X":
            circuit.insert(pq.RX(qbits, param))
        elif generators == "Y":
            circuit.insert(pq.RY(qbits, param))
        else:
            circuit.insert(pq.RZ(qbits, param))

    def circuits(params, generators, circuit):
        gen(params[0], generators[0], nqbits[0], circuit)
        gen(params[1], generators[1], nqbits[1], circuit)
        circuit.insert(pq.CNOT(nqbits[0], nqbits[1]))
        prog = pq.QProg()
        prog.insert(circuit)
        return prog

    def ansatz1(params: QTensor, generators):
        circuit = pq.QCircuit()
        params = params.to_numpy()
        prog = circuits(params, generators, circuit)
        return expval(machine, prog, {"Z0": 1},
                    nqbits), expval(machine, prog, {"Y1": 1}, nqbits)


    def ansatz2(params: QTensor, generators):
        circuit = pq.QCircuit()
        params = params.to_numpy()
        prog = circuits(params, generators, circuit)
        return expval(machine, prog, {"X0": 1}, nqbits)


    def loss(params, generators):
        z, y = ansatz1(params, generators)
        x = ansatz2(params, generators)
        return 0.5 * y + 0.8 * z - 0.2 * x


    def rotosolve(d, params, generators, cost, M_0):#pylint:disable=invalid-name
        """
        rotosolve algorithm implementation
        """
        params[d] = np.pi / 2.0
        m0_plus = cost(QTensor(params), generators)
        params[d] = -np.pi / 2.0
        m0_minus = cost(QTensor(params), generators)
        a = np.arctan2(2.0 * M_0 - m0_plus - m0_minus,
                    m0_plus - m0_minus)  # returns value in (-pi,pi]
        params[d] = -np.pi / 2.0 - a
        if params[d] <= -np.pi:
            params[d] += 2 * np.pi
        return cost(QTensor(params), generators)


    def optimal_theta_and_gen_helper(index, params, generators):
        """
        find optimal varaibles
        """
        params[index] = 0.
        m0 = loss(QTensor(params), generators)  #init value
        for kind in ["X", "Y", "Z"]:
            generators[index] = kind
            params_cost = rotosolve(index, params, generators, loss, m0)
            if kind == "X" or params_cost <= params_opt_cost:
                params_opt_d = params[index]
                params_opt_cost = params_cost
                generators_opt_d = kind
        return params_opt_d, generators_opt_d


    def rotoselect_cycle(params: np, generators):
        for index in range(params.shape[0]):
            params[index], generators[index] = optimal_theta_and_gen_helper(
                index, params, generators)
        return params, generators


    params = QTensor(np.array([0.3, 0.25]))
    params = params.to_numpy()
    generator = ["X", "Y"]
    generators = copy.deepcopy(generator)
    epoch = 20
    state_save = []
    for i in range(epoch):
        state_save.append(loss(QTensor(params), generators))
        params, generators = rotoselect_cycle(params, generators)

    print("Optimal generators are: {}".format(generators))
    print("Optimal params are: {}".format(params))
    steps = np.arange(0, epoch)


    plt.plot(steps, state_save, "o-")
    plt.title("rotoselect")
    plt.xlabel("cycles")
    plt.ylabel("cost")
    plt.yticks(np.arange(-1.25, 0.80, 0.25))
    plt.tight_layout()
    plt.show()


运行上述代码得到的量子线路结构。可见为一个 :math:`RX`,一个 :math:`RY`

.. image:: ./images/final_quantum_circuit.png
   :width: 600 px
   :align: center

以及逻辑门中的参数 :math:`\theta_1`, :math:`\theta_2` 不同参数下的损失函数

.. image:: ./images/loss3d.png
   :width: 600 px
   :align: center

|

量子经典神经网络混合模型
*******************************

1.混合量子经典神经网络模型
============================

机器学习 (ML) 已成为一个成功的跨学科领域，旨在从数据中以数学方式提取可概括的信息。量子机器学习寻求利用量子力学原理来增强机器学习，反之亦然。
无论您的目标是通过将困难的计算外包给量子计算机来增强经典 ML 算法，还是使用经典 ML 架构优化量子算法——两者都属于量子机器学习 (QML) 的范畴。
在本章中，我们将探讨如何部分量化经典神经网络以创建混合量子经典神经网络。量子线路由量子逻辑门构成，这些逻辑门实现的量子计算被论文 `Quantum Circuit Learning <https://arxiv.org/abs/1803.00745>`_ 证明是可微分。因此研究者尝试将量子线路与经典神经网络模块放到一起同时进行混合量子经典机器学习任务的训练。
我们将编写一个简单的示例，使用VQNet实现一个神经网络模型训练任务。此示例的目的是展示VQNet的简便性，并鼓励 ML 从业者探索量子计算的可能性。

数据准备
----------------

我们将使用 `MNIST datasets <http://yann.lecun.com/exdb/mnist/>`_ 这一神经网络最基础的手写数字数据库作为分类数据 。
我们首先加载MNIST并过滤包含0和1的数据样本。这些样本分为训练数据 training_data 和测试数据 testing_data，它们每条数据均为1*784的维度大小。

.. code-block::

    import time
    import os
    import struct
    import gzip
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.conv import Conv2D

    from pyvqnet.nn import activation as F
    from pyvqnet.nn.pooling import MaxPool2D
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.tensor import QTensor
    import pyqpanda as pq

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  #pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }

    def _download(dataset_dir,file_name):
        file_path = dataset_dir + "/" + file_name
        
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as f:
                file_path_ungz = file_path[:-3].replace('\\', '/')
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz,"wb").write(f.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as f:
                    file_path_ungz = file_path[:-3].replace('\\', '/')
                    file_path_ungz = file_path_ungz.replace('-idx', '.idx')
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz,"wb").write(f.read())
        print("Done")
        
    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir,v)
    
    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         # 下载数据
        import os, struct
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
        elif dataset == "testing_data":
            fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        N = len(ind)
        images = np.zeros((N, rows, cols))
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def data_select(train_num, test_num):
        x_train, y_train = load_mnist("training_data")  
        x_test, y_test = load_mnist("testing_data")
        # Train Leaving only labels 0 and 1
        idx_train = np.append(np.where(y_train == 0)[0][:train_num],
                        np.where(y_train == 1)[0][:train_num])
        x_train = x_train[idx_train]
        y_train = y_train[idx_train]
        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)
        # Test Leaving only labels 0 and 1
        idx_test = np.append(np.where(y_test == 0)[0][:test_num],
                        np.where(y_test == 1)[0][:test_num])
        x_test = x_test[idx_test]
        y_test = y_test[idx_test]
        x_test = x_test / 255
        y_test = np.eye(2)[y_test].reshape(-1, 2)
        return x_train, y_train, x_test, y_test

    n_samples_show = 6

    x_train, y_train, x_test, y_test = data_select(100, 50)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    for img ,targets in zip(x_test,y_test):
        if n_samples_show <= 3:
            break
        
        if targets[0] == 1:
            axes[n_samples_show - 1].set_title("Labeled: 0")
            axes[n_samples_show - 1].imshow(img.squeeze(), cmap='gray')
            axes[n_samples_show - 1].set_xticks([])
            axes[n_samples_show - 1].set_yticks([])
            n_samples_show -= 1

    for img ,targets in zip(x_test,y_test):
        if n_samples_show <= 0:
            break
        
        if targets[0] == 0:
            axes[n_samples_show - 1].set_title("Labeled: 1")
            axes[n_samples_show - 1].imshow(img.squeeze(), cmap='gray')
            axes[n_samples_show - 1].set_xticks([])
            axes[n_samples_show - 1].set_yticks([])
            n_samples_show -= 1    
        
    plt.show()

.. image:: ./images/mnsit_data_examples.png
   :width: 600 px
   :align: center

|

构建量子线路
-----------------

在本例中，我们使用本源量子的 `pyQPanda <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ 
定义了一个1量子比特的简单量子线路，该线路将经典神经网络层的输出作为输入，通过 ``H``, ``RY`` 逻辑门进行量子数据编码，并计算z方向的哈密顿期望值作为输出。

.. code-block::

    from pyqpanda import *
    import pyqpanda as pq
    import numpy as np
    def circuit(weights):
        num_qubits = 1
        #pyQPanda 创建模拟器
        machine = pq.CPUQVM()
        machine.init_qvm()
        #pyQPanda 分配量子比特
        qubits = machine.qAlloc_many(num_qubits)
        #pyQPanda 分配经典比特辅助测量
        cbits = machine.cAlloc_many(num_qubits)
        #构建线路
        circuit = pq.QCircuit()
        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.RY(qubits[0], weights[0]))

        prog = pq.QProg()
        prog.insert(circuit)
        prog << measure_all(qubits, cbits)

        #运行量子程序
        result = machine.run_with_configuration(prog, cbits, 100)
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / 100
        expectation = np.sum(states * probabilities)
        return expectation

.. image:: ./images/hqcnn_quantum_cir.png
   :width: 600 px
   :align: center

|

构建混合量子神经网络
-----------------------

由于量子线路可以和经典神经网络一起进行自动微分的计算，
因此我们可以使用VQNet的2维卷积层 ``Conv2D`` ，池化层 ``MaxPool2D`` ，全连接层 ``Linear`` 以及刚才构建的量子线路circuit构建模型。
通过以下代码中继承于VQNet自动微分模块 ``Module`` 的 Net 以及 Hybrid 类的定义，以及模型前传函数 ``forward()`` 中对数据前向计算的定义，我们构建了一个可以自动微分的模型
将本例中MNIST的数据进行卷积，降维，量子编码，测量，获取分类任务所需的最终特征。

.. code-block::

    #量子计算层的前传和梯度计算函数的定义，其需要继承于抽象类Module
    class Hybrid(Module):
        """ Hybrid quantum - Quantum layer definition """
        def __init__(self, shift):
            super(Hybrid, self).__init__()
            self.shift = shift
        def forward(self, input): 
            self.input = input
            expectation_z = circuit(np.array(input.data))
            result = [[expectation_z]]
            requires_grad = input.requires_grad
            def _backward(g, input):
                """ Backward pass computation """
                input_list = np.array(input.data)
                shift_right = input_list + np.ones(input_list.shape) * self.shift
                shift_left = input_list - np.ones(input_list.shape) * self.shift

                gradients = []
                for i in range(len(input_list)):
                    expectation_right = circuit(shift_right[i])
                    expectation_left = circuit(shift_left[i])

                    gradient = expectation_right - expectation_left
                    gradients.append(gradient)
                gradients = np.array([gradients]).T
                return gradients * np.array(g)

            nodes = []
            if input.requires_grad:
                nodes.append(QTensor.GraphNode(tensor=input, df=lambda g: _backward(g, input)))
            return QTensor(data=result, requires_grad=requires_grad, nodes=nodes)

    #模型定义
    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.conv2 = Conv2D(input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.fc1 = Linear(input_channels=256, output_channels=64)
            self.fc2 = Linear(input_channels=64, output_channels=1)
            self.hybrid = Hybrid(np.pi / 2)
            self.fc3 = Linear(input_channels=1, output_channels=2)

        def forward(self, x):
            x = F.ReLu()(self.conv1(x))  # 1 6 24 24
            x = self.maxpool1(x)
            x = F.ReLu()(self.conv2(x))  # 1 16 8 8
            x = self.maxpool2(x)
            x = tensor.flatten(x, 1)   # 1 256
            x = F.ReLu()(self.fc1(x))  # 1 64
            x = self.fc2(x)    # 1 1
            x = self.hybrid(x)
            x = self.fc3(x)
            return x

.. image:: ./images/hqcnnmodel.PNG
   :width: 600 px
   :align: center

|

训练和测试
-----------------

通过上面代码示例，我们已经定义了模型。与经典神经网络模型训练类似， 我们还需要做的是实例化该模型，定义损失函数以及优化器以及定义整个训练测试流程。
对于形如下图的混合神经网络模型，我们通过循环输入数据前向计算损失值，并在反向计算中自动计算出各个待训练参数的梯度，并使用优化器进行参数优化，直到迭代次数满足预设值。

.. image:: ./images/hqcnnarch.PNG
   :width: 600 px
   :align: center

|

.. code-block::

    x_train, y_train, x_test, y_test = data_select(1000, 100)
    #实例化
    model = Net() 
    #使用Adam完成此任务就足够了，model.parameters（）是模型需要计算的参数。
    optimizer = Adam(model.parameters(), lr=0.005)
    #分类任务使用交叉熵函数
    loss_func = CategoricalCrossEntropy()

    #训练次数    
    epochs = 10
    train_loss_list = []
    val_loss_list = []
    train_acc_list =[]
    val_acc_list = []


    for epoch in range(1, epochs):
        total_loss = []
        model.train()
        batch_size = 1
        correct = 0
        n_train = 0
        for x, y in data_generator(x_train, y_train, batch_size=1, shuffle=True):

            x = x.reshape(-1, 1, 28, 28)
            optimizer.zero_grad()
            output = model(x)       
            loss = loss_func(y, output)  
            loss_np = np.array(loss.data)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_train += batch_size
            loss.backward()
            optimizer._step()
            total_loss.append(loss_np)

        train_loss_list.append(np.sum(total_loss) / len(total_loss))
        train_acc_list.append(np.sum(correct) / n_train)
        print("{:.0f} loss is : {:.10f}".format(epoch, train_loss_list[-1]))

        model.eval()
        correct = 0
        n_eval = 0

        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            loss = loss_func(y, output)
            loss_np = np.array(loss.data)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_eval += 1
            
            total_loss.append(loss_np)
        print(f"Eval Accuracy: {correct / n_eval}")
        val_loss_list.append(np.sum(total_loss) / len(total_loss))
        val_acc_list.append(np.sum(correct) / n_eval)

数据可视化
----------------

训练和测试数据上的数据损失函数与准确率的可视化曲线。

.. code-block::

    import os
    plt.figure()
    xrange = range(1,len(train_loss_list)+1)
    figure_path = os.path.join(os.getcwd(), 'HQCNN LOSS.png')
    plt.plot(xrange,train_loss_list, color="blue", label="train")
    plt.plot(xrange,val_loss_list, color="red", label="validation")
    plt.title('HQCNN')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(np.arange(1, epochs +1,step = 2))
    plt.legend(loc="upper right")
    plt.savefig(figure_path)
    plt.show()

    plt.figure()
    figure_path = os.path.join(os.getcwd(), 'HQCNN Accuracy.png')
    plt.plot(xrange,train_acc_list, color="blue", label="train")
    plt.plot(xrange,val_acc_list, color="red", label="validation")
    plt.title('HQCNN')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(1, epochs +1,step = 2))
    plt.legend(loc="lower right")
    plt.savefig(figure_path)
    plt.show()


.. image:: ./images/HQCNNLOSS.png
   :width: 600 px
   :align: center

.. image:: ./images/HQCNNAccuracy.png
   :width: 600 px
   :align: center

|

.. code-block::

    n_samples_show = 6
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    model.eval()
    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        if count == n_samples_show:
            break
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        pred = QTensor.argmax(output, [1],False)
        axes[count].imshow(x[0].squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
        count += 1
    plt.show()

.. image:: ./images/eval_test.png
   :width: 600 px
   :align: center

|

2.混合量子经典迁移学习模型
===============================

我们将一种称为迁移学习的机器学习方法应用于基于混合经典量子网络的图像分类器。我们将编写一个将pyQPanda与VQNet集成的简单示例。
迁移学习是一种成熟的人工神经网络训练技术，它基于一般直觉，即如果预训练的网络擅长解决给定的问题，那么，只需一些额外的训练，它也可以用来解决一个不同但相关的问题。

                                                            .. centered:: 量子部分线路图

.. image:: ./images/QTransferLearning_cir.png
   :width: 600 px
   :align: center

|

.. code-block::

    """
    Quantum Classic Nerual Network Transfer Learning demo

    
    """

    import os
    import os.path
    import gzip
    import struct
    import numpy as np
    import sys
    sys.path.insert(0,"../")
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.conv import Conv2D
    from pyvqnet.utils.storage import load_parameters, save_parameters
    from pyvqnet.nn import activation as F
    from pyvqnet.nn.pooling import MaxPool2D

    from pyvqnet.nn.loss import SoftmaxCrossEntropy
    from pyvqnet.optim.sgd import SGD
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    import pyqpanda as pq
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  #pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = "http://yann.lecun.com/exdb/mnist/"
    key_file = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_label": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_label": "t10k-labels-idx1-ubyte.gz"
    }


    def _download(dataset_dir, file_name):
        """
        Download dataset
        """
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                file_path_ungz = file_path_ungz.replace("-idx", ".idx")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
        print("Done")


    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir, v)

    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass

    class CNN(Module):
        """
        Classical CNN
        """
        def __init__(self):
            super(CNN, self).__init__()

            self.conv1 = Conv2D(input_channels=1,
                                output_channels=16,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding="valid")
            self.relu1 = F.ReLu()

            self.conv2 = Conv2D(input_channels=16,
                                output_channels=32,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding="valid")
            self.relu2 = F.ReLu()
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")

            self.conv3 = Conv2D(input_channels=32,
                                output_channels=64,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding="valid")
            self.relu3 = F.ReLu()

            self.conv4 = Conv2D(input_channels=64,
                                output_channels=128,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding="valid")

            self.relu4 = F.ReLu()
            self.maxpool4 = MaxPool2D([2, 2], [2, 2], padding="valid")

            self.fc1 = Linear(input_channels=128 * 4 * 4, output_channels=1024)
            self.fc2 = Linear(input_channels=1024, output_channels=128)
            self.fc3 = Linear(input_channels=128, output_channels=10)

        def forward(self, x):

            x = self.relu1(self.conv1(x))

            x = self.maxpool2(self.relu2(self.conv2(x)))

            x = self.relu3(self.conv3(x))

            x = self.maxpool4(self.relu4(self.conv4(x)))

            x = tensor.flatten(x, 1)
            x = F.ReLu()(self.fc1(x))

            x = F.ReLu()(self.fc2(x))

            x = self.fc3(x)

            return x


    def load_mnist(dataset="training_data",
                digits=np.arange(2),
                path="./"):
        """
        Load mnist data
        """
        from array import array as pyarray
        download_mnist(path)
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
        images = np.zeros((num, rows, cols))
        labels = np.zeros((num, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                    cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels


    train_size = 50
    eval_size = 2
    EPOCHES = 10


    def classcal_cnn_model_training():
        """
        load train data
        """

        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = CNN()

        optimizer = SGD(model.parameters(), lr=0.005)
        loss_func = SoftmaxCrossEntropy()

        epochs = EPOCHES
        loss_list = []
        model.train()

        save_flag = True
        temp_loss = 0
        for epoch in range(1, epochs):
            total_loss = []
            for x, y in data_generator(x_train,
                                    y_train,
                                    batch_size=4,
                                    shuffle=True):

                x = x.reshape(-1, 1, 28, 28)
                optimizer.zero_grad()
                # Forward pass
                output = model(x)

                # Calculating loss
                loss = loss_func(y, output)  # target output
                loss_np = np.array(loss.data)
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer._step()

                total_loss.append(loss_np)

            loss_list.append(np.sum(total_loss) / len(total_loss))
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

            if save_flag:
                temp_loss = loss_list[-1]
                save_parameters(model.state_dict(), "./result/QCNN_TL_1.model")
                save_flag = False
            else:
                if temp_loss > loss_list[-1]:
                    temp_loss = loss_list[-1]
                    save_parameters(model.state_dict(), "./result/QCNN_TL_1.model")

        model.eval()
        correct = 0
        n_eval = 0

        for x, y in data_generator(x_test, y_test, batch_size=4, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            loss = loss_func(y, output)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_eval += 1
        print(f"Eval Accuracy: {correct / n_eval}")

        n_samples_show = 6
        count = 0
        _, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap="gray")
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title("Predicted {}".format(np.array(pred.data)))
            count += 1
        plt.show()


    def classical_cnn_transferlearning_predict():
        """
        Use test data to eval classic NN model
        """
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_test = x_test / 255

        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = CNN()

        model_parameter = load_parameters("./result/QCNN_TL_1.model")
        model.load_state_dict(model_parameter)
        model.eval()
        correct = 0
        n_eval = 0

        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)

            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_eval += 1

        print(f"Eval Accuracy: {correct / n_eval}")

        n_samples_show = 6
        count = 0
        _, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap="gray")
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title("Predicted {}".format(np.array(pred.data)))
            count += 1
        plt.show()

    n_qubits = 4  # Number of qubits
    q_depth = 6  # Depth of the quantum circuit (number of variational layers)

    def Q_H_layer(qubits, nqubits):#pylint:disable=invalid-name
        """Layer of single-qubit Hadamard gates.
        """
        circuit = pq.QCircuit()
        for idx in range(nqubits):
            circuit.insert(pq.H(qubits[idx]))
        return circuit

    def Q_RY_layer(qubits, w):#pylint:disable=invalid-name
        """
        Layer of parametrized qubit rotations around the y axis.
        """
        circuit = pq.QCircuit()
        for idx, element in enumerate(w):
            circuit.insert(pq.RY(qubits[idx], element))
        return circuit

    def Q_entangling_layer(qubits, nqubits):#pylint:disable=invalid-name
        """
        Layer of CNOTs followed by another shifted layer of CNOT.
        """
        # In other words it should apply something like :
        # CNOT  CNOT  CNOT  CNOT...  CNOT
        #   CNOT  CNOT  CNOT...  CNOT
        circuit = pq.QCircuit()
        for i in range(0, nqubits - 1,
                        2):  # Loop over even indices: i=0,2,...N-2
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        for i in range(1, nqubits - 1,
                        2):  # Loop over odd indices:  i=1,3,...N-3
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        return circuit

    def quantum_net(q_input_features, q_weights_flat, qubits, cubits,#pylint:disable=unused-argument
                    machine):
        """
        The variational quantum circuit.
        """
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(n_qubits)
        circuit = pq.QCircuit()

        # Reshape weights
        q_weights = q_weights_flat.reshape([q_depth, n_qubits])

        # Start from state |+> , unbiased w.r.t. |0> and |1>
        circuit.insert(Q_H_layer(qubits, n_qubits))

        # Embed features in the quantum node
        circuit.insert(Q_RY_layer(qubits, q_input_features))

        # Sequence of trainable variational layers
        for k in range(q_depth):
            circuit.insert(Q_entangling_layer(qubits, n_qubits))
            circuit.insert(Q_RY_layer(qubits, q_weights[k]))

        # Expectation values in the Z basis
        prog = pq.QProg()
        prog.insert(circuit)

        exp_vals = []
        for position in range(n_qubits):
            pauli_str = "Z" + str(position)
            pauli_map = pq.PauliOperator(pauli_str, 1)
            hamiltion = pauli_map.toHamiltonian(True)
            exp = machine.get_expectation(prog, hamiltion, qubits)
            exp_vals.append(exp)

        return exp_vals
    def quantum_cnn_transferlearning():
        """
        The quantum cnn transferLearning model main function
        """


        class Q_DressedQuantumNet(Module):#pylint:disable=invalid-name
            """
            module implementing the *dressed* quantum net.
            """
            def __init__(self):
                """
                Definition of the *dressed* layout.
                """

                super().__init__()
                self.pre_net = Linear(128, n_qubits)
                self.post_net = Linear(n_qubits, 10)
                self.qlayer = QuantumLayer(quantum_net, q_depth * n_qubits,
                                        "cpu", n_qubits, n_qubits)

            def forward(self, input_features):
                """
                Defining how tensors are supposed to move through the *dressed* quantum
                net.
                """

                # obtain the input features for the quantum circuit
                # by reducing the feature dimension from 512 to 4
                pre_out = self.pre_net(input_features)
                q_in = tensor.tanh(pre_out) * np.pi / 2.0
                q_out_elem = self.qlayer(q_in)

                result = q_out_elem
                # return the two-dimensional prediction from the postprocessing layer
                return self.post_net(result)

        x_train, y_train = load_mnist("training_data",
                                    digits=np.arange(10))  # 下载训练数据
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = CNN()
        model_param = load_parameters("./result/QCNN_TL_1.model")
        model.load_state_dict(model_param)

        loss_func = SoftmaxCrossEntropy()

        epochs = EPOCHES
        loss_list = []

        eval_losses = []

        model_hybrid = model
        print(model_hybrid)

        for param in model_hybrid.parameters():
            param.requires_grad = False

        model_hybrid.fc3 = Q_DressedQuantumNet()

        optimizer_hybrid = Adam(model_hybrid.fc3.parameters(), lr=0.001)
        model_hybrid.train()

        save_flag = True
        temp_loss = 0
        for epoch in range(1, epochs):
            total_loss = []
            for x, y in data_generator(x_train,
                                    y_train,
                                    batch_size=4,
                                    shuffle=True):
                x = x.reshape(-1, 1, 28, 28)
                optimizer_hybrid.zero_grad()
                # Forward pass
                output = model_hybrid(x)

                loss = loss_func(y, output)  # target output
                loss_np = np.array(loss.data)
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer_hybrid._step()
                total_loss.append(loss_np)

            loss_list.append(np.sum(total_loss) / len(total_loss))
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
            if save_flag:
                temp_loss = loss_list[-1]
                save_parameters(model_hybrid.fc3.state_dict(),
                                "./result/QCNN_TL_FC3.model")
                save_parameters(model_hybrid.state_dict(),
                                "./result/QCNN_TL_ALL.model")
                save_flag = False
            else:
                if temp_loss > loss_list[-1]:
                    temp_loss = loss_list[-1]
                    save_parameters(model_hybrid.fc3.state_dict(),
                                    "./result/QCNN_TL_FC3.model")
                    save_parameters(model_hybrid.state_dict(),
                                    "./result/QCNN_TL_ALL.model")

            correct = 0
            n_eval = 0
            loss_temp = []
            for x1, y1 in data_generator(x_test,
                                        y_test,
                                        batch_size=4,
                                        shuffle=True):
                x1 = x1.reshape(-1, 1, 28, 28)
                output = model_hybrid(x1)
                loss = loss_func(y1, output)
                np_loss = np.array(loss.data)
                np_output = np.array(output.data, copy=False)
                mask = (np_output.argmax(1) == y1.argmax(1))
                correct += np.sum(np.array(mask))
                n_eval += 1
                loss_temp.append(np_loss)
            eval_losses.append(np.sum(loss_temp) / n_eval)
            print("{:.0f} eval loss is : {:.10f}".format(epoch, eval_losses[-1]))

        plt.title("model loss")
        plt.plot(loss_list, color="green", label="train_losses")
        plt.plot(eval_losses, color="red", label="eval_losses")
        plt.ylabel("loss")
        plt.legend(["train_losses", "eval_losses"])
        plt.savefig("qcnn_transfer_learning_classical")
        plt.show()
        plt.close()

        n_samples_show = 6
        count = 0
        _, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model_hybrid.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model_hybrid(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap="gray")
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title("Predicted {}".format(np.array(pred.data)))
            count += 1
        plt.show()


    def quantum_cnn_transferlearning_predict():
        """
        Eval quantum cnn transferlearning model on test data
        """
        n_qubits = 4  # Number of qubits
        q_depth = 6  # Depth of the quantum circuit (number of variational layers)

        def Q_H_layer(qubits, nqubits):#pylint:disable=invalid-name
            """Layer of single-qubit Hadamard gates.
            """
            circuit = pq.QCircuit()
            for idx in range(nqubits):
                circuit.insert(pq.H(qubits[idx]))
            return circuit

        def Q_RY_layer(qubits, w):#pylint:disable=invalid-name
            """Layer of parametrized qubit rotations around the y axis.
            """
            circuit = pq.QCircuit()
            for idx, element in enumerate(w):
                circuit.insert(pq.RY(qubits[idx], element))
            return circuit

        def Q_entangling_layer(qubits, nqubits):#pylint:disable=invalid-name
            """Layer of CNOTs followed by another shifted layer of CNOT.
            """
            # In other words it should apply something like :
            # CNOT  CNOT  CNOT  CNOT...  CNOT
            #   CNOT  CNOT  CNOT...  CNOT
            circuit = pq.QCircuit()
            for i in range(0, nqubits - 1,
                        2):  # Loop over even indices: i=0,2,...N-2
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            for i in range(1, nqubits - 1,
                        2):  # Loop over odd indices:  i=1,3,...N-3
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            return circuit

        def quantum_net(q_input_features, q_weights_flat, qubits, cubits,#pylint:disable=unused-argument
                        machine):
            """
            The variational quantum circuit.
            """
            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(n_qubits)
            circuit = pq.QCircuit()

            # Reshape weights
            q_weights = q_weights_flat.reshape([q_depth, n_qubits])

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            circuit.insert(Q_H_layer(qubits, n_qubits))

            # Embed features in the quantum node
            circuit.insert(Q_RY_layer(qubits, q_input_features))

            # Sequence of trainable variational layers
            for k in range(q_depth):
                circuit.insert(Q_entangling_layer(qubits, n_qubits))
                circuit.insert(Q_RY_layer(qubits, q_weights[k]))

            # Expectation values in the Z basis
            prog = pq.QProg()
            prog.insert(circuit)
            exp_vals = []
            for position in range(n_qubits):
                pauli_str = "Z" + str(position)
                pauli_map = pq.PauliOperator(pauli_str, 1)
                hamiltion = pauli_map.toHamiltonian(True)
                exp = machine.get_expectation(prog, hamiltion, qubits)
                exp_vals.append(exp)

            return exp_vals

        class Q_DressedQuantumNet(Module):
            """
            module implementing the *dressed* quantum net.
            """
            def __init__(self):
                """
                Definition of the *dressed* layout.
                """

                super().__init__()
                self.pre_net = Linear(128, n_qubits)
                self.post_net = Linear(n_qubits, 10)
                self.qlayer = QuantumLayer(quantum_net, q_depth * n_qubits,
                                        "cpu", n_qubits, n_qubits)

            def forward(self, input_features):
                """
                Defining how tensors are supposed to move through the *dressed* quantum
                net.
                """

                # obtain the input features for the quantum circuit
                # by reducing the feature dimension from 512 to 4
                pre_out = self.pre_net(input_features)
                q_in = tensor.tanh(pre_out) * np.pi / 2.0
                q_out_elem = self.qlayer(q_in)

                result = q_out_elem
                # return the two-dimensional prediction from the postprocessing layer
                return self.post_net(result)

        x_train, y_train = load_mnist("training_data",
                                    digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
        x_train = x_train[:2000]
        y_train = y_train[:2000]
        x_test = x_test[:500]
        y_test = y_test[:500]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        # The second method: unified storage and unified reading
        model = CNN()
        model_hybrid = model
        model_hybrid.fc3 = Q_DressedQuantumNet()
        for param in model_hybrid.parameters():
            param.requires_grad = False
        model_param_quantum = load_parameters("./result/QCNN_TL_ALL.model")

        model_hybrid.load_state_dict(model_param_quantum)
        model_hybrid.eval()

        loss_func = SoftmaxCrossEntropy()
        eval_losses = []

        correct = 0
        n_eval = 0
        loss_temp = []
        eval_batch_size = 4
        for x1, y1 in data_generator(x_test,
                                    y_test,
                                    batch_size=eval_batch_size,
                                    shuffle=True):
            x1 = x1.reshape(-1, 1, 28, 28)
            output = model_hybrid(x1)
            loss = loss_func(y1, output)
            np_loss = np.array(loss.data)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y1.argmax(1))
            correct += np.sum(np.array(mask))

            n_eval += 1
            loss_temp.append(np_loss)

        eval_losses.append(np.sum(loss_temp) / n_eval)
        print(f"Eval Accuracy: {correct / (eval_batch_size*n_eval)}")

        n_samples_show = 6
        count = 0
        _, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model_hybrid.eval()
        for x, _ in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model_hybrid(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap="gray")
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title("Predicted {}".format(np.array(pred.data)))
            count += 1
        plt.show()


    if __name__ == "__main__":

        if not os.path.exists("./result/QCNN_TL_1.model"):
            classcal_cnn_model_training()
            classical_cnn_transferlearning_predict()
        #train quantum circuits.

        quantum_cnn_transferlearning()
        #eval quantum circuits.
        quantum_cnn_transferlearning_predict()



训练集上Loss情况

.. image:: ./images/qcnn_transfer_learning_classical.png
   :width: 600 px
   :align: center

|

测试集上运行分类情况

.. image:: ./images/qcnn_transfer_learning_predict.png
   :width: 600 px
   :align: center

|

3.混合量子经典的QUnet网络模型
=======================================

图像分割（Image Segmeation）是计算机视觉研究中的一个经典难题，已经成为图像理解领域关注的一个热点，图像分割是图像分析的第一步，是计算机视觉的基础，
是图像理解的重要组成部分，同时也是图像处理中最困难的问题之一。所谓图像分割是指根据灰度、彩色、空间纹理、几何形状等特征把图像
划分成若干个互不相交的区域，使得这些特征在同一区域内表现出一致性或相似性，而在不同区域间表现出明显的不同。
简单而言就是给定一张图片，对图片上的每一个像素点分类。将不同分属不同物体的像素区域分开。 `Unet <https://arxiv.org/abs/1505.04597>`_ 是一种用于解决经典图像分割的算法。
在这里我们探索如何将经典神经网络部分量化，以创建适合量子数据的 `QUnet - Quantum Unet` 神经网络。我们将编写一个将 `pyQPanda <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ 与 `VQNet` 集成的简单示例。
QUnet主要是用于解决图像分割的技术。


数据准备
-------------------
我们将使用VOCdevkit/VOC2012官方库的数据: `VOC2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>`_ , 作为图像分割数据。
这些样本分为训练数据 training_data 和测试数据 testing_data,文件夹中包含images 和 labels。 

.. image:: ./images/Unet_data_imshow.png
   :width: 600 px
   :align: center

|

构建量子线路
---------------
在本例中，我们使用本源量子的 pyQPanda 定义了一个量子线路。将输入的3通道彩色图片数据压缩为单通道的灰度图片并进行存储，
再利用量子卷积操作对数据的特征进行提取降维操作。

.. image:: ./images/qunet_cir.png
   :width: 600 px
   :align: center

|

导入必须的库和函数

.. code-block::

    import os
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.conv import Conv2D, ConvT2D
    from pyvqnet.nn import activation as F
    from pyvqnet.nn.batch_norm import BatchNorm2d
    from pyvqnet.nn.loss import BinaryCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.dtype import *
    from pyvqnet.tensor import tensor
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyqpanda import *
    from pyvqnet.utils.storage import load_parameters, save_parameters

    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import matplotlib.pyplot as plt

    import cv2

预处理数据

.. code-block::

    #预处理数据
    class PreprocessingData:
        def __init__(self, path):
            self.path = path
            self.x_data = []
            self.y_label = []


        def processing(self):
            list_path = os.listdir((self.path+"/images"))
            for i in range(len(list_path)):

                temp_data = cv2.imread(self.path+"/images" + '/' + list_path[i], cv2.IMREAD_COLOR)
                temp_data = cv2.resize(temp_data, (128, 128))
                grayimg = cv2.cvtColor(temp_data, cv2.COLOR_BGR2GRAY)
                temp_data = grayimg.reshape(temp_data.shape[0], temp_data.shape[0], 1).astype(np.float32)
                self.x_data.append(temp_data)

                label_data = cv2.imread(self.path+"/labels" + '/' +list_path[i].split(".")[0] + ".png", cv2.IMREAD_COLOR)
                label_data = cv2.resize(label_data, (128, 128))

                label_data = cv2.cvtColor(label_data, cv2.COLOR_BGR2GRAY)
                label_data = label_data.reshape(label_data.shape[0], label_data.shape[0], 1).astype(np.int64)
                self.y_label.append(label_data)

            return self.x_data, self.y_label

        def read(self):
            self.x_data, self.y_label = self.processing()
            x_data = np.array(self.x_data)
            y_label = np.array(self.y_label)

            return x_data, y_label

    #进行量子编码的线路
    class QCNN_:
        def __init__(self, image):
            self.image = image

        def encode_cir(self, qlist, pixels):
            cir = pq.QCircuit()
            for i, pix in enumerate(pixels):
                theta = np.arctan(pix)
                phi = np.arctan(pix**2)
                cir.insert(pq.RY(qlist[i], theta))
                cir.insert(pq.RZ(qlist[i], phi))
            return cir

        def entangle_cir(self, qlist):
            k_size = len(qlist)
            cir = pq.QCircuit()
            for i in range(k_size):
                ctr = i
                ctred = i+1
                if ctred == k_size:
                    ctred = 0
                cir.insert(pq.CNOT(qlist[ctr], qlist[ctred]))
            return cir

        def qcnn_circuit(self, pixels):
            k_size = len(pixels)
            machine = pq.MPSQVM()
            machine.init_qvm()
            qlist = machine.qAlloc_many(k_size)
            cir = pq.QProg()

            cir.insert(self.encode_cir(qlist, np.array(pixels) * np.pi / 2))
            cir.insert(self.entangle_cir(qlist))

            result0 = machine.prob_run_list(cir, [qlist[0]], -1)
            result1 = machine.prob_run_list(cir, [qlist[1]], -1)
            result2 = machine.prob_run_list(cir, [qlist[2]], -1)
            result3 = machine.prob_run_list(cir, [qlist[3]], -1)

            result = [result0[-1]+result1[-1]+result2[-1]+result3[-1]]
            machine.finalize()
            return result

    def quanconv_(image):
        """Convolves the input image with many applications of the same quantum circuit."""
        out = np.zeros((64, 64, 1))
        
        for j in range(0, 128, 2):
            for k in range(0, 128, 2):
                # Process a squared 2x2 region of the image with a quantum circuit
                q_results = QCNN_(image).qcnn_circuit(
                    [
                        image[j, k, 0],
                        image[j, k + 1, 0],
                        image[j + 1, k, 0],
                        image[j + 1, k + 1, 0]
                    ]
                )
                
                for c in range(1):
                    out[j // 2, k // 2, c] = q_results[c]
        return out

    def quantum_data_preprocessing(images):
        quantum_images = []
        for _, img in enumerate(images):
            quantum_images.append(quanconv_(img))
        quantum_images = np.asarray(quantum_images)
        return quantum_images

构建混合经典量子神经网络
----------------------------

我们按照Unet网络框架，使用 `VQNet` 框架搭建经典网络部分。下采样神经网络层用于降低维度，特征提取；
上采样神经网络层，用于恢复维度；上采样与下采样层之间通过concatenate进行连接，用于特征融合。

.. image:: ./images/Unet.png
   :width: 600 px
   :align: center

|

.. code-block::

    #下采样神经网络层的定义
    class DownsampleLayer(Module):
        def __init__(self, in_ch, out_ch):
            super(DownsampleLayer, self).__init__()
            self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d1 = BatchNorm2d(out_ch)
            self.Relu1 = F.ReLu()
            self.conv2 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d2 = BatchNorm2d(out_ch)
            self.Relu2 = F.ReLu()
            self.conv3 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                                padding=(1,1))
            self.BatchNorm2d3 = BatchNorm2d(out_ch)
            self.Relu3 = F.ReLu()

        def forward(self, x):
            """
            :param x:
            :return: out(Output to deep)，out_2(enter to next level)，
            """
            x1 = self.conv1(x)
            x2 = self.BatchNorm2d1(x1)
            x3 = self.Relu1(x2)
            x4 = self.conv2(x3)
            x5 = self.BatchNorm2d2(x4)
            out = self.Relu2(x5)
            x6 = self.conv3(out)
            x7 = self.BatchNorm2d3(x6)
            out_2 = self.Relu3(x7)
            return out, out_2

    #上采样神经网络层的定义
    class UpSampleLayer(Module):
        def __init__(self, in_ch, out_ch):
            super(UpSampleLayer, self).__init__()

            self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d1 = BatchNorm2d(out_ch * 2)
            self.Relu1 = F.ReLu()
            self.conv2 = Conv2D(input_channels=out_ch * 2, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d2 = BatchNorm2d(out_ch * 2)
            self.Relu2 = F.ReLu()

            self.conv3 = ConvT2D(input_channels=out_ch * 2, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                                 padding=(1,1))
            self.BatchNorm2d3 = BatchNorm2d(out_ch)
            self.Relu3 = F.ReLu()

        def forward(self, x):
            '''
            :param x: input conv layer
            :param out: connect with UpsampleLayer
            :return:
            '''
            x = self.conv1(x)
            x = self.BatchNorm2d1(x)
            x = self.Relu1(x)
            x = self.conv2(x)
            x = self.BatchNorm2d2(x)
            x = self.Relu2(x)
            x = self.conv3(x)
            x = self.BatchNorm2d3(x)
            x_out = self.Relu3(x)
            return x_out

    #Unet整体网络架构
    class UNet(Module):
        def __init__(self):
            super(UNet, self).__init__()
            out_channels = [2 ** (i + 4) for i in range(5)]

            # DownSampleLayer
            self.d1 = DownsampleLayer(1, out_channels[0])  # 3-64
            self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
            self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
            self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
            # UpSampleLayer
            self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
            self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
            self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
            self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
            # output
            self.conv1 = Conv2D(input_channels=out_channels[1], output_channels=out_channels[0], kernel_size=(3, 3),
                                stride=(1, 1), padding="same")
            self.BatchNorm2d1 = BatchNorm2d(out_channels[0])
            self.Relu1 = F.ReLu()
            self.conv2 = Conv2D(input_channels=out_channels[0], output_channels=out_channels[0], kernel_size=(3, 3),
                                stride=(1, 1), padding="same")
            self.BatchNorm2d2 = BatchNorm2d(out_channels[0])
            self.Relu2 = F.ReLu()
            self.conv3 = Conv2D(input_channels=out_channels[0], output_channels=1, kernel_size=(3, 3),
                                stride=(1, 1), padding="same")
            self.Sigmoid = F.Sigmoid()

        def forward(self, x):
            out_1, out1 = self.d1(x)
            out_2, out2 = self.d2(out1)
            out_3, out3 = self.d3(out2)
            out_4, out4 = self.d4(out3)

            out5 = self.u1(out4)
            out5_pad_out4 = tensor.pad2d(out5, (1, 0, 1, 0), 0)
            cat_out5 = tensor.concatenate([out5_pad_out4, out_4], axis=1)

            out6 = self.u2(cat_out5)
            out6_pad_out_3 = tensor.pad2d(out6, (1, 0, 1, 0), 0)
            cat_out6 = tensor.concatenate([out6_pad_out_3, out_3], axis=1)

            out7 = self.u3(cat_out6)
            out7_pad_out_2 = tensor.pad2d(out7, (1, 0, 1, 0), 0)
            cat_out7 = tensor.concatenate([out7_pad_out_2, out_2], axis=1)

            out8 = self.u4(cat_out7)
            out8_pad_out_1 = tensor.pad2d(out8, (1, 0, 1, 0), 0)
            cat_out8 = tensor.concatenate([out8_pad_out_1, out_1], axis=1)

            out = self.conv1(cat_out8)
            out = self.BatchNorm2d1(out)
            out = self.Relu1(out)
            out = self.conv2(out)
            out = self.BatchNorm2d2(out)
            out = self.Relu2(out)
            out = self.conv3(out)
            out = self.Sigmoid(out)
            return out

训练和模型保存
--------------------

通过上面代码示例，我们已经定义了模型。与经典神经网络模型训练类似， 我们还需要做的是实例化该模型，
定义损失函数以及优化器以及定义整个训练测试流程。 对于形如下图的混合神经网络模型，我们通过循环输入数据前向计算损失值，
并在反向计算中自动计算出各个待训练参数的梯度，并使用优化器进行参数优化，直到迭代次数满足预设值。
我们这里使用前面下载的VOC2012数据中选取100张作为训练集，10张作为测试集。训练集目录指定为 `path0`,测试集目录指定为 `path1`。
其中，图像以及其对应的标签图像都经过了量子卷积模块 ``quantum_data_preprocessing`` 进行预处理，我们Unet的训练目标是使得同时经过量子线路预处理的图像和标签尽可能贴近。
如果已经进行了量子化数据预处理，可以设置 ``PREPROCESS`` 为False。

.. code-block::

    PREPROCESS = True

    class MyDataset():
        def __init__(self, x_data, x_label):
            self.x_set = x_data
            self.label = x_label

        def __getitem__(self, item):
            img, target = self.x_set[item], self.label[item]
            img_np = np.uint8(img).transpose(2, 0, 1)
            target_np = np.uint8(target).transpose(2, 0, 1)

            img = img_np
            target = target_np
            return img, target

        def __len__(self):
            return len(self.x_set)

    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass
    if not os.path.exists("./Intermediate_results"):
        os.makedirs("./Intermediate_results")
    else:
        pass

    # prepare train/test data and label
    path0 = 'training_data'
    path1 = 'testing_data'
    train_images, train_labels = PreprocessingData(path0).read()
    test_images, test_labels = PreprocessingData(path1).read()

    print('train: ', train_images.shape, '\ntest: ', test_images.shape)
    print('train: ', train_labels.shape, '\ntest: ', test_labels.shape)
    train_images = train_images / 255
    test_images = test_images / 255

    # use quantum encoder to preprocess data

    
    
    if PREPROCESS == True:
        print("Quantum pre-processing of train images:")
        q_train_images = quantum_data_preprocessing(train_images)
        q_test_images = quantum_data_preprocessing(test_images)
        q_train_label = quantum_data_preprocessing(train_labels)
        q_test_label = quantum_data_preprocessing(test_labels)

        # Save pre-processed images
        print('Quantum Data Saving...')
        np.save("./result/q_train.npy", q_train_images)
        np.save("./result/q_test.npy", q_test_images)
        np.save("./result/q_train_label.npy", q_train_label)
        np.save("./result/q_test_label.npy", q_test_label)
        print('Quantum Data Saving Over!')

    # loading quantum data
    SAVE_PATH = "./result/"
    train_x = np.load(SAVE_PATH + "q_train.npy")
    train_labels = np.load(SAVE_PATH + "q_train_label.npy")
    test_x = np.load(SAVE_PATH + "q_test.npy")
    test_labels = np.load(SAVE_PATH + "q_test_label.npy")

    train_x = train_x.astype(np.uint8)
    test_x = test_x.astype(np.uint8)
    train_labels = train_labels.astype(np.uint8)
    test_labels = test_labels.astype(np.uint8)
    train_y = train_labels
    test_y = test_labels

    trainset = MyDataset(train_x, train_y)
    testset = MyDataset(test_x, test_y)
    x_train = []
    y_label = []
    model = UNet()
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_func = BinaryCrossEntropy()
    epochs = 200

    loss_list = []
    SAVE_FLAG = True
    temp_loss = 0
    file = open("./result/result.txt", 'w').close()
    for epoch in range(1, epochs):
        total_loss = []
        model.train()
        for i, (x, y) in enumerate(trainset):
            x_img = QTensor(x, dtype=kfloat32)
            x_img_Qtensor = tensor.unsqueeze(x_img, 0)
            y_img = QTensor(y, dtype=kfloat32)
            y_img_Qtensor = tensor.unsqueeze(y_img, 0)
            optimizer.zero_grad()
            img_out = model(x_img_Qtensor)

            print(f"=========={epoch}==================")
            loss = loss_func(y_img_Qtensor, img_out)  # target output
            if i == 1:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.title("predict")
                img_out_tensor = tensor.squeeze(img_out, 0)

                if matplotlib.__version__ >= '3.4.2':
                    plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
                else:
                    plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
                plt.subplot(1, 2, 2)
                plt.title("label")
                y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
                if matplotlib.__version__ >= '3.4.2':
                    plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
                else:
                    plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))

                plt.savefig("./Intermediate_results/" + str(epoch) + "_" + str(i) + ".jpg")

            loss_data = np.array(loss.data)
            print("{} - {} loss_data: {}".format(epoch, i, loss_data))
            loss.backward()
            optimizer._step()
            total_loss.append(loss_data)

        loss_list.append(np.sum(total_loss) / len(total_loss))
        out_read = open("./result/result.txt", 'a')
        out_read.write(str(loss_list[-1]))
        out_read.write(str("\n"))
        out_read.close()
        print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
        if SAVE_FLAG:
            temp_loss = loss_list[-1]
            save_parameters(model.state_dict(), "./result/Q-Unet_End.model")
            SAVE_FLAG = False
        else:
            if temp_loss > loss_list[-1]:
                temp_loss = loss_list[-1]
                save_parameters(model.state_dict(), "./result/Q-Unet_End.model")


数据可视化
---------------

训练数据的损失函数曲线显示保存，以及测试数据结果保存。

.. code-block::

    out_read = open("./result/result.txt", 'r')
    plt.figure()
    lines_read = out_read.readlines()
    data_read = []
    for line in lines_read:
        float_line = float(line)
        data_read.append(float_line)
    out_read.close()
    plt.plot(data_read)
    plt.title('Unet Training')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.savefig("./result/traing_loss.jpg")

    modela = load_parameters("./result/Q-Unet_End.model")
    print("----------------PREDICT-------------")
    model.load_state_dict(modela)
    model.eval()

    for i, (x1, y1) in enumerate(testset):
        x_img = QTensor(x1, dtype=kfloat32)
        x_img_Qtensor = tensor.unsqueeze(x_img, 0)
        y_img = QTensor(y1, dtype=kfloat32)
        y_img_Qtensor = tensor.unsqueeze(y_img, 0)
        img_out = model(x_img_Qtensor)
        loss = loss_func(y_img_Qtensor, img_out)
        loss_data = np.array(loss.data)
        print("{} loss_eval: {}".format(i, loss_data))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("predict")
        img_out_tensor = tensor.squeeze(img_out, 0)
        if matplotlib.__version__ >= '3.4.2':
            plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
        else:
            plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
        plt.subplot(1, 2, 2)
        plt.title("label")
        y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
        if matplotlib.__version__ >= '3.4.2':
            plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
        else:
            plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))
        plt.savefig("./result/eval_" + str(i) + "_1" + ".jpg")
    print("end!")

训练集上Loss情况

.. image:: ./images/qunet_train_loss.png
   :width: 600 px
   :align: center

|

可视化运行情况

.. image:: ./images/qunet_eval_1.jpg
   :width: 600 px
   :align: center

.. image:: ./images/qunet_eval_2.jpg
   :width: 600 px
   :align: center

.. image:: ./images/qunet_eval_3.jpg
   :width: 600 px
   :align: center

|


4.混合量子经典的QCNN网络模型
==================================

我们介绍并分析了一种由卷积神经网络驱动的新型量子机器学习模型。`Quantum Convolutional Neural Networks <https://arxiv.org/pdf/1810.03787.pdf>`_ 是一种用于解决经典图像分类的算法。
我们将编写一个将 `pyQPanda <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ 与 `VQNet` 集成的简单示例。



构建混合经典量子神经网络
---------------------------

.. code-block::

    import random
    import numpy as np
    import pyqpanda as pq
    from qiskit.utils import algorithm_globals
    from pyvqnet.qnn.measure import expval
    from scipy.optimize import minimize
    from abc import abstractmethod
    from sklearn.model_selection import train_test_split
    from sklearn.base import ClassifierMixin
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  #pylint:disable=bare-except
        print("Can not use matplot TkAgg")

    def network(input_data, weights):
        datasets = np.array(input_data)
        number_of_qubits = 8
        circuit_data = []
        for x in datasets:
            machine = pq.CPUQVM()
            machine.init_qvm()
            qlist = machine.qAlloc_many(number_of_qubits)
            clist = machine.cAlloc_many(number_of_qubits)

            circuit = pq.QCircuit()
            circuit.insert(build_VQNet_cir(qlist, number_of_qubits, x, weights))

            prog = pq.QProg()
            prog.insert(circuit)

            pauli_dict = {'Z7': 1}
            exp2 = expval(machine, prog, pauli_dict, qlist)
            circuit_data.append([exp2])

        output = np.array(circuit_data).reshape([-1, 1])

        return output

    class ObjectiveFunction:
        # pylint: disable=invalid-name
        def __init__(
            self, X: np.ndarray, y: np.ndarray, neural_network
        ) -> None:
            """
            Args:
                X: The input data.
                y: The target values.
                neural_network: An instance of an quantum neural network to be used by this
                    objective function.
            """
            super().__init__()
            self._X = X
            self._num_samples = X.shape[0]
            self._y = y
            self._neural_network = neural_network
            self._last_forward_weights = None
            self._last_forward = None
            self._loss_result = []

        @abstractmethod
        def objective(self, weights: np.ndarray) -> float:
            """Computes the value of this objective function given weights.

            Args:
                weights: an array of weights to be used in the objective function.

            Returns:
                Value of the function.
            """
            raise NotImplementedError

        @abstractmethod
        def gradient(self, weights: np.ndarray) -> np.ndarray:
            """Computes gradients of this objective function given weights.

            Args:
                weights: an array of weights to be used in the objective function.

            Returns:
                Gradients of the function.
            """
            raise NotImplementedError

        def _neural_network_forward(self, weights: np.ndarray):
            """
            Computes and caches the results of the forward pass. Cached values may be re-used in
            gradient computation.

            Args:
                weights: an array of weights to be used in the forward pass.

            Returns:
                The result of the neural network.
            """
            # if we get the same weights, we don't compute the forward pass again.
            if self._last_forward_weights is None or (
                not np.all(np.isclose(weights, self._last_forward_weights))
            ):
                # compute forward and cache the results for re-use in backward
                self._last_forward = self._neural_network(self._X, weights)
                # a copy avoids keeping a reference to the same array, so we are sure we have
                # different arrays on the next iteration.
                self._last_forward_weights = np.copy(weights)
            return self._last_forward

        def _neural_network_backward(self, weights: np.ndarray):

            datasetsy = np.array(self._y)
            datasetsy = datasetsy.reshape([-1, 1])

            weights1 = weights + np.pi / 2
            weights2 = weights - np.pi / 2
            exp1 = self._neural_network_forward(weights1)
            exp2 = self._neural_network_forward(weights2)

            circuit_grad = (exp1 - exp2) / 2
            output = self._neural_network_forward(weights)
            result = 2 * (output - datasetsy)
            grad = result[:, 0] @ circuit_grad
            grad = grad.reshape(1, -1) / self._X.shape[0]
            return grad

    class BinaryObjectiveFunction(ObjectiveFunction):
        """An objective function for binary representation of the output,
        e.g. classes of ``-1`` and ``+1``."""

        def objective(self, weights: np.ndarray) -> float:
            # predict is of shape (N, 1), where N is a number of samples
            output = self._neural_network_forward(weights)
            datasetsy = np.array(self._y).reshape([-1, 1])
            if len(output.shape) <= 1:
                loss_data = (output - datasetsy) ** 2
            else:
                loss_data = np.linalg.norm(output - datasetsy, axis=tuple(range(1, len(output.shape)))) ** 2

            result = float(np.sum(loss_data) / self._X.shape[0])
            self._loss_result.append(result)
            return np.average(np.array(result))

        def gradient(self, weights: np.ndarray) -> np.ndarray:
            # weight grad is of shape (N, 1, num_weights)
            weight_grad = self._neural_network_backward(weights)
            return weight_grad

        def loss_value(self):
            # get loss curve data
            return self._loss_result

    class VQNet_QCNN(ClassifierMixin):
        def __init__(self, forward_func=None, init_weight=None, optimizer=None):
            """
            forward_func: An instance of an quantum neural network.
            init_weight: Initial weight for the optimizer to start from.
            optimizer: An instance of an optimizer to be used in training. When `None` defaults to L-BFGS-B.
            """
            self.init_weight = init_weight
            self._fit_result = None
            self._forward_func = forward_func
            self._optimizer = optimizer
            self._loss_result = []


        def fit(self, X: np.ndarray, y: np.ndarray): # pylint: disable=invalid-name
            """
            Function operation to solve the optimal solution.
            """
            function = BinaryObjectiveFunction(X, y, self._forward_func)
            self._fit_result = minimize(fun=function.objective, x0=self.init_weight,
                                        method=self._optimizer, jac=function.gradient)
            self._loss_result.append(function.loss_value())
            return self._fit_result.x

        def predict(self, X:np.ndarray):
            """
            Predict
            """
            if self._fit_result is None:
                raise Exception("Model needs to be fit to some training data first!")

            return np.sign(self._forward_func(X, self._fit_result.x))

        # pylint: disable=invalid-name
        def score(
            self, X: np.ndarray, y: np.ndarray, sample_weight=None
        ) -> float:
            """
            Calculate the score.
            """
            result_score = ClassifierMixin.score(self, X, y, sample_weight)
            return result_score

        def get_loss_value(self):
            """
            Get loss curve data.
            """
            result = self._loss_result
            return result

    def ZFeatureMap_VQNet(qlist, n_qbits, weights):
        r"""The first order Pauli Z-evolution circuit.
        On 3 qubits and with 2 repetitions the circuit is represented by:

        .. parsed-literal::
            ┌───┐┌──────────────┐┌───┐┌──────────────┐
            ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
            ├───┤├──────────────┤├───┤├──────────────┤
            ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
            ├───┤├──────────────┤├───┤├──────────────┤
            ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
            └───┘└──────────────┘└───┘└──────────────┘

        """
        circuit = pq.QCircuit()
        for i in range(n_qbits):
            circuit.insert(pq.H(qlist[i]))
            circuit.insert(pq.U1(qlist[i], 2.0*weights[i]))
            circuit.insert(pq.H(qlist[i]))
            circuit.insert(pq.U1(qlist[i], 2.0*weights[i]))
        return circuit

    def conv_circuit_VQNet(qlist, n_qbits, weights):
        """
        Quantum Convolutional
        """
        circuit = pq.QCircuit()
        circuit.insert(pq.RZ(qlist[1], -np.pi))
        circuit.insert(pq.CNOT(qlist[1], qlist[0]))
        circuit.insert(pq.RZ(qlist[0], weights[0]))
        circuit.insert(pq.RY(qlist[1], weights[1]))
        circuit.insert(pq.CNOT(qlist[0], qlist[1]))
        circuit.insert(pq.RY(qlist[1], weights[2]))
        circuit.insert(pq.CNOT(qlist[1], qlist[0]))
        circuit.insert(pq.RZ(qlist[0], np.pi))
        return circuit

    def conv_layer_VQNet(qlist, n_qbits, weights):
        """
         Define the Convolutional Layers of our QCNN
        """
        qubits = list(range(n_qbits))
        param_index = 0
        cir = pq.QCircuit()

        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qlist_q = [qlist[q1], qlist[q2]]
            cir.insert(conv_circuit_VQNet(qlist_q, n_qbits, weights[param_index:(param_index + 3)]))
            cir.insert(pq.BARRIER(qlist))
            param_index += 3

        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qlist_q = [qlist[q1], qlist[q2]]
            cir.insert(conv_circuit_VQNet(qlist_q, n_qbits, weights[param_index:(param_index + 3)]))
            cir.insert(pq.BARRIER(qlist))
            param_index += 3

        return cir

    def pool_circuit_VQNet(qlist, n_qbits, weights):
        """
        Quantum Pool
        """
        cir = pq.QCircuit()
        cir.insert(pq.RZ(qlist[1], -np.pi))
        cir.insert(pq.CNOT(qlist[1], qlist[0]))
        cir.insert(pq.RZ(qlist[0], weights[0]))
        cir.insert(pq.RY(qlist[1], weights[1]))
        cir.insert(pq.CNOT(qlist[0], qlist[1]))
        cir.insert(pq.RY(qlist[1], weights[2]))

        return cir

    def pool_layer_VQNet(sources, sinks, qlist, n_qbits, weights):
        """
        Create a QCNN Pooling Layer
        """
        num_qubits = len(sources) + len(sinks)
        if num_qubits != n_qbits:
            raise ValueError("the number of qubits is error!")
        param_index = 0
        cir = pq.QCircuit()
        for source, sink in zip(sources, sinks):
            qlist_q = [qlist[source], qlist[sink]]
            cir.insert(pool_circuit_VQNet(qlist_q, n_qbits, weights[param_index:(param_index + 3)]))
            cir.insert(pq.BARRIER(qlist))
            param_index += 3

        return cir

    def build_VQNet_cir(qubits, n_qbits, input_data, weights):
        """
        Create a VQNet Quantum Convolutional Neural Network.
        """
        circuit = pq.QCircuit()
        circuit.insert(ZFeatureMap_VQNet(qubits, n_qbits, input_data))
        circuit.insert(conv_layer_VQNet(qubits, n_qbits, weights))
        sources = [0, 1, 2, 3]
        sinks = [4, 5, 6, 7]
        circuit.insert(pool_layer_VQNet(sources, sinks, qubits, n_qbits, weights))

        qubits_select_4_8 = qubits[4:8]
        n_qbits_len_4_8 = len(qubits_select_4_8)
        circuit.insert(conv_layer_VQNet(qubits_select_4_8, n_qbits_len_4_8, weights))
        sources = [0, 1]
        sinks = [2, 3]
        circuit.insert(pool_layer_VQNet(sources, sinks, qubits_select_4_8, n_qbits_len_4_8, weights))

        qubits_select_6_8 = qubits[6:8]
        n_qbits_len_6_8 = len(qubits_select_6_8)
        circuit.insert(conv_layer_VQNet(qubits_select_6_8, n_qbits_len_6_8, weights))
        sources = [0]
        sinks = [1]
        circuit.insert(pool_layer_VQNet(sources, sinks, qubits_select_6_8, n_qbits_len_6_8, weights))

        return circuit

    def generate_dataset(num_images):
        images = []
        labels = []
        hor_array = np.zeros((6, 8))
        ver_array = np.zeros((4, 8))

        j = 0
        for i in range(0, 7):
            if i != 3:
                hor_array[j][i] = np.pi / 2
                hor_array[j][i + 1] = np.pi / 2
                j += 1

        j = 0
        for i in range(0, 4):
            ver_array[j][i] = np.pi / 2
            ver_array[j][i + 4] = np.pi / 2
            j += 1

        for n in range(num_images):
            rng = algorithm_globals.random.integers(0, 2)
            if rng == 0:
                labels.append(-1)
                random_image = algorithm_globals.random.integers(0, 6)
                images.append(np.array(hor_array[random_image]))
            elif rng == 1:
                labels.append(1)
                random_image = algorithm_globals.random.integers(0, 4)
                images.append(np.array(ver_array[random_image]))

            # Create noise
            for i in range(8):
                if images[-1][i] == 0:
                    images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
        return images, labels

    def vqnet_qcnn_test():
        algorithm_globals.random_seed = 12345
        random.seed(24)

        images, labels = generate_dataset(50)

        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.3
        )

        fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
        for i in range(4):
            ax[i // 2, i % 2].imshow(
                train_images[i].reshape(2, 4),  # Change back to 2 by 4
                aspect="equal",
            )
        plt.subplots_adjust(wspace=0.1, hspace=0.025)
        # plt.show()

        _initial_point = algorithm_globals.random.random(63)
        qcnn_compute = VQNet_QCNN(forward_func=network, init_weight=_initial_point, optimizer="COBYLA")
        x = np.asarray(train_images)
        y = np.asarray(train_labels)
        result = qcnn_compute.fit(x, y)
        print(len(result))

        # score classifier
        print(f"Accuracy from the train data : {np.round(100 * qcnn_compute.score(x, y), 2)}%")

        # show the loss
        plt.figure()
        loss_value = qcnn_compute.get_loss_value()[0]
        loss_len = len(loss_value)
        x = np.arange(1, loss_len+1)
        plt.plot(x, loss_value, "b")
        plt.show()


    if __name__ == "__main__":
        vqnet_qcnn_test()


数据结果
------------------------

训练数据的损失函数曲线显示保存，以及测试数据结果保存。
训练集上Loss情况

.. image:: ./images/qcnn_vqnet.png
   :width: 600 px
   :align: center

|

可视化运行情况

.. image:: ./images/qcnn_vqnet_result.png
   :width: 600 px
   :align: center


|


5.混合量子经典的QMLP网络模型
==================================

我们介绍并分析了提出了一种量子多层感知器 (QMLP) 架构，其特点是具有容错输入嵌入、丰富的非线性和带有参数化双量子比特纠缠门的增强变分电路模拟。`QMLP: An Error-Tolerant Nonlinear Quantum MLP Architecture using Parameterized Two-Qubit Gates <https://arxiv.org/pdf/2206.01345.pdf>`_ 。
我们将编写一个将 `pyQPanda <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ 与 `VQNet` 集成的简单示例。



构建混合经典量子神经网络
-----------------------------

.. code-block::

    import os
    import gzip
    import struct
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError, CrossEntropyLoss
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer, QuantumLayerMultiProcess
    from pyvqnet.nn.pooling import AvgPool2D
    from pyvqnet.nn.linear import Linear
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    import matplotlib
    from matplotlib import pyplot as plt
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
        print("Can not use matplot TkAgg")

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = "http://yann.lecun.com/exdb/mnist/"
    key_file = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_label": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_label": "t10k-labels-idx1-ubyte.gz"
    }


    def _download(dataset_dir, file_name):
        """
        Download mnist data if needed.
        """
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                file_path_ungz = file_path_ungz.replace("-idx", ".idx")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
        print("Done")

    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir, v)

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):
        """
        load mnist data
        """
        from array import array as pyarray
        download_mnist(path)
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
        images = np.zeros((num, rows, cols))
        labels = np.zeros((num, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                     cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def data_select(train_num, test_num):
        """
        Select data from mnist dataset.
        """
        x_train, y_train = load_mnist("training_data")  #pylint:disable=redefined-outer-name
        x_test, y_test = load_mnist("testing_data")  #pylint:disable=redefined-outer-name
        idx_train = np.append(
            np.where(y_train == 0)[0][:train_num],
            np.where(y_train == 1)[0][:train_num])

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]

        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)

        # Test Leaving only labels 0 and 1
        idx_test = np.append(
            np.where(y_test == 0)[0][:test_num],
            np.where(y_test == 1)[0][:test_num])

        x_test = x_test[idx_test]
        y_test = y_test[idx_test]
        x_test = x_test / 255
        y_test = np.eye(2)[y_test].reshape(-1, 2)

        return x_train, y_train, x_test, y_test

    def RotCircuit(para, qlist):
        r"""

        Arbitrary single qubit rotation.Number of qlist should be 1,and number of parameters should
        be 3

        .. math::

            R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
            e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
            e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
            \end{bmatrix}.


        :param para: numpy array which represents paramters [\phi, \theta, \omega]
        :param qlist: qubits allocated by pyQpanda.qAlloc_many()
        :return: quantum circuits

        Example::

            m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
            m_clist = m_machine.cAlloc_many(2)
            m_prog = pq.QProg()
            m_qlist = m_machine.qAlloc_many(1)
            param = np.array([3,4,5])
            c = RotCircuit(param,m_qlist)
            print(c)
            pq.destroy_quantum_machine(m_machine)

        """
        if isinstance(para, QTensor):
            para = QTensor._to_numpy(para)
        if para.ndim > 1:
            raise ValueError(" dim of paramters in Rot should be 1")
        if para.shape[0] != 3:
            raise ValueError(" numbers of paramters in Rot should be 3")

        cir = pq.QCircuit()
        cir.insert(pq.RZ(qlist, para[2]))
        cir.insert(pq.RY(qlist, para[1]))
        cir.insert(pq.RZ(qlist, para[0]))

        return cir

    def build_RotCircuit(qubits, weights):
        cir = pq.QCircuit()
        cir.insert(RotCircuit(weights[0:3], qubits[0]))
        cir.insert(RotCircuit(weights[3:6], qubits[1]))
        cir.insert(RotCircuit(weights[6:9], qubits[2]))
        cir.insert(RotCircuit(weights[9:12], qubits[3]))
        cir.insert(RotCircuit(weights[12:15], qubits[4]))
        cir.insert(RotCircuit(weights[15:18], qubits[5]))
        cir.insert(RotCircuit(weights[18:21], qubits[6]))
        cir.insert(RotCircuit(weights[21:24], qubits[7]))
        cir.insert(RotCircuit(weights[24:27], qubits[8]))
        cir.insert(RotCircuit(weights[27:30], qubits[9]))
        cir.insert(RotCircuit(weights[30:33], qubits[10]))
        cir.insert(RotCircuit(weights[33:36], qubits[11]))
        cir.insert(RotCircuit(weights[36:39], qubits[12]))
        cir.insert(RotCircuit(weights[39:42], qubits[13]))
        cir.insert(RotCircuit(weights[42:45], qubits[14]))
        cir.insert(RotCircuit(weights[45:48], qubits[15]))

        return cir

    def CRXCircuit(para, control_qlists, rot_qlists):
        cir = pq.QCircuit()
        cir.insert(pq.RX(rot_qlists, para))
        cir.set_control(control_qlists)
        return cir

    def build_CRotCircuit(qubits, weights):
        cir = pq.QCircuit()
        cir.insert(CRXCircuit(weights[0], qubits[0], qubits[1]))
        cir.insert(CRXCircuit(weights[1], qubits[1], qubits[2]))
        cir.insert(CRXCircuit(weights[2], qubits[2], qubits[3]))
        cir.insert(CRXCircuit(weights[3], qubits[3], qubits[4]))
        cir.insert(CRXCircuit(weights[4], qubits[4], qubits[5]))
        cir.insert(CRXCircuit(weights[5], qubits[5], qubits[6]))
        cir.insert(CRXCircuit(weights[6], qubits[6], qubits[7]))
        cir.insert(CRXCircuit(weights[7], qubits[7], qubits[8]))
        cir.insert(CRXCircuit(weights[8], qubits[8], qubits[9]))
        cir.insert(CRXCircuit(weights[9], qubits[9], qubits[10]))
        cir.insert(CRXCircuit(weights[10], qubits[10], qubits[11]))
        cir.insert(CRXCircuit(weights[11], qubits[11], qubits[12]))
        cir.insert(CRXCircuit(weights[12], qubits[12], qubits[13]))
        cir.insert(CRXCircuit(weights[13], qubits[13], qubits[14]))
        cir.insert(CRXCircuit(weights[14], qubits[14], qubits[15]))
        cir.insert(CRXCircuit(weights[15], qubits[15], qubits[0]))

        return cir


    def build_qmlp_circuit(x, weights, qubits, clist, machine):
        cir = pq.QCircuit()
        num_qubits = len(qubits)
        for i in range(num_qubits):
            cir.insert(pq.RX(qubits[i], x[i]))

        cir.insert(build_RotCircuit(qubits, weights[0:48]))
        cir.insert(build_CRotCircuit(qubits, weights[48:64]))

        for i in range(num_qubits):
            cir.insert(pq.RX(qubits[i], x[i]))

        cir.insert(build_RotCircuit(qubits, weights[64:112]))
        cir.insert(build_CRotCircuit(qubits, weights[112:128]))

        prog = pq.QProg()
        prog.insert(cir)
        # print(prog)
        # exit()

        exp_vals = []
        for position in range(num_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)

        return exp_vals

    def build_multiprocess_qmlp_circuit(x, weights, num_qubits, num_clist):
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)
        cir = pq.QCircuit()
        for i in range(num_qubits):
            cir.insert(pq.RX(qubits[i], x[i]))

        cir.insert(build_RotCircuit(qubits, weights[0:48]))
        cir.insert(build_CRotCircuit(qubits, weights[48:64]))

        for i in range(num_qubits):
            cir.insert(pq.RX(qubits[i], x[i]))

        cir.insert(build_RotCircuit(qubits, weights[64:112]))
        cir.insert(build_CRotCircuit(qubits, weights[112:128]))

        prog = pq.QProg()
        prog.insert(cir)
        # print(prog)
        # exit()

        exp_vals = []
        for position in range(num_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)

        return exp_vals

    class QMLPModel(Module):
        def __init__(self):
            super(QMLPModel, self).__init__()
            self.ave_pool2d = AvgPool2D([7, 7], [7, 7], "valid")
            # self.quantum_circuit = QuantumLayer(build_qmlp_circuit, 128, "CPU", 16, diff_method="finite_diff")
            self.quantum_circuit = QuantumLayerMultiProcess(build_multiprocess_qmlp_circuit, 128, 
                                                            16, 1, diff_method="finite_diff")
            self.linear = Linear(16, 10)

        def forward(self, x):
            bsz = x.shape[0]
            x = self.ave_pool2d(x)
            input_data = x.reshape([bsz, 16])
            quanutum_result = self.quantum_circuit(input_data)
            result = self.linear(quanutum_result)
            return result

    def vqnet_test_QMLPModel():
        # train num=1000, test_num=100
        # x_train, y_train, x_test, y_test = data_select(1000, 100)

        train_size = 1000
        eval_size = 100
        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = QMLPModel()
        optimizer = Adam(model.parameters(), lr=0.005)
        loss_func = CrossEntropyLoss()
        loss_list = []
        epochs = 30
        for epoch in range(1, epochs):
            total_loss = []

            correct = 0
            n_train = 0
            for x, y in data_generator(x_train,
                                       y_train,
                                       batch_size=16,
                                       shuffle=True):

                x = x.reshape(-1, 1, 28, 28)
                optimizer.zero_grad()
                # Forward pass
                output = model(x)
                # Calculating loss
                loss = loss_func(y, output)
                loss_np = np.array(loss.data)
                print("loss: ", loss_np)
                np_output = np.array(output.data, copy=False)

                temp_out = np_output.argmax(axis=1)
                temp_output = np.zeros((temp_out.size, 10))
                temp_output[np.arange(temp_out.size), temp_out] = 1
                temp_maks = (temp_output == y)

                correct += np.sum(np.array(temp_maks))
                n_train += 160

                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer._step()
                total_loss.append(loss_np)
            print("##########################")
            print(f"Train Accuracy: {correct / n_train}")
            loss_list.append(np.sum(total_loss) / len(total_loss))
            # train_acc_list.append(correct / n_train)
            print("epoch: ", epoch)
            # print(100. * (epoch + 1) / epochs)
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

    if __name__ == "__main__":

        vqnet_test_QMLPModel()



数据结果
-----------------------

训练数据的损失函数曲线显示保存，以及测试数据结果保存。
训练集上Loss情况

.. image:: ./images/QMLP.png
   :width: 600 px
   :align: center

|

6.混合量子经典的QDRL网络模型
====================================


我们介绍并分析了提出了一种量子强化学习网络 (QDRL) ，其特点将经典的深度强化学习算法（如经验回放和目标网络）重塑为变分量子电路的表示。
此外，与经典神经网络相比，我们使用量子信息编码方案来减少模型参数的数量。 `QDRL: Variational Quantum Circuits for Deep Reinforcement Learning <https://arxiv.org/pdf/1907.00397.pdf>`_ 。
我们将编写一个将 `pyQPanda <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ 与 `VQNet` 集成的简单示例。



构建混合经典量子神经网络
---------------------------

需要安装 ``gym`` == 0.23.0 , ``pygame`` == 2.1.2 。

.. code-block::


    import numpy as np
    import random
    import gym
    import time
    from matplotlib import animation
    import pyqpanda as pq
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor,kfloat32
    from pyvqnet.qnn.quantumlayer import QuantumLayerMultiProcess
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.measure import expval
    from pyvqnet._core import Tensor as CoreTensor
    import matplotlib
    from matplotlib import pyplot as plt
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
        print("Can not use matplot TkAgg")
    def display_frames_as_gif(frames, c_index):
        patch = plt.imshow(frames[0])
        plt.axis('off')
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        name_result = "./result_"+str(c_index)+".gif"
        anim.save(name_result, writer='pillow', fps=10)
    CIRCUIT_SIZE = 4
    MAX_ITERATIONS = 50
    MAX_STEPS = 250
    BATCHSIZE = 5
    TARGET_MAX = 20
    GAMMA = 0.99
    STATE_T = 0
    ACTION = 1
    REWARD = 2
    STATE_NT = 3
    DONE = 4
    def RotCircuit(para, qlist):
        r"""
        Arbitrary single qubit rotation.Number of qlist should be 1,and number of parameters should
        be 3
        .. math::
            R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
            e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
            e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
            \end{bmatrix}.
        :param para: numpy array which represents paramters [\phi, \theta, \omega]
        :param qlist: qubits allocated by pyQpanda.qAlloc_many()
        :return: quantum circuits
        Example::
            m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
            m_clist = m_machine.cAlloc_many(2)
            m_prog = pq.QProg()
            m_qlist = m_machine.qAlloc_many(1)
            param = np.array([3,4,5])
            c = RotCircuit(param,m_qlist)
            print(c)
            pq.destroy_quantum_machine(m_machine)
        """
        if isinstance(para, QTensor):
            para = QTensor._to_numpy(para)
        if para.ndim > 1:
            raise ValueError(" dim of paramters in Rot should be 1")
        if para.shape[0] != 3:
            raise ValueError(" numbers of paramters in Rot should be 3")
        cir = pq.QCircuit()
        cir.insert(pq.RZ(qlist, para[2]))
        cir.insert(pq.RY(qlist, para[1]))
        cir.insert(pq.RZ(qlist, para[0]))
        return cir
    def layer_circuit(qubits, weights):
        cir = pq.QCircuit()
        # Entanglement block
        cir.insert(pq.CNOT(qubits[0], qubits[1]))
        cir.insert(pq.CNOT(qubits[1], qubits[2]))
        cir.insert(pq.CNOT(qubits[2], qubits[3]))
        # u3 gate
        cir.insert(RotCircuit(weights[0], qubits[0]))  # weights shape = [4, 3]
        cir.insert(RotCircuit(weights[1], qubits[1]))
        cir.insert(RotCircuit(weights[2], qubits[2]))
        cir.insert(RotCircuit(weights[3], qubits[3]))
        return cir
    def encoder(encodings):
        encodings = int(encodings[0])
        return [i for i, b in enumerate(f'{encodings:0{CIRCUIT_SIZE}b}') if b == '1']
    def build_qc(x, weights, num_qubits, num_clist):
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)
        cir = pq.QCircuit()
        if x:
            wires = encoder(x)
            for wire in wires:
                cir.insert(pq.RX(qubits[wire], np.pi))
                cir.insert(pq.RZ(qubits[wire], np.pi))
        # parameter number = 24
        weights = weights.reshape([2, 4, 3])
        # layer wise
        for w in weights:
            cir.insert(layer_circuit(qubits, w))
        prog = pq.QProg()
        prog.insert(cir)
        exp_vals = []
        for position in range(num_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)
        return exp_vals
    class DRLModel(Module):
        def __init__(self):
            super(DRLModel, self).__init__()
            self.quantum_circuit = QuantumLayerMultiProcess(build_qc, 24,  
                                                            4, 1, diff_method="finite_diff")
        def forward(self, x):
            quanutum_result = self.quantum_circuit(x)
            return quanutum_result
    env = gym.make("FrozenLake-v1", is_slippery = False, map_name = '4x4')
    state = env.reset()
    n_layers = 2
    n_qubits = 4
    targ_counter = 0
    sampled_vs = []
    memory = {}
    param = QTensor(0.01 * np.random.randn(n_layers, n_qubits, 3))
    bias = QTensor([[0.0, 0.0, 0.0, 0.0]])
    param_targ = param.copy().reshape([1, -1]).pdata[0]
    bias_targ = bias.copy()
    loss_func = MeanSquaredError()
    model = DRLModel()
    opt = Adam(model.parameters(), lr=5)
    for i in range(MAX_ITERATIONS):
        start = time.time()
        state_t = env.reset()
        a_init = env.action_space.sample()
        total_reward = 0
        done = False
        frames = []
        for t in range(MAX_STEPS):
            frames.append(env.render(mode='rgb_array'))
            time.sleep(0.1)
            input_x = QTensor([[state_t]],dtype=kfloat32)
            acts = model(input_x) + bias
            # print(f'type of acts: {type(acts)}')
            act_t = tensor.QTensor.argmax(acts)
            # print(f'act_t: {act_t} type of act_t: {type(act_t)}')
            act_t_np = int(act_t.pdata[0])
            print(f'Episode: {i}, Steps: {t}, act: {act_t_np}')
            state_nt, reward, done, info = env.step(action=act_t_np)
            targ_counter += 1
            input_state_nt = QTensor([[state_nt]],dtype=kfloat32)
            act_nt = QTensor.argmax(model(input_state_nt)+bias)
            act_nt_np = int(act_nt.pdata[0])
            memory[i, t] = (state_t, act_t, reward, state_nt, done)
            if len(memory) >= BATCHSIZE:
                # print('Optimizing...')
                sampled_vs = [memory[k] for k in random.sample(list(memory), BATCHSIZE)]
                target_temp = []
                for s in sampled_vs:
                    if s[DONE]:
                        target_temp.append(QTensor(s[REWARD]).reshape([1, -1]))
                    else:
                        input_s = QTensor([[s[STATE_NT]]],dtype=kfloat32)
                        out_temp = s[REWARD] + GAMMA * tensor.max(model(input_s) + bias_targ)
                        out_temp = out_temp.reshape([1, -1])
                        target_temp.append(out_temp)
                target_out = []
                for b in sampled_vs:
                    input_b = QTensor([[b[STATE_T]]], requires_grad=True,dtype=kfloat32)
                    out_result = model(input_b) + bias
                    index = int(b[ACTION].pdata[0])
                    out_result_temp = out_result[0][index].reshape([1, -1])
                    target_out.append(out_result_temp)
                opt.zero_grad()
                target_label = tensor.concatenate(target_temp, 1)
                output = tensor.concatenate(target_out, 1)
                loss = loss_func(target_label, output)
                loss.backward()
                opt.step()
            # update parameters in target circuit
            if targ_counter == TARGET_MAX:
                param_targ = param.copy().reshape([1, -1]).pdata[0]
                bias_targ = bias.copy()
                targ_counter = 0
            state_t, act_t_np = state_nt, act_nt_np
            if done:
                print("reward", reward)
                if reward == 1.0:
                    frames.append(env.render(mode='rgb_array'))
                    display_frames_as_gif(frames, i)
                    exit()
                break
        end = time.time()

数据结果
---------------------------------

训练结果如下图所示，可以看出经过一定步骤之后达到最终位置。

.. image:: ./images/result_QDRL.gif
   :width: 600 px
   :align: center

|


7.基于小样本的量子卷积神经网络模型
=============================================

下面的例子使用2.0.8新加入的 `pyvqnet.qnn.vqc` 下的变分线路接口，实现了论文 `Generalization in quantum machine learning from few training data <https://www.nature.com/articles/s41467-022-32550-3>`_ 中的用于小样本的量子卷积神经网络模型。用于探讨量子机器学习模型中的泛化功能。

为了在量子电路中构建卷积层和池化层，我们将遵循论文中提出的 QCNN 结构。前一层将提取局部相关性，而后者允许降低特征向量的维度。在量子电路中，卷积层由沿着整个图像扫描的内核组成，是一个与相邻量子位相关的两个量子位酉。
至于池化层，我们将使用取决于相邻量子位测量的条件单量子位酉。最后，我们使用一个密集层，使用全对全单一门来纠缠最终状态的所有量子位，如下图所示：

.. image:: ./images/qcnn_structrue.png
   :width: 500 px
   :align: center

|

参考这种量子卷积层的设计方式，我们基于IsingXX、IsingYY、IsingZZ三个量子逻辑门对量子线路进行了构建，如下图所示：

.. image:: ./images/Qcnn_circuit.png
   :width: 600 px
   :align: center

|

其中输入数据为维度8*8的手写数字数据集，通过数据编码层，经过第一层卷积，由IsingXX、IsingYY、IsingZZ、U3构成，，随后经过一层池化层，在0、2、5位量子比特上再经过一层卷积和一层池化，最后再经过一层Random Unitary，其中由15个随机酉矩阵构成，对应经典的Dense Layer，测量结果为对手写数据为0和1的预测概率，具体代码实现如下：

以下代码运行需要额外安装 `pandas`, `sklearn`, `seaborn`。

.. code-block::

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import datasets
    import seaborn as sns

    from pyqpanda import *
    from pyvqnet.qnn.vqc.qcircuit import isingxx,isingyy,isingzz,u3,cnot,VQC_AmplitudeEmbedding,rxx,ryy,rzz,rzx
    from pyvqnet.qnn.vqc.qmachine import QMachine
    from pyvqnet.qnn.vqc.qmeasure import probs
    from pyvqnet.nn import Module, Parameter
    from pyvqnet.tensor import tensor,kfloat32
    from pyvqnet.tensor import QTensor
    from pyvqnet.dtype import *
    from pyvqnet.optim import Adam

    sns.set()

    seed = 0
    rng = np.random.default_rng(seed=seed)


    def convolutional_layer(qm, weights, wires, skip_first_layer=True):

        n_wires = len(wires)
        assert n_wires >= 3, "this circuit is too small!"
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    if indx % 2 == 0 and not skip_first_layer:

                        u3(q_machine=qm, wires=w, params=weights[:3])
                        u3(q_machine=qm, wires=wires[indx + 1], params=weights[3:6])

                    isingxx(q_machine=qm,  wires=[w, wires[indx + 1]], params=weights[6])
                    isingyy(q_machine=qm,  wires=[w, wires[indx + 1]], params=weights[7])
                    isingzz(q_machine=qm,  wires=[w, wires[indx + 1]], params=weights[8])
                    u3(q_machine=qm, wires=w, params=weights[9:12])
                    u3(q_machine=qm, wires=wires[indx + 1], params=weights[12:])

        return qm

    def pooling_layer(qm, weights, wires):
        """Adds a pooling layer to a circuit."""
        n_wires = len(wires)
        assert len(wires) >= 2, "this circuit is too small!"
        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                cnot(q_machine=qm, wires=[w, wires[indx - 1]])
                u3(q_machine=qm, params=weights, wires=wires[indx - 1])

    def conv_and_pooling(qm, kernel_weights, n_wires, skip_first_layer=True):
        """Apply both the convolutional and pooling layer."""

        convolutional_layer(qm, kernel_weights[:15], n_wires, skip_first_layer=skip_first_layer)
        pooling_layer(qm, kernel_weights[15:], n_wires)
        return qm

    def dense_layer(qm, weights, wires):
        """Apply an arbitrary unitary gate to a specified set of wires."""
        
        rzz(q_machine=qm,params=weights[0], wires=wires)
        rxx(q_machine=qm,params=weights[1], wires=wires)
        ryy(q_machine=qm,params=weights[2], wires=wires)
        rzx(q_machine=qm,params=weights[3], wires=wires)
        rxx(q_machine=qm,params=weights[5], wires=wires)
        rzx(q_machine=qm,params=weights[6], wires=wires)
        rzz(q_machine=qm,params=weights[7], wires=wires)
        ryy(q_machine=qm,params=weights[8], wires=wires)
        rzz(q_machine=qm,params=weights[9], wires=wires)
        rxx(q_machine=qm,params=weights[10], wires=wires)
        rzx(q_machine=qm,params=weights[11], wires=wires)
        rzx(q_machine=qm,params=weights[12], wires=wires)
        rzz(q_machine=qm,params=weights[13], wires=wires)
        ryy(q_machine=qm,params=weights[14], wires=wires)
        return qm


    num_wires = 6

    def conv_net(qm, weights, last_layer_weights, features):

        layers = weights.shape[1]
        wires = list(range(num_wires))

        VQC_AmplitudeEmbedding(input_feature = features, q_machine=qm)

        # adds convolutional and pooling layers
        for j in range(layers):
            conv_and_pooling(qm, weights[:, j], wires, skip_first_layer=(not j == 0))
            wires = wires[::2]

        assert last_layer_weights.size == 4 ** (len(wires)) - 1, (
            "The size of the last layer weights vector is incorrect!"
            f" \n Expected {4 ** (len(wires)) - 1}, Given {last_layer_weights.size}"
        )
        dense_layer(qm, last_layer_weights, wires)

        return probs(q_state=qm.states, num_wires=qm.num_wires, wires=[0])


    def load_digits_data(num_train, num_test, rng):
        """Return training and testing data of digits dataset."""
        digits = datasets.load_digits()
        features, labels = digits.data, digits.target

        # only use first two classes
        features = features[np.where((labels == 0) | (labels == 1))]
        labels = labels[np.where((labels == 0) | (labels == 1))]

        # normalize data
        features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))

        # subsample train and test split
        train_indices = rng.choice(len(labels), num_train, replace=False)
        test_indices = rng.choice(
            np.setdiff1d(range(len(labels)), train_indices), num_test, replace=False
        )

        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]

        return x_train, y_train,x_test, y_test


    class Qcnn_ising(Module):

        def __init__(self):
            super(Qcnn_ising, self).__init__()
            self.conv = conv_net
            self.qm = QMachine(num_wires,dtype=kcomplex128)
            self.weights = Parameter((18, 2), dtype=7)
            self.weights_last = Parameter((4 ** 2 -1,1), dtype=7)

        def forward(self, input):

            return self.conv(self.qm, self.weights, self.weights_last, input)


    from tqdm import tqdm


    def train_qcnn(n_train, n_test, n_epochs):

        # load data
        x_train, y_train, x_test, y_test = load_digits_data(n_train, n_test, rng)

        # init weights and optimizer
        model = Qcnn_ising()

        opti = Adam(model.parameters(), lr=0.01)

        # data containers
        train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []

        for step in range(n_epochs):
            model.train()
            opti.zero_grad()

            result = model(QTensor(x_train))

            train_cost = 1.0 - tensor.sums(result[tensor.arange(0, len(y_train)), y_train]) / len(y_train)
            # print(f"step {step}, train_cost {train_cost}")

            train_cost.backward()
            opti.step()

            train_cost_epochs.append(train_cost.to_numpy()[0])
            # compute accuracy on training data

            # print(tensor.sums(result[tensor.arange(0, len(y_train)), y_train] > 0.5))
            train_acc = tensor.sums(result[tensor.arange(0, len(y_train)), y_train] > 0.5) / result.shape[0]
            # print(train_acc)
            # print(f"step {step}, train_acc {train_acc}")
            train_acc_epochs.append(train_acc.to_numpy()[0])

            # compute accuracy and cost on testing data
            test_out = model(QTensor(x_test))
            test_acc = tensor.sums(test_out[tensor.arange(0, len(y_test)), y_test] > 0.5) / test_out.shape[0]
            test_acc_epochs.append(test_acc.to_numpy()[0])
            test_cost = 1.0 - tensor.sums(test_out[tensor.arange(0, len(y_test)), y_test]) / len(y_test)
            test_cost_epochs.append(test_cost.to_numpy()[0])

            # print(f"step {step}, test_cost {test_cost}")
            # print(f"step {step}, test_acc {test_acc}")

        return dict(
            n_train=[n_train] * n_epochs,
            step=np.arange(1, n_epochs + 1, dtype=int),
            train_cost=train_cost_epochs,
            train_acc=train_acc_epochs,
            test_cost=test_cost_epochs,
            test_acc=test_acc_epochs,
        )

    n_reps = 100
    n_test = 100
    n_epochs = 100

    def run_iterations(n_train):
        results_df = pd.DataFrame(
            columns=["train_acc", "train_cost", "test_acc", "test_cost", "step", "n_train"]
        )

        for _ in tqdm(range(n_reps)):
            results = train_qcnn(n_train=n_train, n_test=n_test, n_epochs=n_epochs)
            # np.save('test_qcnn.npy', results)
            results_df = pd.concat(
                [results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True
            )

        return results_df

    # run training for multiple sizes
    train_sizes = [2, 5, 10, 20, 40, 80]
    results_df = run_iterations(n_train=2)


    for n_train in train_sizes[1:]:
        results_df = pd.concat([results_df, run_iterations(n_train=n_train)])

    save = 0 # 保存数据
    draw = 0 # 绘图

    if save:
        results_df.to_csv('test_qcnn.csv', index=False)
    import pickle

    if draw:
        # aggregate dataframe
        results_df = pd.read_csv('test_qcnn.csv')
        df_agg = results_df.groupby(["n_train", "step"]).agg(["mean", "std"])
        df_agg = df_agg.reset_index()

        sns.set_style('whitegrid')
        colors = sns.color_palette()
        fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))

        generalization_errors = []

        # plot losses and accuracies
        for i, n_train in enumerate(train_sizes):
            df = df_agg[df_agg.n_train == n_train]

            dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
            lines = ["o-", "x--", "o-", "x--"]
            labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
            axs = [0, 0, 2, 2]

            for k in range(4):
                ax = axes[axs[k]]
                ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=10, color=colors[i], alpha=0.8)

            # plot final loss difference
            dif = df[df.step == 100].test_cost["mean"] - df[df.step == 100].train_cost["mean"]
            generalization_errors.append(dif)

        # format loss plot
        ax = axes[0]
        ax.set_title('Train and Test Losses', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        # format generalization error plot
        ax = axes[1]
        ax.plot(train_sizes, generalization_errors, "o-", label=r"$gen(\alpha)$")
        ax.set_xscale('log')
        ax.set_xticks(train_sizes)
        ax.set_xticklabels(train_sizes)
        ax.set_title(r'Generalization Error $gen(\alpha) = R(\alpha) - \hat{R}_N(\alpha)$', fontsize=14)
        ax.set_xlabel('Training Set Size')

        # format loss plot
        ax = axes[2]
        ax.set_title('Train and Test Accuracies', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.5, 1.05)

        legend_elements = [
                                mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes)
                            ] + [
                                mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
                                mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
                            ]

        axes[0].legend(handles=legend_elements, ncol=3)
        axes[2].legend(handles=legend_elements, ncol=3)

        axes[1].set_yscale('log', base=2)
        plt.show()



运行后的实验结果如下图所示：

.. image:: ./images/result_qcnn_small.png
   :width: 1000 px
   :align: center

|


8.用于手写数字识别的量子核函数模型
=============================================

下面的例子使用 `pyvqnet.qnn.vqc` 下的变分量子线路接口实现了论文 `Quantum Advantage Seeker with Kernels (QuASK): a software framework to speed up the research in quantum machine learning <https://link.springer.com/article/10.1007/s42484-023-00107-2>`_ 中的量子核函数，基于手写数字数据集来对量子核的性能进行评估。


本次实验基于crz、ZZFeatureMap逻辑门实现了量子核矩阵以及量子核映射中两种线路的设计。
算法输入数据为维度8*8的手写数字数据集, 通过PCA降维, 将输入的数据降维到相应的比特数的维度如2、4、8, 之后对数据进行标准化处理后, 获取训练数据集以及测试数据用于训练, 本次实现可分为两个, 分别为量子核矩阵以及核映射。
量子核矩阵由量子线路计算每一对数据的相似度，随后组成矩阵后输出；
量子核映射则分别计算两组数据映射后计算两组数据的相似度矩阵。

具体代码实现如下，需要额外安装 `sklearn`, `scipy` 等：

.. code-block::


    import numpy as np
    from sklearn.svm import SVC
    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from scipy.linalg import sqrtm
    import matplotlib.pyplot as plt
    from scipy.linalg import expm
    import numpy.linalg as la


    import sys
    sys.path.insert(0, "../")
    import pyvqnet
    from pyvqnet import _core
    from pyvqnet.dtype import *

    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn.vqc.qcircuit import PauliZ, VQC_ZZFeatureMap,PauliX,PauliY,hadamard,crz,rz
    from pyvqnet.qnn.vqc import QMachine
    from pyvqnet.qnn.vqc.qmeasure import expval
    from pyvqnet import tensor
    import functools as ft

    np.random.seed(42)
    # data load
    digits = datasets.load_digits(n_class=2)
    # create lists to save the results
    gaussian_accuracy = []
    quantum_accuracy = []
    projected_accuracy = []
    quantum_gaussian = []
    projected_gaussian = []

    # reduce dimensionality

    def custom_data_map_func(x):
        """
        custom data map function
        """
        coeff = x[0] if x.shape[0] == 1 else ft.reduce(lambda m, n: m * n, x)
        return coeff

    def vqnet_quantum_kernel(X_1, X_2=None):

        if X_2 is None:
            X_2 = X_1  # Training Gram matrix
        assert (
            X_1.shape[1] == X_2.shape[1]
        ), "The training and testing data must have the same dimensionality"
        N = X_1.shape[1]

        # create device using JAX

        # create projector (measures probability of having all "00...0")
        projector = np.zeros((2**N, 2**N))
        projector[0, 0] = 1
        projector = QTensor(projector,dtype=kcomplex128)
        # define the circuit for the quantum kernel ("overlap test" circuit)

        def kernel(x1, x2):
            qm = QMachine(N, dtype=kcomplex128)

            for i in range(N):
                hadamard(q_machine=qm, wires=i)
                rz(q_machine=qm,params=QTensor(2 * x1[i],dtype=kcomplex128), wires=i)
            for i in range(N):
                for j in range(i + 1, N):
                    crz(q_machine=qm,params=QTensor(2 * (np.pi - x1[i]) * (np.pi - x1[j]),dtype=kcomplex128), wires=[i, j])

            for i in range(N):
                for j in range(i + 1, N):
                    crz(q_machine=qm,params=QTensor(2 * (np.pi - x2[i]) * (np.pi - x2[j]),dtype=kcomplex128), wires=[i, j],use_dagger=True)        
            for i in range(N):
                rz(q_machine=qm,params=QTensor(2 * x2[i],dtype=kcomplex128), wires=i,use_dagger=True)
                hadamard(q_machine=qm, wires=i,use_dagger=True)

            states_1 = qm.states.reshape((1,-1))
            states_1 = tensor.conj(states_1)

            states_2 = qm.states.reshape((-1,1))

            result = tensor.matmul(tensor.conj(states_1), projector)
            result = tensor.matmul(result, states_2)
            return result.to_numpy()[0][0].real

        gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
        for i in range(len(X_1)):
            for j in range(len(X_2)):
                gram[i][j] = kernel(X_1[i], X_2[j])

        return gram


    def vqnet_projected_quantum_kernel(X_1, X_2=None, params=QTensor([1.0])):

        if X_2 is None:
            X_2 = X_1  # Training Gram matrix
        assert (
            X_1.shape[1] == X_2.shape[1]
        ), "The training and testing data must have the same dimensionality"


        def projected_xyz_embedding(X):
            """
            Create a Quantum Kernel given the template written in Pennylane framework

            Args:
                embedding: Pennylane template for the quantum feature map
                X: feature data (matrix)

            Returns:
                projected quantum feature map X
            """
            N = X.shape[1]

            def proj_feature_map(x):
                qm = QMachine(N, dtype=kcomplex128)
                VQC_ZZFeatureMap(x, qm, data_map_func=custom_data_map_func, entanglement="linear")

                return (
                    [expval(qm, i, PauliX(init_params=QTensor(1.0))).to_numpy() for i in range(N)]
                    + [expval(qm, i, PauliY(init_params=QTensor(1.0))).to_numpy() for i in range(N)]
                    + [expval(qm, i, PauliZ(init_params=QTensor(1.0))).to_numpy() for i in range(N)]
                )

            # build the gram matrix
            X_proj = [proj_feature_map(x) for x in X]

            return X_proj
        X_1_proj = projected_xyz_embedding(QTensor(X_1))
        X_2_proj = projected_xyz_embedding(QTensor(X_2))

        # print(X_1_proj)
        # print(X_2_proj)
        # build the gram matrix

        gamma = params[0]
        gram = tensor.zeros(shape=[X_1.shape[0], X_2.shape[0]],dtype=7)

        for i in range(len(X_1_proj)):
            for j in range(len(X_2_proj)):
                result = [a - b for a,b in zip(X_1_proj[i], X_2_proj[j])]
                result = [a**2 for a in result]
                value = tensor.exp(-gamma * sum(result))
                gram[i,j] = value
        return gram


    def calculate_generalization_accuracy(
        training_gram, training_labels, testing_gram, testing_labels
    ):

        svm = SVC(kernel="precomputed")
        svm.fit(training_gram, training_labels)

        y_predict = svm.predict(testing_gram)
        correct = np.sum(testing_labels == y_predict)
        accuracy = correct / len(testing_labels)
        return accuracy

    import time 
    qubits = [2, 4, 8]

    for n in qubits:
        n_qubits = n
        x_tr, x_te , y_tr , y_te = train_test_split(digits.data, digits.target, test_size=0.3, random_state=22)

        pca = PCA(n_components=n_qubits).fit(x_tr)
        x_tr_reduced = pca.transform(x_tr)
        x_te_reduced = pca.transform(x_te)

        # normalize and scale

        std = StandardScaler().fit(x_tr_reduced)
        x_tr_norm = std.transform(x_tr_reduced)
        x_te_norm = std.transform(x_te_reduced)

        samples = np.append(x_tr_norm, x_te_norm, axis=0)
        minmax = MinMaxScaler((-1,1)).fit(samples)
        x_tr_norm = minmax.transform(x_tr_norm)
        x_te_norm = minmax.transform(x_te_norm)

        # select only 100 training and 20 test data

        tr_size = 100
        x_tr = x_tr_norm[:tr_size]
        y_tr = y_tr[:tr_size]

        te_size = 100
        x_te = x_te_norm[:te_size]
        y_te = y_te[:te_size]
        
        quantum_kernel_tr = vqnet_quantum_kernel(X_1=x_tr)

        projected_kernel_tr = vqnet_projected_quantum_kernel(X_1=x_tr)

        quantum_kernel_te = vqnet_quantum_kernel(X_1=x_te, X_2=x_tr)

        projected_kernel_te = vqnet_projected_quantum_kernel(X_1=x_te, X_2=x_tr)
        
        quantum_accuracy.append(calculate_generalization_accuracy(quantum_kernel_tr, y_tr, quantum_kernel_te, y_te))
        print(f"qubits {n}, quantum_accuracy {quantum_accuracy[-1]}")
        projected_accuracy.append(calculate_generalization_accuracy(projected_kernel_tr.to_numpy(), y_tr, projected_kernel_te.to_numpy(), y_te))
        print(f"qubits {n}, projected_accuracy {projected_accuracy[-1]}")

    # train_size 100 test_size 20
    #
    # qubits 2, quantum_accuracy 1.0
    # qubits 2, projected_accuracy 1.0
    # qubits 4, quantum_accuracy 1.0
    # qubits 4, projected_accuracy 1.0
    # qubits 8, quantum_accuracy 0.45
    # qubits 8, projected_accuracy 1.0

    # train_size 100 test_size 100
    #
    # qubits 2, quantum_accuracy 1.0
    # qubits 2, projected_accuracy 0.99
    # qubits 4, quantum_accuracy 0.99
    # qubits 4, projected_accuracy 0.98
    # qubits 8, quantum_accuracy 0.51
    # qubits 8, projected_accuracy 0.99


无监督学习
****************************

1.Quantum Kmeans
==========================

1.1介绍
-------------------

聚类算法是一种典型的无监督学习算法，主要用于将相似的样本自动归为一类。聚类算法中，根据样本之间的相似性，将样本划分为不同的类别。对于不同的相似度计算方法，会得到不同的聚类结果。常用的相似度计算方法是欧氏距离法。我们要展示的是量子 K-Means 算法。 K-Means算法是一种基于距离的聚类算法，它以距离作为相似度的评价指标，即两个对象之间的距离越近，相似度越大。该算法认为簇是由相距很近的对象组成的，因此紧凑且独立的簇是最终目标。

在VQNet中同样可以进行Quantum Kmeans量子机器学习模型的开发。下面给出了Quantum Kmeans聚类任务的一个示例。通过量子线路，我们可以构建一个测量值与经典机器学习的变量的欧几里得距离正相关，达到进行查找最近邻的目标。

1.2算法原理介绍
-------------------------

量子K-Means算法的实现主要使用swap test来比较输入数据点之间的距离。从N个数据点中随机选择K个点作为质心，测量每个点到每个质心的距离，并将其分配到最近的质心类，重新计算已经得到的每个类的质心，迭代2到3步，直到新的质心等于或小于指定的阈值，算法结束。 在我们的示例中，我们选择了100个数据点、2个质心，并使用CSWAP电路来计算距离。 最后，我们获得了两个数据点集群。 :math:`|0\rangle` 是辅助比特, 通过H逻辑门量子比特位将变为 :math:`\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)`. 在量比特 :math:`|1\rangle` 的控制下, 量子线路将会翻转 :math:`|x\rangle` 和 :math:`|y\rangle​` . 最终得到结果:

.. math::

    |0_{anc}\rangle |x\rangle |y\rangle \rightarrow \frac{1}{2}|0_{anc}\rangle(|xy\rangle + |yx\rangle) + \frac{1}{2}|1_{anc}\rangle(|xy\rangle - |yx\rangle)

如果我们单独测量辅助量子比特，那么基态最终状态的概率 :math:`|1\rangle` 是:

.. math::

    P(|1_{anc}\rangle) = \frac{1}{2} - \frac{1}{2}|\langle x | y \rangle|^2

两个量子态之间的欧几里得距离如下：

.. math::

    Euclidean \ distance = \sqrt{(2 - 2|\langle x | y \rangle|)}

可见测量量子比特位 :math:`|1\rangle` ​与欧几里得距离有正相关. 本算法的量子线路如下所述：

.. image:: ./images/Kmeans.jpg
   :width: 600 px
   :align: center

|

1.3 VQNet实现
-----------------

1.3.1 环境准备
^^^^^^^^^^^^^^^^^^^^^^^

环境采用python3.8，建议使用conda进行环境配置，自带numpy，scipy，matplotlib，sklearn等工具包，方便使用，如果采用的是python环境，需要安装相关的包
还需要准备如下环境pyvqnet

1.3.2 数据准备
^^^^^^^^^^^^^^^^^^^^^^^

数据采用scipy下的make_blobs来随机产生，并定义函数用于生成高斯分布数据。

.. code-block::

    import sys
    sys.path.insert(0, "../")
    import math
    import numpy as np
    from pyvqnet.tensor.tensor import QTensor, zeros
    import pyvqnet.tensor.tensor as tensor
    import pyqpanda as pq
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  #pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass
    # 根据数据的数据量n，聚类中心k和数据标准差std返回对应数据点和聚类中心点
    def get_data(n, k, std):
        data = make_blobs(n_samples=n,
                        n_features=2,
                        centers=k,
                        cluster_std=std,
                        random_state=100)
        points = data[0]
        centers = data[1]
        return points, centers


1.3.3 量子线路
^^^^^^^^^^^^^^^^^^^^^^^

使用VQNet构建量子线路

.. code-block::

    # 根据输入的坐标点d(x,y)来计算输入的量子门旋转角度
    def get_theta(d):
        x = d[0]
        y = d[1]
        theta = 2 * math.acos((x.item() + y.item()) / 2.0)
        return theta

    # 根据输入的量子数据点构建量子线路
    def qkmeans_circuits(x, y):
        """
        Quantum Circuit for kmeans
        """
        theta_1 = get_theta(x)
        theta_2 = get_theta(y)

        num_qubits = 3
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)
        cbits = machine.cAlloc_many(num_qubits)
        circuit = pq.QCircuit()

        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.H(qubits[1]))
        circuit.insert(pq.H(qubits[2]))

        circuit.insert(pq.U3(qubits[1], theta_1, np.pi, np.pi))
        circuit.insert(pq.U3(qubits[2], theta_2, np.pi, np.pi))

        circuit.insert(pq.SWAP(qubits[1], qubits[2]).control([qubits[0]]))

        circuit.insert(pq.H(qubits[0]))

        prog = pq.QProg()
        prog.insert(circuit)
        prog << pq.Measure(qubits[0], cbits[0])  #pylint:disable=expression-not-assigned
        prog.insert(pq.Reset(qubits[0]))
        prog.insert(pq.Reset(qubits[1]))
        prog.insert(pq.Reset(qubits[2]))

        result = machine.run_with_configuration(prog, cbits, 1024)

        data = result

        if len(data) == 1:
            return 0.0
        else:
            return data["001"] / 1024.0

1.3.4 数据可视化
^^^^^^^^^^^^^^^^^^^^^^^

对相关聚类数据进行可视化计算

.. code-block::

    # 对散点和聚类中心进行可视化
    def draw_plot(points, centers, label=True):
        points = np.array(points)
        centers = np.array(centers)
        if label is False:
            plt.scatter(points[:, 0], points[:, 1])
        else:
            plt.scatter(points[:, 0], points[:, 1], c=centers, cmap="viridis")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

1.3.5 聚类计算
^^^^^^^^^^^^^^^^^^^^^^^

对相关聚类数据进行聚类中心计算

.. code-block::

    # 随机生成聚类中心点
    def initialize_centers(points,k):
        return points[np.random.randint(points.shape[0],size=k),:]


    def find_nearest_neighbour(points, centroids):
        """
        Find nearest neighbour
        """
        n = points.shape[0]
        k = centroids.shape[0]

        centers = zeros([n], dtype=points.dtype)

        for i in range(n):
            min_dis = 10000
            ind = 0
            for j in range(k):

                temp_dis = qkmeans_circuits(points[i, :], centroids[j, :])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    ind = j
            centers[i] = ind

        return centers


    def find_centroids(points, centers):
        """
        find centroids
        """
        k = int(tensor.max(centers).item()) + 1
        centroids = tensor.zeros([k, 2], dtype=points.dtype)

        for i in range(k):

            cur_i = centers == i
            x = points[:, 0]
            x = x[cur_i]
            y = points[:, 1]
            y = y[cur_i]
            centroids[i, 0] = tensor.mean(x)
            centroids[i, 1] = tensor.mean(y)

        return centroids

    def preprocess(points):
        n = len(points)
        x = 30.0 * np.sqrt(2)
        for i in range(n):
            points[i, :] += 15
            points[i, :] /= x

        return points


    def qkmeans_run():
        """
        Main function for run qkmeans algorithm
        """
        n = 100  # number of data points
        k = 3  # Number of centers
        std = 2  # std of datapoints

        points, o_centers = get_data(n, k, std)  # dataset

        points = preprocess(points)  # Normalize dataset

        centroids = initialize_centers(points, k)  # Intialize centroids

        epoch = 5
        points = QTensor(points)
        centroids = QTensor(centroids)
        plt.figure()
        plt.title("origin")
        draw_plot(points.data, o_centers, label=False)

        # run k-means algorithm
        for i in range(epoch):
            print(f"iteration {i}")
            centers = find_nearest_neighbour(points,
                                            centroids)  # find nearest centers
            centroids = find_centroids(points, centers)  # find centroids

        plt.figure()
        plt.title(f"result after iteration {epoch}")
        draw_plot(points.data, centers.data)

    # 运行程序入口
    if __name__ == "__main__":
        qkmeans_run()


1.3.6 聚类前数据分布
^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ./images/ep_1.png
   :width: 600 px
   :align: center

1.3.7聚类后数据分布
^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ./images/ep_9.png
   :width: 600 px
   :align: center

|


量子机器学习研究
****************************

量子模型拟合fourier series算法
===================================

通过参数化的量子线路将数据输入映射到预测的模型，量子计算机可用于监督学习。虽然已经做了大量工作来研究这种方法的实际意义，
但这些模型的许多重要理论特性仍然未知。在这里，我们研究了将数据编码到模型中的策略如何影响参数化量子电路作为函数逼近器的表达能力。

本例参照 `The effect of data encoding on the expressive power of variational quantum machine learning models <https://arxiv.org/pdf/2008.08605.pdf>`_ 论文将量子计算机设计的常见量子机器学习模型与傅里叶级数联系起来。

1.1用串行Pauli旋转编码拟合傅里叶级数
----------------------------------------

首先我们展示使用Pauli旋转作为数据编码门的量子模型如何只能在一定程度上拟合傅里叶级数。为简单起见，我们将只看单量子比特电路：

.. image:: ./images/single_qubit_model.png
   :width: 600 px
   :align: center

|

制作输入数据，定义并行量子模型，不进行模型训练结果。

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)


    degree = 1  # degree of the target function
    scaling = 1  # scaling of the data
    coeffs = [0.15 + 0.15j]*degree  # coefficients of non-zero frequencies
    coeff0 = 0.1  # coefficient of zero frequency

    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx+1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(x_) for x_ in x])

    plt.plot(x, target_y, c='black')
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.ylim(-1, 1)
    plt.show()


    def S(scaling, x, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RX(qubits[0], scaling * x))
        return cir

    def W(theta, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RZ(qubits[0], theta[0]))
        cir.insert(pq.RY(qubits[0], theta[1]))
        cir.insert(pq.RZ(qubits[0], theta[2]))
        return cir

    def serial_quantum_model(weights, x, num_qubits, scaling):
        cir = pq.QCircuit()
        machine = pq.CPUQVM()  
        machine.init_qvm()  
        qubits = machine.qAlloc_many(num_qubits)

        for theta in weights[:-1]:
            cir.insert(W(theta, qubits))
            cir.insert(S(scaling, x, qubits))

        # (L+1)'th unitary
        cir.insert(W(weights[-1], qubits))
        prog = pq.QProg()
        prog.insert(cir)

        exp_vals = []
        for position in range(num_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)

        return exp_vals

    r = 1
    weights = 2 * np.pi * np.random.random(size=(r+1, 3))  # some random initial weights

    x = np.linspace(-6, 6, 70)
    random_quantum_model_y = [serial_quantum_model(weights, x_, 1, 1) for x_ in x]

    plt.plot(x, target_y, c='black', label="true")
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.plot(x, random_quantum_model_y, c='blue', label="predict")
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    plt.show()


不训练的量子线路运行结果为：

.. image:: ./images/single_qubit_model_result_no_train.png
   :width: 600 px
   :align: center

|


制作输入数据，定义串行量子模型，并结合VQNet框架的QuantumLayer层构建训练模型。

.. code-block::

    """
    Quantum Fourier Series Serial
    """
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    degree = 1  # degree of the target function
    scaling = 1  # scaling of the data
    coeffs = [0.15 + 0.15j]*degree  # coefficients of non-zero frequencies
    coeff0 = 0.1  # coefficient of zero frequency

    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx+1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(xx) for xx in x])

    plt.plot(x, target_y, c='black')
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.ylim(-1, 1)
    plt.show()


    def S(x, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RX(qubits[0], x))
        return cir

    def W(theta, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RZ(qubits[0], theta[0]))
        cir.insert(pq.RY(qubits[0], theta[1]))
        cir.insert(pq.RZ(qubits[0], theta[2]))
        return cir


    r = 1
    weights = 2 * np.pi * np.random.random(size=(r+1, 3))  # some random initial weights

    x = np.linspace(-6, 6, 70)


    def q_circuits_loop(x, weights, qubits, clist, machine):

        result = []
        for xx in x:
            cir = pq.QCircuit()
            weights = weights.reshape([2, 3])

            for theta in weights[:-1]:
                cir.insert(W(theta, qubits))
                cir.insert(S(xx, qubits))

            cir.insert(W(weights[-1], qubits))
            prog = pq.QProg()
            prog.insert(cir)

            exp_vals = []
            for position in range(1):
                pauli_str = {"Z" + str(position): 1.0}
                exp2 = expval(machine, prog, pauli_str, qubits)
                exp_vals.append(exp2)
                result.append(exp2)
        return result

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.q_fourier_series = QuantumLayer(q_circuits_loop, 6, "CPU", 1)

        def forward(self, x):
            return self.q_fourier_series(x)

    def run():
        model = Model()

        optimizer = Adam(model.parameters(), lr=0.5)
        batch_size = 2
        epoch = 5
        loss = MeanSquaredError()
        print("start training..............")
        model.train()
        max_steps = 50
        for i in range(epoch):
            sum_loss = 0
            count = 0
            for step in range(max_steps):
                optimizer.zero_grad()
                # Select batch of data
                batch_index = np.random.randint(0, len(x), (batch_size,))
                x_batch = x[batch_index]
                y_batch = target_y[batch_index]
                data, label = QTensor([x_batch]), QTensor([y_batch])
                result = model(data)
                loss_b = loss(label, result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count += batch_size
            print(f"epoch:{i}, #### loss:{sum_loss/count} ")

            model.eval()
            predictions = []
            for xx in x:
                data = QTensor([[xx]])
                result = model(data)
                predictions.append(result.pdata[0])

            plt.plot(x, target_y, c='black', label="true")
            plt.scatter(x, target_y, facecolor='white', edgecolor='black')
            plt.plot(x, predictions, c='blue', label="predict")
            plt.ylim(-1, 1)
            plt.legend(loc="upper right")
            plt.show()

    if __name__ == "__main__":
        run()

其中量子模型为：

.. image:: ./images/single_qubit_model_circuit.png
   :width: 600 px
   :align: center

|

网络训练结果为：

.. image:: ./images/single_qubit_model_result.png
   :width: 600 px
   :align: center

|

网络训练损失为：

.. code-block::

    start training..............
    epoch:0, #### loss:0.04852807720773853
    epoch:1, #### loss:0.012945819365559146
    epoch:2, #### loss:0.0009359727291666786
    epoch:3, #### loss:0.00015995280153333625
    epoch:4, #### loss:3.988249877352246e-05


1.2用并行Pauli旋转编码拟合傅里叶级数
------------------------------------

根据论文所示，我们期望与串行模型相似的结果：只有在量子模型中编码门至少有r个重复时，才能拟合r阶的傅立叶级数。量子比特电路：

.. image:: ./images/parallel_model.png
   :width: 600 px
   :align: center

|

制作输入数据，定义并行量子模型，不进行模型训练结果。

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    degree = 1  # degree of the target function
    scaling = 1  # scaling of the data
    coeffs = [0.15 + 0.15j] * degree  # coefficients of non-zero frequencies
    coeff0 = 0.1  # coefficient of zero frequency

    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx + 1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(xx) for xx in x])


    def S1(x, qubits):
        cir = pq.QCircuit()
        for q in qubits:
            cir.insert(pq.RX(q, x))
        return cir

    def W1(theta, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[0][i][0]))
            cir.insert(pq.RY(qubits[i], theta[0][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[0][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[1][i][0]))
            cir.insert(pq.RY(qubits[i], theta[1][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[1][i][2]))

        cir.insert(pq.CNOT(qubits[0], qubits[len(qubits) - 1]))
        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i + 1], qubits[i]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[2][i][0]))
            cir.insert(pq.RY(qubits[i], theta[2][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[2][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        return cir

    def parallel_quantum_model(weights, x, num_qubits):
        cir = pq.QCircuit()
        machine = pq.CPUQVM()  
        machine.init_qvm()  
        qubits = machine.qAlloc_many(num_qubits)

        cir.insert(W1(weights[0], qubits))
        cir.insert(S1(x, qubits))

        cir.insert(W1(weights[1], qubits))
        prog = pq.QProg()
        prog.insert(cir)

        exp_vals = []
        for position in range(1):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)

        return exp_vals

    r = 3

    trainable_block_layers = 3
    weights = 2 * np.pi * np.random.random(size=(2, trainable_block_layers, r, 3))
    # print(weights)
    x = np.linspace(-6, 6, 70)
    random_quantum_model_y = [parallel_quantum_model(weights, xx, r) for xx in x]

    plt.plot(x, target_y, c='black', label="true")
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.plot(x, random_quantum_model_y, c='blue', label="predict")
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    plt.show()

不训练的量子线路运行结果为：

.. image:: ./images/parallel_model_result_no_train.png
   :width: 600 px
   :align: center

|


制作输入数据，定义并行量子模型，并结合VQNet框架的QuantumLayer层构建训练模型。

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np

    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer, QuantumLayerMultiProcess
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    degree = 1  # degree of the target function
    scaling = 1  # scaling of the data
    coeffs = [0.15 + 0.15j] * degree  # coefficients of non-zero frequencies
    coeff0 = 0.1  # coefficient of zero frequency

    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx + 1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(xx) for xx in x])

    plt.plot(x, target_y, c='black')
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.ylim(-1, 1)
    plt.show()

    def S1(x, qubits):
        cir = pq.QCircuit()
        for q in qubits:
            cir.insert(pq.RX(q, x))
        return cir

    def W1(theta, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[0][i][0]))
            cir.insert(pq.RY(qubits[i], theta[0][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[0][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[1][i][0]))
            cir.insert(pq.RY(qubits[i], theta[1][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[1][i][2]))

        cir.insert(pq.CNOT(qubits[0], qubits[len(qubits) - 1]))
        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i + 1], qubits[i]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[2][i][0]))
            cir.insert(pq.RY(qubits[i], theta[2][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[2][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        return cir

    def q_circuits_loop(x, weights, qubits, clist, machine):

        result = []
        for xx in x:
            cir = pq.QCircuit()
            weights = weights.reshape([2, 3, 3, 3])

            cir.insert(W1(weights[0], qubits))
            cir.insert(S1(xx, qubits))

            cir.insert(W1(weights[1], qubits))
            prog = pq.QProg()
            prog.insert(cir)

            exp_vals = []
            for position in range(1):
                pauli_str = {"Z" + str(position): 1.0}
                exp2 = expval(machine, prog, pauli_str, qubits)
                exp_vals.append(exp2)
                result.append(exp2)
        return result


    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()

            self.q_fourier_series = QuantumLayer(q_circuits_loop, 2 * 3 * 3 * 3, "CPU", 3)

        def forward(self, x):
            return self.q_fourier_series(x)

    def run():
        model = Model()

        optimizer = Adam(model.parameters(), lr=0.01)
        batch_size = 2
        epoch = 5
        loss = MeanSquaredError()
        print("start training..............")
        model.train()
        max_steps = 50
        for i in range(epoch):
            sum_loss = 0
            count = 0
            for step in range(max_steps):
                optimizer.zero_grad()
                # Select batch of data
                batch_index = np.random.randint(0, len(x), (batch_size,))
                x_batch = x[batch_index]
                y_batch = target_y[batch_index]
                data, label = QTensor([x_batch]), QTensor([y_batch])
                result = model(data)
                loss_b = loss(label, result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count += batch_size

            loss_cout = sum_loss / count
            print(f"epoch:{i}, #### loss:{loss_cout} ")

            if loss_cout < 0.002:
                model.eval()
                predictions = []
                for xx in x:
                    data = QTensor([[xx]])
                    result = model(data)
                    predictions.append(result.pdata[0])

                plt.plot(x, target_y, c='black', label="true")
                plt.scatter(x, target_y, facecolor='white', edgecolor='black')
                plt.plot(x, predictions, c='blue', label="predict")
                plt.ylim(-1, 1)
                plt.legend(loc="upper right")
                plt.show()


    if __name__ == "__main__":
        run()

其中量子模型为：

.. image:: ./images/parallel_model_circuit.png
   :width: 600 px
   :align: center

|

网络训练结果为：

.. image:: ./images/parallel_model_result.png
   :width: 600 px
   :align: center

|

网络训练损失为：

.. code-block::

    start training..............
    epoch:0, #### loss:0.0037272341538482578
    epoch:1, #### loss:5.271130586635309e-05
    epoch:2, #### loss:4.714951917250687e-07
    epoch:3, #### loss:1.0968826371082763e-08
    epoch:4, #### loss:2.1258629738507562e-10



量子线路表达能力
===================================


在论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/abs/1905.10876>`_ 中，
作者提出了基于神经网络输出态之间的保真度概率分布的表达能力量化方法。
对任意量子神经网络 :math:`U(\vec{\theta})` ，采样两次神经网络参数（设为 :math:`\vec{\phi}` 和 :math:`\vec{\psi}` ），
则两个量子电路输出态之间的保真度 :math:`F=|\langle0|U(\vec{\phi})^\dagger U(\vec{\psi})|0\rangle|^2` 服从某个概率分布：

.. math::

    F\sim{P}(f)

文献指出，量子神经网络 :math:`U` 能够均匀地分布在所有酉矩阵上时（此时称 :math:`U` 服从哈尔分布），保真度的概率分布 :math:`P_\text{Haar}(f)` 满足

.. math::

    P_\text{Haar}(f)=(2^{n}-1)(1-f)^{2^n-2}

统计数学中的 K-L 散度（也称相对熵）可以衡量两个概率分布之间的差异。两个离散概率分布 :math:`P,Q` 之间的 K-L 散度定义为

.. math::

    D_{KL}(P||Q)=\sum_jP(j)\ln\frac{P(j)}{Q(j)}

如果将量子神经网络输出的保真度分布记为 :math:`P_\text{QNN}(f)` ，则量子神经网络的表达能力定义为 :math:`P_\text{QNN}(f)` 和 :math:`P_\text{Haar}(f)` 之间的 K-L 散度 ：

.. math::

    \text{Expr}_\text{QNN}=D_{KL}(P_\text{QNN}(f)||P_\text{Haar}(f))


因此，当 :math:`P_\text{QNN}(f)` 越接近 :math:`P_\text{Haar}(f)` 时， :math:`\text{Expr}` 将越小（越趋近于 0），
量子神经网络的表达能力也就越强；反之， :math:`\text{Expr}` 越大，量子神经网络的表达能力也就越弱。

我们可以根据该定义直接计算单比特量子神经网络 :math:`R_Y(\theta)` ， :math:`R_Y(\theta_1)R_Z(\theta_2)` 和 :math:`R_Y(\theta_1)R_Z(\theta_2)R_Y(\theta_3)` 的表达能力：

以下用VQNet展示了 `HardwareEfficientAnsatz <https://arxiv.org/abs/1704.05018>`_ 在不同深度下（1，2，3）的量子线路表达能力。


.. code-block::

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from scipy import integrate
    from scipy.linalg import sqrtm
    from scipy.stats import entropy

    import pyqpanda as pq
    import numpy as np
    from pyvqnet.qnn.ansatz import HardwareEfficientAnsatz
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.quantum_expressibility.quantum_express import fidelity_of_cir, fidelity_harr_sample
    num_qubit = 1  # the number of qubit
    num_sample = 2000  # the number of sample
    outputs_y = list()  # save QNN outputs


    # plot histgram
    def plot_hist(data, num_bin, title_str):
        def to_percent(y, position):
            return str(np.around(y * 100, decimals=2)) + '%'

        plt.hist(data,
                weights=[1. / len(data)] * len(data),
                bins=np.linspace(0, 1, num=num_bin),
                facecolor="blue",
                edgecolor="black",
                alpha=0.7)
        plt.xlabel("Fidelity")
        plt.ylabel("frequency")
        plt.title(title_str)
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.show()



    def cir(num_qubits, depth):

        machine = pq.CPUQVM()
        machine.init_qvm()
        qlist = machine.qAlloc_many(num_qubits)
        az = HardwareEfficientAnsatz(num_qubits, ["rx", "RY", "rz"],
                                    qlist,
                                    entangle_gate="cnot",
                                    entangle_rules="linear",
                                    depth=depth)
        w = tensor.QTensor(
            np.random.uniform(size=[az.get_para_num()], low=0, high=2 * np.pi))

        cir1 = az.create_ansatz(w)
        return cir1, machine, qlist

哈尔采样输出的保真度服从分布：

.. image:: ./images/haar-fidelity.png
   :width: 600 px
   :align: center

|

.. code-block::


    # 设置电路宽度和最大深度
    num_qubit = 4
    max_depth = 3
    # 计算哈尔采样对应的保真度分布

    flist, p_haar, theory_haar = fidelity_harr_sample(num_qubit, num_sample)
    title_str = "haar, %d qubit(s)" % num_qubit
    plot_hist(flist, 50, title_str)



    Expr_cel = list()
    # 计算不同深度的神经网络的表达能力
    for DEPTH in range(1, max_depth + 1):

        f_list, p_cel = fidelity_of_cir(HardwareEfficientAnsatz, num_qubit, DEPTH,
                                        num_sample)
        title_str = f"HardwareEfficientAnsatz, {num_qubit} qubit(s) {DEPTH} layer(s)"
        plot_hist(f_list, 50, title_str)
        expr = entropy(p_cel, theory_haar)
        Expr_cel.append(expr)
    # 比较不同深度的神经网络的表达能力
    print(
        f"深度为 1,2,3 的神经网络的表达能力分别为 { np.around(Expr_cel, decimals=4)} 越小越好。", )
    plt.plot(range(1, max_depth + 1), Expr_cel, marker='>')
    plt.xlabel("depth")
    plt.yscale('log')
    plt.ylabel("Expr.")
    plt.xticks(range(1, max_depth + 1))
    plt.title("Expressibility vs Circuit Depth")
    plt.show()


采样深度为 1 的电路的保真度分布

.. image:: ./images/f1.png
   :width: 600 px
   :align: center

|

采样深度为 2 的电路的保真度分布

.. image:: ./images/f2.png
   :width: 600 px
   :align: center

|

采样深度为 3 的电路的保真度分布

.. image:: ./images/f3.png
   :width: 600 px
   :align: center

|

可见，随着量子线路深度提升，保真度KL散度越小，表达力越强。

.. image:: ./images/express.png
   :width: 600 px
   :align: center

|

贫瘠高原现象
===================================


在经典神经网络的训练中，基于梯度的优化方法不仅会遇到局部极小值的问题，
而且还会遇到梯度接近于零的鞍点等几何结构。
相应的，量子神经网络中也存在 "贫瘠高原效应" 。
这一奇特现象最早由McClean等人在2018年发现 `Barren plateaus in quantum neural network training landscapes <https://arxiv.org/abs/1803.11173>`_。
简单地说，当你选择的随机电路结构满足一定程度的复杂性时，优化曲面会变得非常平坦，
这使得基于梯度下降的优化方法很难找到全局最小值
。对于大多数变分量子算法(VQE等)来说，这种现象意味着当量子位的数量增加时，
具有随机结构的电路可能无法很好地工作。
这将使精心设计的损失函数对应的优化曲面变成一个巨大的平台，
使量子神经网络的训练更加困难。
模型随机找到的初值很难逃离这个平台，梯度下降的收敛速度会非常慢。


本案例主要使用VQNet展示贫瘠高原现象，使用梯度分析函数对用户自定义量子神经网络中的参数梯度进行分析。

以下代码按照原论文中提及的类似方法搭建如下随机电路：

首先作用在所有量子比特上绕布洛赫球的 Y-轴旋转 :math:`\pi/4` 。

其余的结构加起来构成一个模块（Block）, 每个模块共分为两层：

- 第一层搭建随机的旋转门, 其中 :math:`R \in \{R_x, R_y, R_z\}` 。
- 第二层由 CZ 门组成，作用在每两个相邻的量子比特上。

线路代码如rand_circuit_pq函数所示。

当我们确定了电路的结构之后，我们还需要定义一个损失函数（loss function）来确定优化曲面。
按照原论文中提及的，我们采用 VQE算法中常用的损失函数：

.. math::

    \mathcal{L}(\boldsymbol{\theta})= \langle0| U^{\dagger}(\boldsymbol{\theta})H U(\boldsymbol{\theta}) |0\rangle

其中的酉矩阵  :math:`U(\boldsymbol{\theta})` 就是我们上一部分搭建的带有随机结构的量子神经网络。
哈密顿量 :math:`H = |00\cdots 0\rangle\langle00\cdots 0|` 。
本案例使用在不同量子比特数上构建如上VQE算法，并生成200组随机网络结构和不同的随机初始参数.
含参线路中参数的梯度按照paramter-shift算法进行计算。
然后统计得到的其中变分参数的200个梯度的平均值和方差。

以下例子对变分量子参数的最后一个进行分析，读者也可自行修改为其他合理的值。
通过运行，读者不难发现，随着量子比特数增加，量子参数的梯度的方差越来越小，均值越来越接近0。

.. code-block::

        """
        贫瘠高原
        """
        import pyqpanda as pq
        import numpy as np
        import matplotlib.pyplot as plt

        from pyvqnet.qnn import Hermitian_expval, grad

        param_idx = -1
        gate_set = [pq.RX, pq.RY, pq.RZ]


        def rand_circuit_pq(params, num_qubits):
            cir = pq.QCircuit()
            machine = pq.CPUQVM()
            machine.init_qvm()
            qlist = machine.qAlloc_many(num_qubits)

            for i in range(num_qubits):
                cir << pq.RY(
                    qlist[i],
                    np.pi / 4,
                )

            random_gate_sequence = {
                i: np.random.choice(gate_set)
                for i in range(num_qubits)
            }
            for i in range(num_qubits):
                cir << random_gate_sequence[i](qlist[i], params[i])

            for i in range(num_qubits - 1):
                cir << pq.CZ(qlist[i], qlist[i + 1])

            prog = pq.QProg()
            prog.insert(cir)
            machine.directly_run(prog)
            result = machine.get_qstate()

            H = np.zeros((2**num_qubits, 2**num_qubits))
            H[0, 0] = 1
            expval = Hermitian_expval(H, result, [i for i in range(num_qubits)],
                                    num_qubits)

            return expval


        qubits = [2, 3, 4, 5, 6]
        variances = []
        num_samples = 200
        means = []
        for num_qubits in qubits:
            grad_vals = []
            for i in range(num_samples):
                params = np.random.uniform(0, np.pi, size=num_qubits)
                g = grad(rand_circuit_pq, params, num_qubits)

                grad_vals.append(g[-1])
            variances.append(np.var(grad_vals))
            means.append(np.mean(grad_vals))
        variances = np.array(variances)
        means = np.array(means)
        qubits = np.array(qubits)


        plt.figure()

        plt.plot(qubits, variances, "v-")

        plt.xlabel(r"N Qubits")
        plt.ylabel(r"variance")
        plt.show()


        plt.figure()

        plt.plot(qubits, means, "v-")

        plt.xlabel(r"N Qubits")
        plt.ylabel(r"means")

        plt.show()


下图显示了参数梯度的均值随着量子比特数变化的情况，随着量子比特数增加，参数梯度趋近0。

.. image:: ./images/Barren_Plateau_mean.png
   :width: 600 px
   :align: center

|

下图显示了参数梯度的方差随着量子比特数变化的情况，随着量子比特数增加，参数梯度几乎不变化。
可以预见，任意含参逻辑门搭建的量子线路随着量子比特提升，在任意参数初始化情况下，参数训练容易陷入难以更新的情况。

.. image:: ./images/Barren_Plateau_variance.png
   :width: 600 px
   :align: center

|


量子感知机
===================================

人工神经网络是机器学习算法和人工智能的一种经典方法。从历史上看，人工神经元的最简单实现可以追溯到经典Rosenblatt 的“感知器”，但其长期实际应用可能会受到计算复杂度快速扩展的阻碍，尤其是与多层感知器的训练相关网络。这里我们参照论文 `An Artificial Neuron Implemented on an Actual Quantum Processor <https://arxiv.org/abs/1811.02266>`__ 一种基于量子信息的算法实现量子计算机版本的感知器，在编码资源方面显示出相比经典模型指数优势。

对于该量子感知机，处理的数据是 0 1 二进制比特字符串。其目标是想识别形如下图 :math:`w` 十字形状的模式。

.. image:: ./images/QP-data.png
   :width: 600 px
   :align: center

|

使用二进制比特字符串对其进行编码，其中黑为0，白为1，可知 :math:`w` 编码为（1，1，1，1，1，1，0，1，1，0，0，0，1，1，0，1）。共16位的字符串正好可以编码进4bit的量子态的振幅的符号上，符号为负数编码为0，符号为正数编码为1。通过以上编码方式，我们算法输入input转化为16位的二进制串。这样的不重复的二进制串可以分别对应特定的输入线路 :math:`U_i` 。
 
该论文提出的量子感知机线路结构如下：

.. image:: ./images/QP-cir.png
   :width: 600 px
   :align: center

|

在比特0~3上构建编码线路 :math:`U_i` ，包含多受控的 :math:`CZ` 门， :math:`CNOT` 门， :math:`H` 门；在 :math:`U_i` 后面紧接着构建权重变换线路 :math:`U_w` ，同样由受控门以及 :math:`H` 门构成。使用 :math:`U_i` 可以进行酉矩阵变化，将数据编码到量子态上：

.. math::
    U_i|0\rangle^{\otimes N}=\left|\psi_i\right\rangle

使用酉矩阵变换 :math:`U_w` 来计算输入和权重之间的内积：

.. math::
    U_w\left|\psi_i\right\rangle=\sum_{j=0}^{m-1} c_j|j\rangle \equiv\left|\phi_{i, w}\right\rangle

使用一个目标比特在辅助比特上的多受控 :math:`NOT` 门，并使用一些后续的 :math:`H` 门， :math:`X` 门，:math:`CX` 门作为激活函数可以获取 :math:`U_i` 和 :math:`U_w` 的归一化激活概率值：

.. math::
    \left|\phi_{i, w}\right\rangle|0\rangle_a \rightarrow \sum_{j=0}^{m-2} c_j|j\rangle|0\rangle_a+c_{m-1}|m-1\rangle|1\rangle_a

当输入 :math:`i` 的2进制串和 :math:`w` 完全一致时，该归一化概率值应为最大。

VQNet提供了 ``QuantumNeuron`` 模块实现该算法。首先初始化一个量子感知机 ``QuantumNeuron``。

.. code-block::

    perceptron = QuantumNeuron()

使用 ``gen_4bitstring_data`` 接口生成论文中的各种数据以及其类别标签。

.. code-block::

    training_label, test_label = perceptron.gen_4bitstring_data()

使用 ``train`` 接口遍历所有数据，可以获取最后训练好的量子感知器线路Uw。

.. code-block::

    trained_para = perceptron.train(training_label, test_label)

.. image:: ./images/QP-pic.png
   :width: 600 px
   :align: center

|

在测试数据上，可以获取测试数据上的准确率结果

.. image:: ./images/QP-acc.png
   :width: 600 px
   :align: center

|

量子自然梯度
===================================
量子机器学习模型一般使用梯度下降法对可变量子逻辑线路中参数进行优化。经典梯度下降法公式如下：

.. math:: \theta_{t+1} = \theta_t -\eta \nabla \mathcal{L}(\theta),

本质上，每次迭代时候，我们将计算参数空间下，梯度下降最陡的方向作为参数变化的方向。
在空间中任何一个方向，在局部范围内下降的速度都不如负梯度方向快。
不同空间上，最速下降方向的推导是依赖于参数微分的范数——距离度量。距离度量在这里起着核心作用，
不同的度量会得到不同的最速下降方向。对于经典优化问题中参数所处的欧几里得空间，最速下降方向就是负梯度方向。
即使如此，在参数优化的每一步，由于损失函数随着参数的变化，其参数空间发生变换。使得找到另一个更优的距离范数成为可能。

`量子自然梯度法 <https://arxiv.org/abs/1909.02108>`_ 借鉴经典自然梯度法的概念 `Amari (1998) <https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017746>`__ ，
我们改为将优化问题视为给定输入的可能输出值的概率分布（即，最大似然估计），则更好的方法是在分布
空间中执行梯度下降，它相对于参数化是无量纲和不变的. 因此，无论参数化如何，每个优化步骤总是会为每个参数选择最佳步长。
在量子机器学习任务中，量子态空间拥有一个独特的不变度量张量，称为 Fubini-Study 度量张量 :math:`g_{ij}`。
该张量将量子线路参数空间中的最速下降转换为分布空间中的最速下降。
量子自然梯度的公式如下：

.. math:: \theta_{t+1} = \theta_t - \eta g^{+}(\theta_t)\nabla \mathcal{L}(\theta),

其中 :math:`g^{+}` 是伪逆。

以下我们基于VQNet实现对一个量子变分线路参数进行量子自然梯度优化的例子，其中 `wrapper_calculate_qng` 是需要加到待计算量子自然梯度的模型的forward函数的装饰器。
通过 `pyvqnet.qnn.vqc.QNG` 的 量子自然梯度优化器，可对模型注册的 `Parameter` 类型的参数优化。

我们的目标是使如下的量子变分线路的期望最小，可见其中含有两层共3个量子含参逻辑门，第一层由0和1比特上的 RZ, RY 逻辑门构成，第二层由2比特上的RX 逻辑门构成。

.. image:: ./images/qng_all_cir.png
   :width: 600 px
   :align: center

|

.. code-block::


    import sys
    sys.path.insert(0, "../")
    import numpy as np
    import pyvqnet
    from pyvqnet.qnn import vqc
    from pyvqnet.qnn.vqc import wrapper_calculate_qng
    from pyvqnet.tensor import QTensor
    import matplotlib.pyplot as plt



    class Hmodel(vqc.Module):
        def __init__(self, num_wires, dtype,init_t):
            super(Hmodel, self).__init__()
            self._num_wires = num_wires
            self._dtype = dtype
            self.qm = vqc.QMachine(num_wires, dtype=dtype)

            self.p = pyvqnet.nn.Parameter([4], dtype=pyvqnet.kfloat64)
            self.p.init_from_tensor(init_t)
            self.ma = vqc.MeasureAll(obs={"Y0":1})

        @wrapper_calculate_qng
        def forward(self, x, *args, **kwargs):
            self.qm.reset_states(1)
            vqc.ry(q_machine=self.qm, wires=0, params=np.pi / 4)
            vqc.ry(q_machine=self.qm, wires=1, params=np.pi / 3)
            vqc.ry(q_machine=self.qm, wires=2, params=np.pi / 7)

            # V0(theta0, theta1): Parametrized layer 0
            vqc.rz(q_machine=self.qm, wires=0, params=self.p[0])
            vqc.rz(q_machine=self.qm, wires=1, params=self.p[1])

            # W1: non-parametrized gates
            vqc.cnot(q_machine=self.qm, wires=[0, 1])
            vqc.cnot(q_machine=self.qm, wires=[1, 2])

            # V_1(theta2, theta3): Parametrized layer 1
            vqc.ry(q_machine=self.qm, params=self.p[2], wires=1)
            vqc.rx(q_machine=self.qm, params=self.p[3], wires=2)

            # W2: non-parametrized gates
            vqc.cnot(q_machine=self.qm, wires=[0, 1])
            vqc.cnot(q_machine=self.qm, wires=[1, 2])

            return self.ma(q_machine=self.qm)



    class Hmodel2(vqc.Module):
        def __init__(self, num_wires, dtype,init_t):
            super(Hmodel2, self).__init__()
            self._num_wires = num_wires
            self._dtype = dtype
            self.qm = vqc.QMachine(num_wires, dtype=dtype)

            self.p = pyvqnet.nn.Parameter([4], dtype=pyvqnet.kfloat64)
            self.p.init_from_tensor(init_t)
            self.ma = vqc.MeasureAll(obs={"Y0":1})

        def forward(self, x, *args, **kwargs):
            self.qm.reset_states(1)
            vqc.ry(q_machine=self.qm, wires=0, params=np.pi / 4)
            vqc.ry(q_machine=self.qm, wires=1, params=np.pi / 3)
            vqc.ry(q_machine=self.qm, wires=2, params=np.pi / 7)

            # V0(theta0, theta1): Parametrized layer 0
            vqc.rz(q_machine=self.qm, wires=0, params=self.p[0])
            vqc.rz(q_machine=self.qm, wires=1, params=self.p[1])

            # W1: non-parametrized gates
            vqc.cnot(q_machine=self.qm, wires=[0, 1])
            vqc.cnot(q_machine=self.qm, wires=[1, 2])

            # V_1(theta2, theta3): Parametrized layer 1
            vqc.ry(q_machine=self.qm, params=self.p[2], wires=1)
            vqc.rx(q_machine=self.qm, params=self.p[3], wires=2)

            # W2: non-parametrized gates
            vqc.cnot(q_machine=self.qm, wires=[0, 1])
            vqc.cnot(q_machine=self.qm, wires=[1, 2])

            return self.ma(q_machine=self.qm)


使用SGD经典梯度下降法作为基线比较两者在相同迭代次数下的损失值变化情况，可见使用量子自然梯度，该损失函数下降更快。

.. code-block::

    steps = range(200)

    x = QTensor([0.432, -0.123, 0.543, 0.233],
                dtype=pyvqnet.kfloat64)
    qng_model = Hmodel(3, pyvqnet.kcomplex128,x)
    qng = pyvqnet.qnn.vqc.QNG(qng_model, 0.01)
    qng_cost = []
    for s in steps:
        qng.zero_grad()
        qng.step(None)
        yy = qng_model(None).to_numpy().reshape([1])
        qng_cost.append(yy)

    x = QTensor([0.432, -0.123, 0.543, 0.233],
                requires_grad=True,
                dtype=pyvqnet.kfloat64)
    qng_model = Hmodel2(3, pyvqnet.kcomplex128,x)
    sgd = pyvqnet.optim.SGD(qng_model.parameters(), lr=0.01)
    sgd_cost = []
    for s in steps:
        
        sgd.zero_grad()
        y = qng_model(None)
        y.backward()
        sgd.step()

        sgd_cost.append(y.to_numpy().reshape([1]))


    plt.style.use("seaborn")
    plt.plot(qng_cost, "b", label="Quantum natural gradient descent")
    plt.plot(sgd_cost, "g", label="Vanilla gradient descent")

    plt.ylabel("Cost function value")
    plt.xlabel("Optimization steps")
    plt.legend()
    plt.savefig('qng_new_compare.png')



.. image:: ./images/qng_vs_sgd.png
   :width: 600 px
   :align: center

|

随机参数偏移算法
===================================

在量子变分线路中，使用参数偏移法 `parameter-shift` 计算量子参数的梯度是一种常用的方法。
参数偏移法并不普遍适用所有的量子含参逻辑门。
在它不成立（或不知道成立）的情况下，我们要么必须将门分解为兼容的门，要么使用梯度的替代估计器，例如有限差分近似。
但是，由于增加了电路复杂性或梯度值中的潜在误差，这两种替代方案都可能存在缺陷。
Banchi 和 Crooks 1 发现一种可以适用在任一酉矩阵量子逻辑门上的 `随机参数偏移算法(Stochastic Parameter-Shift Rule) <https://arxiv.org/abs/2005.10299>`_ 。

下面展示适用VQNet对一个量子变分线路使用随机参数偏移法计算梯度的示例。其中， **pyqpanda建议版本为3.7.12** 。示例线路定义如下：

.. code-block::

    import pyqpanda as pq
    import numpy as np
    from pyvqnet.qnn.measure import expval
    from scipy.linalg import expm
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import matplotlib.pyplot as plt


    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    q = machine.qAlloc_many(2)
    c = machine.cAlloc_many(2)

    # some basic Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])

    def Generator(theta1, theta2, theta3):
        G = theta1.item() * np.kron(X, I) - \
            theta2 * np.kron(Z, X) + \
            theta3 * np.kron(I, X)
        return G

    def pq_demo_circuit(gate_pars):
        G = Generator(*gate_pars)
        G = expm(-1j * G)
        x = G.flatten().tolist()

        cir = pq.matrix_decompose(q, x)
        m_prog = pq.QProg()
        m_prog.insert(cir)
        pauli_dict = {'Z0': 1}
        exp2 = expval(machine, m_prog, pauli_dict, q)
        return exp2

随机参数偏移法首先随机从[0,1]的均匀分布中采样一个变量s，接着对线路分别进行如下的酉矩阵变换：

     a) :math:`e^{i(1-s)(\hat{H} + \theta\hat{V})}`
     b) :math:`e^{+i\tfrac{\pi}{4}\hat{V}}`
     c) :math:`e^{is(\hat{H} + \theta\hat{V})}`

其中 :math:`\hat{V}` 是一个泡利算符的张量积， :math:`\hat{H}` 是任意泡利算符张量积的线性组合。
此时获取的观测量的期望值我们定义为 :math:`\langle r_+ \rangle` 。

.. code-block::

    def pq_SPSRgates(gate_pars, s, sign):
        G = Generator(*gate_pars)
        # step a)
        G1 = expm(1j * (1 - s) * G)
        x = G1.flatten().tolist()

        cir = pq.matrix_decompose(q, x)
        m_prog = pq.QProg()
        m_prog.insert(cir)

        # step b)
        G2 = expm(1j * sign * np.pi / 4 * X)
        x = G2.flatten().tolist()
        cir = pq.matrix_decompose(q[0], x)
        m_prog.insert(cir)

        # step c)
        G3 = expm(1j * s * G)
        x = G3.flatten().tolist()
        cir = pq.matrix_decompose(q, x)
        m_prog.insert(cir)
        pauli_dict = {'Z0': 1}
        exp2 = expval(machine, m_prog, pauli_dict, q)
        return exp2

将上一步骤中 :math:`\tfrac{\pi}{4}` 变成  :math:`-\tfrac{\pi}{4}`，
重复进行 a, b, c 操作，获取观测量的期望 :math:`\langle r_- \rangle` 。

随机参数偏移算法计算的梯度公式如下：

 .. math::

     \mathbb{E}_{s\in\mathcal{U}[0,1]}[\langle r_+ \rangle - \langle r_-\rangle]

我们画出使用随机参数偏移法计算的参数 :math:`\theta_1` 梯度与观测量期望的之间的关系。
通过观察可见，观测量期望符合 :math:`\cos(2\theta_1)` 的函数形式；而使用随机参数偏移法计算梯度
符合 :math:`-2\sin(2\theta_1)` , 正好是 :math:`\cos(2\theta_1)` 的微分。

.. code-block::

    theta2, theta3 = -0.15, 1.6
    angles = np.linspace(0, 2 * np.pi, 50)
    pos_vals = np.array([[
        pq_SPSRgates([theta1, theta2, theta3], s=s, sign=+1)
        for s in np.random.uniform(size=10)
    ] for theta1 in angles])
    neg_vals = np.array([[
        pq_SPSRgates([theta1, theta2, theta3], s=s, sign=-1)
        for s in np.random.uniform(size=10)
    ] for theta1 in angles])

    # Plot the results
    evals = [pq_demo_circuit([theta1, theta2, theta3]) for theta1 in angles]
    spsr_vals = (pos_vals - neg_vals).mean(axis=1)
    plt.plot(angles, evals, 'b', label="Expectation Value")
    plt.plot(angles, spsr_vals, 'r', label="Stochastic parameter-shift rule")
    plt.xlabel("theta1")
    plt.legend()
    plt.title("VQNet")
    plt.show()

.. image:: ./images/stochastic_parameter-shift.png
   :width: 600 px
   :align: center

|


双随机梯度下降
===================================

在变分量子算法中，参数化量子电路通过经典梯度下降法进行优化，以最小化期望函数值。
虽然可以在经典模拟中分析计算期望值，在量子硬件上，程序仅限于从期望值中采样；随着样本数量以及shots次数的增加，这种方式获得的期望值会收敛于理论期望值，但可能永远得到准确值。
Sweke 等人 在 `论文 <https://arxiv.org/abs/1910.01155>`_ 中发现了一种双随机梯度下降法。
在本文中，他们证明了使用有限数量的测量样本（或shots）来估计梯度的量子梯度下降是随机梯度下降的一种形式。
此外，如果优化涉及期望值的线性组合（例如 VQE），从该线性组合中的项中抽样可以进一步减少所需的时间复杂度。

VQNet实现了该算法的一个示例：使用VQE 求解目标Hamiltonian的基态能量。注意此处我们设置量子线路观测的次数shots仅为1次。

.. math::

    H = \begin{bmatrix}
          8 & 4 & 0 & -6\\
          4 & 0 & 4 & 0\\
          0 & 4 & 8 & 0\\
          -6 & 0 & 0 & 0
        \end{bmatrix}.

.. code-block::

    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.template import StronglyEntanglingTemplate
    from pyvqnet.qnn.measure import Hermitian_expval
    from pyvqnet.qnn import QuantumLayerV2
    from pyvqnet.optim import SGD
    import pyvqnet._core as _core
    from pyvqnet.tensor import QTensor
    from matplotlib import pyplot as plt

    num_layers = 2
    num_wires = 2
    eta = 0.01
    steps = 200
    n = 1
    param_shape = [2, 2, 3]
    shots = 1

    H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])

    # some basic Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    init_params = np.random.uniform(low=0,
                                    high=2 * np.pi,
                                    size=param_shape)

    def pq_circuit(params):
        params = params.reshape(param_shape)
        num_qubits = 2

        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)
        circuit = StronglyEntanglingTemplate(params, num_qubits=num_qubits)
        qcir = circuit.create_circuit(qubits)
        prog = pq.QProg()
        prog.insert(qcir)
        machine.directly_run(prog)
        result = machine.get_qstate()
        return result


该示例中的哈密顿量是厄密特矩阵，我们总是可以将其表示为泡利矩阵的总和。

.. math::

    H = \sum_{i,j=0,1,2,3} a_{i,j} (\sigma_i\otimes \sigma_j),

其中

.. math::

    a_{i,j} = \frac{1}{4}\text{tr}[(\sigma_i\otimes \sigma_j )H], ~~ \sigma = \{I, X, Y, Z\}.

代入以上公式，可见

.. math::

    H = 4  + 2I\otimes X + 4I \otimes Z - X\otimes X + 5 Y\otimes Y + 2Z\otimes X.

为了执行“双随机”梯度下降，我们简单地应用随机梯度下降方法，但另外也均匀采样每个优化步骤的哈密顿期望项的子集。
vqe_func_analytic()函数是使用参数偏移计算理论梯度，vqe_func_shots()则是使用随机采样值以及随机采样哈密顿期望子集的“双随机”梯度计算。

.. code-block::

    terms = np.array([
        2 * np.kron(I, X),
        4 * np.kron(I, Z),
        -np.kron(X, X),
        5 * np.kron(Y, Y),
        2 * np.kron(Z, X),
    ])


    def vqe_func_analytic(input, init_params):
        qstate = pq_circuit(init_params)
        expval = Hermitian_expval(H, qstate, [0, 1], 2)
        return  expval

    def vqe_func_shots(input, init_params):
        qstate = pq_circuit(init_params)
        idx = np.random.choice(np.arange(5), size=n, replace=False)
        A = np.sum(terms[idx], axis=0)
        expval = Hermitian_expval(A, qstate, [0, 1], 2, shots)
        return 4 + (5 / 1) * expval


使用VQNet进行参数优化，对比损失函数的曲线，由于双随机梯度下降法每次仅计算H的部分泡利算符和，
故使用其平均值才能代表最终观测量的期望结果，这里使用滑动平均moving_average()进行计算。

.. code-block::

    ##############################################################################
    # Optimizing the circuit using gradient descent via the parameter-shift rule:
    qlayer_ana = QuantumLayerV2(vqe_func_analytic, 2*2*3 )
    qlayer_shots = QuantumLayerV2(vqe_func_shots, 2*2*3 )
    cost_sgd = []
    cost_dsgd = []
    temp = _core.Tensor(init_params)
    _core.vqnet.copyTensor(temp, qlayer_ana.m_para.data)
    opti_ana = SGD(qlayer_ana.parameters())

    _core.vqnet.copyTensor(temp, qlayer_shots.m_para.data)
    opti_shots = SGD(qlayer_shots.parameters())

    for i in range(steps):
        opti_ana.zero_grad()
        loss = qlayer_ana(QTensor([[1.0]]))

        loss.backward()
        cost_sgd.append(loss.item())
        opti_ana._step()

    for i in range(steps+50):
        opti_shots.zero_grad()
        loss = qlayer_shots(QTensor([[1.0]]))

        loss.backward()
        cost_dsgd.append(loss.item())
        opti_shots._step()

    def moving_average(data, n=3):
        ret = np.cumsum(data, dtype=np.float64)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    ta = moving_average(np.array(cost_dsgd), n=50)
    ta = ta[:-26]
    average = np.vstack([np.arange(25, 200),ta ])
    final_param = qlayer_shots.parameters()[0].to_numpy()
    print("Doubly stochastic gradient descent min energy = ", vqe_func_analytic(QTensor([1]),final_param))
    final_param  = qlayer_ana.parameters()[0].to_numpy()
    print("stochastic gradient descent min energy = ", vqe_func_analytic(QTensor([1]),final_param))

    plt.plot(cost_sgd, label="Vanilla gradient descent")
    plt.plot(cost_dsgd, ".", label="Doubly QSGD")
    plt.plot(average[0], average[1], "--", label="Doubly QSGD (moving average)")

    plt.ylabel("Cost function value")
    plt.xlabel("Optimization steps")
    plt.xlim(-2, 200)
    plt.legend()
    plt.show()

    #Doubly stochastic gradient descent min energy =  -4.337801834749975
    #stochastic gradient descent min energy =  -4.531484333030544

.. image:: ./images/dsgd.png
   :width: 600 px
   :align: center

|

基于梯度的剪枝
===================================

下面的例子实现了论文 `Towards Efficient On-Chip Training of Quantum Neural Networks <https://openreview.net/forum?id=vKefw-zKOft>`_ 中的算法。
通过仔细研究量子变分线路中参数的过程，作者观察到小梯度在量子噪声下往往具有较大的相对变化甚至错误的方向。
此外，并非所有梯度计算对于训练过程都是必需的，尤其是对于小幅度梯度。
受此启发，作者提出了一种概率梯度修剪方法来预测并仅计算高可靠性的梯度。
该方法可以减少噪声影响，还可以节省在真实量子机器上运行所需的电路数量。

在gradient based pruning算法中，对于参数的优化过程，划分了积累窗口和修剪窗口两个阶段，所有训练时期分成一个重复的累积窗口，然后是一个修剪窗口。 概率梯度修剪方法中有三个重要的超参数：

    * 累积窗口宽度 :math:`\omega_a` ，
    * 修剪比例 :math:`r` ，
    * 修剪窗口宽度 :math:`\omega_p` .

在累积窗口中，算法收集每个训练步骤中的梯度信息。 在修剪窗口的每一步中，算法根据从累积窗口和修剪比率收集的信息，
概率地免除一些梯度的计算。

.. image:: ./images/gbp_arch.png
   :width: 600 px
   :align: center

|

剪枝比例 :math:`r` ,累积窗口宽度 :math:`\omega_a` 和剪枝窗口宽度 :math:`\omega_p` 分别决定了梯度趋势评估的可靠性。
因此，我们的概率梯度修剪方法节省的时间百分比是 :math:`r\tfrac{\omega_p}{\omega_a +\omega_p}\times 100\%`。
以下是运用梯度剪枝方法在QVC分类任务的示例。

.. code-block::

    import random
    import numpy as np
    import pyqpanda as pq
    
    from pyvqnet.data import data_generator as dataloader
    from pyvqnet.nn.module import Module
    from pyvqnet.optim import sgd
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn import Gradient_Prune_Instance
    random.seed(1234)

    qvc_train_data = [
        0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 0
    ]
    qvc_test_data = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]


    def qvc_circuits(x, weights, qlist, clist, machine):#pylint:disable=unused-argument
        """
        Quantum circuits run function
        """
        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits) - 1):
                cir.insert(pq.CNOT(nqubits[i], nqubits[i + 1]))
            cir.insert(pq.CNOT(nqubits[len(nqubits) - 1], nqubits[0]))
            return cir

        def build_circult(weights, xx, nqubits):
            def Rot(weights_j, qubits):#pylint:disable=invalid-name
                circult = pq.QCircuit()
                circult.insert(pq.RZ(qubits, weights_j[0]))
                circult.insert(pq.RY(qubits, weights_j[1]))
                circult.insert(pq.RZ(qubits, weights_j[2]))
                return circult

            def basisstate():
                circult = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i] == 1:
                        circult.insert(pq.X(nqubits[i]))
                return circult

            circult = pq.QCircuit()
            circult.insert(basisstate())

            for i in range(weights.shape[0]):

                weights_i = weights[i, :, :]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circult.insert(Rot(weights_j, nqubits[j]))
                cnots = get_cnot(nqubits)
                circult.insert(cnots)

            circult.insert(pq.Z(nqubits[0]))

            prog = pq.QProg()

            prog.insert(circult)
            return prog

        weights = weights.reshape([2, 4, 3])
        prog = build_circult(weights, x, qlist)
        prob = machine.prob_run_dict(prog, qlist[0], -1)
        prob = list(prob.values())

        return prob


    def qvc_circuits2(x, weights, qlist, clist, machine):#pylint:disable=unused-argument
        """
        Quantum circuits run function
        """
        prog = pq.QProg()
        circult = pq.QCircuit()
        circult.insert(pq.RZ(qlist[0], x[0]))
        circult.insert(pq.RZ(qlist[1], x[1]))
        circult.insert(pq.CNOT(qlist[0], qlist[1]))
        circult.insert(pq.CNOT(qlist[1], qlist[2]))
        circult.insert(pq.CNOT(qlist[2], qlist[3]))
        circult.insert(pq.RY(qlist[0], weights[0]))
        circult.insert(pq.RY(qlist[1], weights[1]))
        circult.insert(pq.RY(qlist[2], weights[2]))

        circult.insert(pq.CNOT(qlist[0], qlist[1]))
        circult.insert(pq.CNOT(qlist[1], qlist[2]))
        circult.insert(pq.CNOT(qlist[2], qlist[3]))
        prog.insert(circult)
        prob = machine.prob_run_dict(prog, qlist[0], -1)
        prob = list(prob.values())

        return prob

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits, 24, "cpu", 4)

        def forward(self, x):
            y = self.qvc(x)
            #y = self.qvc2(y)
            return y


    def get_data(dataset_str):
        """
        Tranform data to valid form
        """
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)

        else:
            datasets = np.array(qvc_test_data)

        datasets = datasets.reshape([-1, 5])
        data = datasets[:, :-1]
        label = datasets[:, -1].astype(int)
        label = np.eye(2)[label].reshape(-1, 2)
        return data, label

    def get_accuary(result, label):
        result, label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result, axis=1) == np.argmax(label, 1))
        return score


我们使用 ``Gradient_Prune_Instance`` 类，
输入24为当前模型的参数个数。裁剪比例 `prune_ratio` 输入为0.5，
累计窗口大小 `accumulation_window_size` 为4，
剪枝窗口 `pruning_window_size` 为2。
当每次运行反向传播部分代码，在优化器 ``step`` 之前，
运行 ``Gradient_Prune_Instance`` 的 ``step`` 函数。

.. code-block::

    def run():
        """
        Main run function
        """
        model = Model()

        optimizer = sgd.SGD(model.parameters(), lr=0.5)
        batch_size = 3
        epoch = 10
        loss = CategoricalCrossEntropy()
        print("start training..............")
        model.train()

        datas, labels = get_data("train")
        print(datas)
        print(labels)
        print(datas.shape)


        GBP_HELPER = Gradient_Prune_Instance(param_num = 24,prune_ratio=0.5,accumulation_window_size=4,pruning_window_size=2)
        for i in range(epoch):
            count = 0
            sum_loss = 0
            accuary = 0
            t = 0
            for data, label in dataloader(datas, labels, batch_size, False):
                optimizer.zero_grad()
                data, label = QTensor(data), QTensor(label)

                result = model(data)

                loss_b = loss(label, result)

                loss_b.backward()
                
                GBP_HELPER.step(model.parameters())
                optimizer._step()
                sum_loss += loss_b.item()
                count += batch_size
                accuary += get_accuary(result, label)
                t = t + 1

            print(
                f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}"
            )
        print("start testing..............")
        model.eval()
        count = 0

        test_data, test_label = get_data("test")
        test_batch_size = 1
        accuary = 0
        sum_loss = 0
        for testd, testl in dataloader(test_data, test_label, test_batch_size):
            testd = QTensor(testd)
            test_result = model(testd)
            test_loss = loss(testl, test_result)
            sum_loss += test_loss
            count += test_batch_size
            accuary += get_accuary(test_result, testl)
        print(
            f"test:--------------->loss:{sum_loss/count} #####accuray:{accuary/count}"
        )


    if __name__ == "__main__":

        run()

    # epoch:0, #### loss:0.2255942871173223 #####accuray:0.5833333333333334
    # epoch:1, #### loss:0.1989427705605825 #####accuray:1.0
    # epoch:2, #### loss:0.16489211718241373 #####accuray:1.0
    # epoch:3, #### loss:0.13245886812607446 #####accuray:1.0
    # epoch:4, #### loss:0.11463981121778488 #####accuray:1.0
    # epoch:5, #### loss:0.1078591321905454 #####accuray:1.0
    # epoch:6, #### loss:0.10561319688955943 #####accuray:1.0
    # epoch:7, #### loss:0.10483601937691371 #####accuray:1.0
    # epoch:8, #### loss:0.10457512239615123 #####accuray:1.0
    # epoch:9, #### loss:0.10448987782001495 #####accuray:1.0
    # start testing..............
    # test:--------------->loss:[0.3134713] #####accuray:1.0


量子奇异值分解
===================================

下面例子实现论文 `Variational Quantum Singular Value Decomposition <https://arxiv.org/abs/2006.02336>`_ 中的算法。 

奇异值分解 (Singular Value Decomposition，简称 ``SVD``) 是线性代数中一种重要的矩阵分解，它作为特征分解在任意维数矩阵上的推广，在机器学习领域中被广泛应用，常用于矩阵压缩、推荐系统以及自然语言处理等。

变分量子奇异值分解(Variational Quantum Singular Value Decomposition，简称 ``QSVD``)将SVD转换成优化问题，并通过变分量子线路求解。

论文中将矩阵奇异值分解分解成四个步骤求解：

    1. 输入带分解矩阵 :math:`\mathbf{M}`，想压缩到的阶数 :math:`\mathbf{T}`, 权重 :math:`\mathbf{W}`，参数话的酉矩阵 :math:`\mathbf{U}(\theta)` 和 :math:`\mathbf{V}(\phi)`
    2. 搭建量子神经网络估算奇异值 :math:`m_j = Re\langle\psi_j\mid U(\theta)^\dagger M V(\phi) \mid\psi_j\rangle`，并最大化加权奇异值的和 :math:`L(\theta, \phi) = \sum_{j=1}^{T} q_j \cdot \operatorname{Re}\langle\psi_j \mid U(\theta)^\dagger MV(\phi) \mid \psi_j\rangle`, 其中，加权是为了让计算出的奇异值从大到小排列
    3. 读出最大化时参数值，计算出 :math:`\mathbf{U}(\alpha^{*})` 和 :math:`\mathbf{V}(\beta^{*})`
    4. 输出结果: 奇异值 :math:`m_1, \dots, m_r`，和奇异矩阵 :math:`\mathbf{U}(\alpha^{*})` 和 :math:`\mathbf{V}(\beta^{*})`

.. image:: ./images/qsvd.png
   :width: 700 px
   :align: center

|

伪代码如下：

.. image:: ./images/qsvd_algorithm.png
   :width: 700 px
   :align: center

|

量子线路设计如下：

.. code-block::

    q0: ──RY(v_theta0)────RZ(v_theta3)────●─────────X────RY(v_theta6)─────RZ(v_theta9)────●─────────X────RY(v_theta12)────RZ(v_theta15)────●─────────X────RY(v_theta18)────RZ(v_theta21)────●─────────X────RY(v_theta24)────RZ(v_theta27)────●─────────X────RY(v_theta30)────RZ(v_theta33)────●─────────X────RY(v_theta36)────RZ(v_theta39)────●─────────X────RY(v_theta42)────RZ(v_theta45)────●─────────X────RY(v_theta48)────RZ(v_theta51)────●─────────X────RY(v_theta54)────RZ(v_theta57)────●─────────X────RY(v_theta60)────RZ(v_theta63)────●─────────X────RY(v_theta66)────RZ(v_theta69)────●─────────X────RY(v_theta72)────RZ(v_theta75)────●─────────X────RY(v_theta78)────RZ(v_theta81)────●─────────X────RY(v_theta84)────RZ(v_theta87)────●─────────X────RY(v_theta90)────RZ(v_theta93)────●─────────X────RY(v_theta96)────RZ(v_theta99)─────●─────────X────RY(v_theta102)────RZ(v_theta105)────●─────────X────RY(v_theta108)────RZ(v_theta111)────●─────────X────RY(v_theta114)────RZ(v_theta117)────●─────────X──
                                          │         │                                     │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                      │         │                                       │         │                                        │         │                                        │         │                                        │         │
    q1: ──RY(v_theta1)────RZ(v_theta4)────X────●────┼────RY(v_theta7)────RZ(v_theta10)────X────●────┼────RY(v_theta13)────RZ(v_theta16)────X────●────┼────RY(v_theta19)────RZ(v_theta22)────X────●────┼────RY(v_theta25)────RZ(v_theta28)────X────●────┼────RY(v_theta31)────RZ(v_theta34)────X────●────┼────RY(v_theta37)────RZ(v_theta40)────X────●────┼────RY(v_theta43)────RZ(v_theta46)────X────●────┼────RY(v_theta49)────RZ(v_theta52)────X────●────┼────RY(v_theta55)────RZ(v_theta58)────X────●────┼────RY(v_theta61)────RZ(v_theta64)────X────●────┼────RY(v_theta67)────RZ(v_theta70)────X────●────┼────RY(v_theta73)────RZ(v_theta76)────X────●────┼────RY(v_theta79)────RZ(v_theta82)────X────●────┼────RY(v_theta85)────RZ(v_theta88)────X────●────┼────RY(v_theta91)────RZ(v_theta94)────X────●────┼────RY(v_theta97)────RZ(v_theta100)────X────●────┼────RY(v_theta103)────RZ(v_theta106)────X────●────┼────RY(v_theta109)────RZ(v_theta112)────X────●────┼────RY(v_theta115)────RZ(v_theta118)────X────●────┼──
                                               │    │                                          │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                           │    │                                            │    │                                             │    │                                             │    │                                             │    │
    q2: ──RY(v_theta2)────RZ(v_theta5)─────────X────●────RY(v_theta8)────RZ(v_theta11)─────────X────●────RY(v_theta14)────RZ(v_theta17)─────────X────●────RY(v_theta20)────RZ(v_theta23)─────────X────●────RY(v_theta26)────RZ(v_theta29)─────────X────●────RY(v_theta32)────RZ(v_theta35)─────────X────●────RY(v_theta38)────RZ(v_theta41)─────────X────●────RY(v_theta44)────RZ(v_theta47)─────────X────●────RY(v_theta50)────RZ(v_theta53)─────────X────●────RY(v_theta56)────RZ(v_theta59)─────────X────●────RY(v_theta62)────RZ(v_theta65)─────────X────●────RY(v_theta68)────RZ(v_theta71)─────────X────●────RY(v_theta74)────RZ(v_theta77)─────────X────●────RY(v_theta80)────RZ(v_theta83)─────────X────●────RY(v_theta86)────RZ(v_theta89)─────────X────●────RY(v_theta92)────RZ(v_theta95)─────────X────●────RY(v_theta98)────RZ(v_theta101)─────────X────●────RY(v_theta104)────RZ(v_theta107)─────────X────●────RY(v_theta110)────RZ(v_theta113)─────────X────●────RY(v_theta116)────RZ(v_theta119)─────────X────●──


以下是具体QSVD实现代码:

.. code-block::

    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    import numpy as np
    import tqdm

    from pyvqnet.nn import Module
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.tensor import tensor
    from pyvqnet.optim import Adam
    from pyvqnet.qnn.measure import expval
    from pyvqnet.nn.parameter import Parameter
    from pyvqnet.dtype import *
    from pyqpanda import *
    import pyvqnet
    from pyvqnet.qnn.vqc import ry, QMachine, cnot, rz

    n_qubits = 3  # qbits number
    cir_depth = 20  # circuit depth
    N = 2**n_qubits
    rank = 8  # learning rank
    step = 7
    ITR = 200  # iterations
    LR = 0.02  # learning rate

    if step == 0:
        weight = QTensor(np.ones(rank))
    else:
        weight = QTensor(np.arange(rank * step, 0, -step),requires_grad=True,dtype=kfloat32).reshape((-1,1))

    # Define random seed
    np.random.seed(42)

    def mat_generator():
        '''
        Generate a random complex matrix
        '''
        matrix = np.random.randint(
            10, size=(N, N)) + 1j * np.random.randint(10, size=(N, N))
        return matrix


    # Generate matrix M which will be decomposed
    M = mat_generator()

    # m_copy is generated to error analysis
    m_copy = np.copy(M)

    # Print M
    print('Random matrix M is: ')
    print(M)

    # Get SVD results
    U, D, v_dagger = np.linalg.svd(M, full_matrices=True)
    # Random matrix M is: 
    # [[6.+1.j 3.+9.j 7.+3.j 4.+7.j 6.+6.j 9.+8.j 2.+7.j 6.+4.j]
    #  [7.+1.j 4.+4.j 3.+7.j 7.+9.j 7.+8.j 2.+8.j 5.+0.j 4.+8.j]
    #  [1.+6.j 7.+8.j 5.+7.j 1.+0.j 4.+7.j 0.+7.j 9.+2.j 5.+0.j]
    #  [8.+7.j 0.+2.j 9.+2.j 2.+0.j 6.+4.j 3.+9.j 8.+6.j 2.+9.j]
    #  [4.+8.j 2.+6.j 6.+8.j 4.+7.j 8.+1.j 6.+0.j 1.+6.j 3.+6.j]
    #  [8.+7.j 1.+4.j 9.+2.j 8.+7.j 9.+5.j 4.+2.j 1.+0.j 3.+2.j]
    #  [6.+4.j 7.+2.j 2.+0.j 0.+4.j 3.+9.j 1.+6.j 7.+6.j 3.+8.j]
    #  [1.+9.j 5.+9.j 5.+2.j 9.+6.j 3.+0.j 5.+3.j 1.+3.j 9.+4.j]]

    print(D)
    # [54.8348498 19.1814107 14.9886625 11.6141956 10.1592704  7.6022325
    #  5.8104054  3.30116  ]

    def vqc_circuit(matrix, para):
        qm = QMachine(3)

        qm.set_states(matrix)

        num = 0
        for _ in range(20):
            for i in range(3):
                ry(q_machine=qm, params=para[num], wires=i, num_wires=3)
                num += 1 

            for i in range(3):
                rz(q_machine=qm, params=para[num], wires=i, num_wires=3)
                num += 1

            for i in range(2):
                cnot(q_machine=qm, wires=[i, i+1], num_wires=3)

            cnot(q_machine=qm, wires=[2, 0], num_wires=3)

        return qm.states

    i_matrix = np.identity(N)

    class VQC_svd(Module):
        def __init__(self):
            super(VQC_svd, self).__init__()

            self.params_u = list()
            self.params_v = list()

            self.params_u_single = Parameter((120, ), dtype=kfloat32)
            self.params_v_single = Parameter((120, ), dtype=kfloat32)


        def forward(self):
            qm_u_list = list()
            qm_v_list = list()

            for i in range(8):

                qm = vqc_circuit(QTensor(i_matrix[i], dtype=kcomplex64).reshape((1,2,2,2)), self.params_u_single)
                qm_u_list.append(qm)    

            for i in range(8):
                qm = vqc_circuit(QTensor(i_matrix[i], dtype=kcomplex64).reshape((1,2,2,2)), self.params_v_single)
                qm_v_list.append(qm)    

            result = []


            for i in range(8):
                states_u = qm_u_list[i].reshape((1,-1))
                states_u = tensor.conj(states_u)

                states_v = qm_v_list[i].reshape((-1,1))

                pred = tensor.matmul(states_u, QTensor(M,dtype=kcomplex64))
                pred = tensor.matmul(pred, states_v)

                result.append(tensor.real(pred))

            return qm_u_list, qm_v_list, result


        def __call__(self):
            return self.forward()

    def loss_(x, w):
        loss = tensor.mul(x, w)
        return loss

    def run():

        model = VQC_svd()
        opt = Adam(model.parameters(), lr = 0.02)
        for itr in tqdm.tqdm(range(ITR)):

            opt.zero_grad()
            model.train()
            qm_u_list, qm_v_list, result = model()
            loss = 0
            for i in range(8):
                loss -= loss_(result[i], weight[i])
            if(itr % 20 == 0):
                print(loss)
                print(result)
            loss.backward()
            opt.step()

        pyvqnet.utils.storage.save_parameters(model.state_dict(),f"vqc_svd_{ITR}.model")

    def eval():
        model = VQC_svd()
        model_para = pyvqnet.utils.storage.load_parameters(f"vqc_svd_{ITR}.model")
        model.load_state_dict(model_para)
        qm_u_list, qm_v_list, result = model()
        # U is qm_u_list
        # V is qm_v_list
        print(result)

    if __name__=="__main__":
        run()
    
运行的loss以及奇异值结果：

.. code-block::

    [[-145.04752]]  ## 20/200 [00:56<09:30,  3.17s/it]
    [[[-5.9279256]], [[0.7229557]], [[12.809682]], [[-3.2357244]], [[-5.232873]], [[4.5523396]], [[0.9724817]], [[7.733829]]]
     [[-4836.0083]] ## 40/200 [02:08<10:11,  3.82s/it]
    [[[30.293152]], [[21.15204]], [[26.832254]], [[11.953516]], [[9.615778]], [[9.914136]], [[5.34158]], [[0.7990487]]]
     [[-5371.0034]] ## 60/200 [03:31<10:04,  4.32s/it]
    [[[52.829674]], [[16.831125]], [[15.112174]], [[12.098673]], [[9.9859915]], [[8.895033]], [[5.1445904]], [[-1.2537733]]]
     [[-5484.087]]  ## 80/200 [05:03<09:23,  4.69s/it]
    [[[54.775055]], [[16.41207]], [[15.00042]], [[13.043125]], [[9.884815]], [[8.17144]], [[5.8188157]], [[-0.5532891]]]
     [[-5516.793]]  ## 100/200 [06:41<08:23,  5.04s/it]
    [[[54.797073]], [[17.457108]], [[14.50795]], [[13.288734]], [[9.7749815]], [[7.900285]], [[5.7255745]], [[-0.2063196]]]
     [[-5531.2007]] ## 120/200 [08:24<07:08,  5.35s/it]
    [[[54.816666]], [[18.107487]], [[14.094158]], [[13.305479]], [[9.837374]], [[7.7387457]], [[5.6890383]], [[-0.1503702]]]
     [[-5539.823]]  ## 140/200 [10:11<05:20,  5.34s/it]
    [[[54.822754]], [[18.518795]], [[13.9633045]], [[13.136647]], [[9.929082]], [[7.647796]], [[5.6548705]], [[-0.2427776]]]
     [[-5545.748]]  ## 160/200 [12:00<03:37,  5.43s/it]
    [[[54.825073]], [[18.766531]], [[14.041204]], [[12.855356]], [[10.009973]], [[7.5971537]], [[5.6524153]], [[-0.3767563]]]
     [[-5550.124]]  ## 180/200 [13:50<01:49,  5.45s/it]
    [[[54.82772]], [[18.913624]], [[14.219269]], [[12.547045]], [[10.063704]], [[7.569273]], [[5.6508512]], [[-0.4574079]]]
     [[-5553.423]]  ## 200/200 [15:40<00:00,  4.70s/it]
    [[[54.829308]], [[19.001402]], [[14.423045]], [[12.262444]], [[10.100731]], [[7.5507345]], [[5.6469355]], [[-0.4976197]]]



变分量子线路的优化
===================================

VQNet当前提供4种方式对用户自定义的变分量子线路中的量子逻辑门进行优化：融合旋转门(commute_controlled_right，commute_controlled_left)，受控门交换(commute_controlled)，单比特逻辑门融合(single_qubit_ops_fuse)。

这里使用 `wrapper_compile` 装饰器对 `QModule` 定义的模型forward函数进行装饰，会默认连续调用 `commute_controlled_right`, `merge_rotations`, `single_qubit_ops_fuse` 三个规则进行线路优化。
最后通过 `op_history_summary` 接口，对 `QModule` 前向函数运行后产生的 `op_history` 的信息对比。


.. code-block::

    from functools import partial

    from pyvqnet.qnn.vqc import op_history_summary
    from pyvqnet.qnn.vqc import QModule
    from pyvqnet import tensor
    from pyvqnet.qnn.vqc import QMachine, wrapper_compile

    from pyvqnet.qnn.vqc import pauliy

    from pyvqnet.qnn.vqc import QMachine, ry,rz, ControlledPhaseShift, \
        rx, S, rot, isingxy,CSWAP, PauliX, T, MeasureAll, RZ, CZ, PhaseShift, u3, cnot, cry, toffoli, cy
    from pyvqnet.tensor import QTensor, tensor
    import pyvqnet

    class QModel_before(QModule):
        def __init__(self, num_wires, dtype):
            super(QModel_before, self).__init__()

            self._num_wires = num_wires
            self._dtype = dtype
            self.qm = QMachine(num_wires, dtype=dtype)
            self.qm.set_save_op_history_flag(True)
            self.cswap = CSWAP(wires=(0, 2, 1))
            self.cz = CZ(wires=[0, 2])

            self.paulix = PauliX(wires=2)

            self.s = S(wires=0)

            self.ps = PhaseShift(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    dtype=dtype)

            self.cps = ControlledPhaseShift(has_params=True,
                                            trainable=True,
                                            wires=(1, 0),
                                            dtype=dtype)
            self.t = T(wires=0)
            self.rz = RZ(has_params=True, wires=1, dtype=dtype)

            self.measure = MeasureAll(obs={
                'wires': [0],
                'observables': ['z'],
                'coefficient': [1]
            })

        def forward(self, x, *args, **kwargs):
            self.qm.reset_states(x.shape[0])
            self.cz(q_machine=self.qm)
            self.paulix(q_machine=self.qm)
            rx(q_machine=self.qm,wires=1,params = x[:,[0]])
            ry(q_machine=self.qm,wires=1,params = x[:,[1]])
            rz(q_machine=self.qm,wires=1,params = x[:,[2]])
            rot(q_machine=self.qm, params=x[:, 0:3], wires=(1, ), use_dagger=True)
            rot(q_machine=self.qm, params=x[:, 1:4], wires=(1, ), use_dagger=True)
            isingxy(q_machine=self.qm, params=x[:, [2]], wires=(0, 1))
            u3(q_machine=self.qm, params=x[:, 0:3], wires=1)
            self.s(q_machine=self.qm)
            self.cswap(q_machine=self.qm)
            cnot(q_machine=self.qm, wires=[0, 1])
            ry(q_machine=self.qm,wires=2,params = x[:,[1]])
            pauliy(q_machine=self.qm, wires=1)
            cry(q_machine=self.qm, params=1 / 2, wires=[0, 1])
            self.ps(q_machine=self.qm)
            self.cps(q_machine=self.qm)
            ry(q_machine=self.qm,wires=2,params = x[:,[1]])
            rz(q_machine=self.qm,wires=2,params = x[:,[2]])
            toffoli(q_machine=self.qm, wires=[0, 1, 2])
            self.t(q_machine=self.qm)

            cy(q_machine=self.qm, wires=(2, 1))
            ry(q_machine=self.qm,wires=1,params = x[:,[1]])
            self.rz(q_machine=self.qm)

            rlt = self.measure(q_machine=self.qm)

            return rlt
    class QModel(QModule):
        def __init__(self, num_wires, dtype):
            super(QModel, self).__init__()

            self._num_wires = num_wires
            self._dtype = dtype
            self.qm = QMachine(num_wires, dtype=dtype)

            self.cswap = CSWAP(wires=(0, 2, 1))
            self.cz = CZ(wires=[0, 2])

            self.paulix = PauliX(wires=2)

            self.s = S(wires=0)

            self.ps = PhaseShift(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    dtype=dtype)

            self.cps = ControlledPhaseShift(has_params=True,
                                            trainable=True,
                                            wires=(1, 0),
                                            dtype=dtype)
            self.t = T(wires=0)
            self.rz = RZ(has_params=True, wires=1, dtype=dtype)

            self.measure = MeasureAll(obs={
                'wires': [0],
                'observables': ['z'],
                'coefficient': [1]
            })

        @partial(wrapper_compile)
        def forward(self, x, *args, **kwargs):
            self.qm.reset_states(x.shape[0])
            self.cz(q_machine=self.qm)
            self.paulix(q_machine=self.qm)
            rx(q_machine=self.qm,wires=1,params = x[:,[0]])
            ry(q_machine=self.qm,wires=1,params = x[:,[1]])
            rz(q_machine=self.qm,wires=1,params = x[:,[2]])
            rot(q_machine=self.qm, params=x[:, 0:3], wires=(1, ), use_dagger=True)
            rot(q_machine=self.qm, params=x[:, 1:4], wires=(1, ), use_dagger=True)
            isingxy(q_machine=self.qm, params=x[:, [2]], wires=(0, 1))
            u3(q_machine=self.qm, params=x[:, 0:3], wires=1)
            self.s(q_machine=self.qm)
            self.cswap(q_machine=self.qm)
            cnot(q_machine=self.qm, wires=[0, 1])
            ry(q_machine=self.qm,wires=2,params = x[:,[1]])
            pauliy(q_machine=self.qm, wires=1)
            cry(q_machine=self.qm, params=1 / 2, wires=[0, 1])
            self.ps(q_machine=self.qm)
            self.cps(q_machine=self.qm)
            ry(q_machine=self.qm,wires=2,params = x[:,[1]])
            rz(q_machine=self.qm,wires=2,params = x[:,[2]])
            toffoli(q_machine=self.qm, wires=[0, 1, 2])
            self.t(q_machine=self.qm)

            cy(q_machine=self.qm, wires=(2, 1))
            ry(q_machine=self.qm,wires=1,params = x[:,[1]])
            self.rz(q_machine=self.qm)

            rlt = self.measure(q_machine=self.qm)

            return rlt

    import pyvqnet
    import pyvqnet.tensor as tensor
    input_x = tensor.QTensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                                dtype=pyvqnet.kfloat64)

    input_x.requires_grad = True
    num_wires = 3
    qunatum_model = QModel(num_wires=num_wires, dtype=pyvqnet.kcomplex128)
    qunatum_model_before = QModel_before(num_wires=num_wires, dtype=pyvqnet.kcomplex128)

    batch_y = qunatum_model(input_x)
    batch_y = qunatum_model_before(input_x)

    flatten_oph_names = []

    print("before")

    print(op_history_summary(qunatum_model_before.qm.op_history))
    flatten_oph_names = []
    for d in qunatum_model.compiled_op_historys:
            if "compile" in d.keys():
                oph = d["op_history"]
                for i in oph:
                    n = i["name"]
                    w = i["wires"]
                    p = i["params"]
                    flatten_oph_names.append({"name":n,"wires":w, "params": p})
    print("after")
    print(op_history_summary(qunatum_model.qm.op_history))


    # ###################Summary#######################
    # qubits num: 3
    # gates: {'cz': 1, 'paulix': 1, 'rx': 1, 'ry': 4, 'rz': 3, 'rot': 2, 'isingxy': 1, 'u3': 1, 's': 1, 'cswap': 1, 'cnot': 1, 'pauliy': 1, 'cry': 1, 'phaseshift': 1, 'controlledphaseshift': 1, 'toffoli': 1, 't': 1, 'cy': 1}
    # total gates: 24
    # total parameter gates: 15
    # total parameters: 21
    # #################################################
        
    # after


    # ###################Summary#######################
    # qubits num: 3
    # gates: {'cz': 1, 'rot': 7, 'isingxy': 1, 'u3': 1, 'cswap': 1, 'cnot': 1, 'cry': 1, 'controlledphaseshift': 1, 'toffoli': 1, 'cy': 1}
    # total gates: 16
    # total parameter gates: 11
    # total parameters: 27
    # #################################################


量子dropout实现
===================================

神经网络（NN）通常需要具有大量可训练参数的高度灵活的模型，以便学习特定的基础函数（或数据分布）。然而，仅仅能够以较低的样本内误差进行学习是不够的；泛化能力也是非常重要的。

表现力强的模型可能会出现过拟合问题，这意味着它们在训练数据上训练得太好，结果在新的未见数据上表现不佳。出现这种情况的原因是，模型学会了训练数据中的噪声，而不是可泛化到新数据的基本模式。

Dropout是经典深度神经网络（DNN）的一种常用技术，可防止计算单元过于专业化，降低过拟合风险。

论文 `A General Approach to Dropout in Quantum Neural Networks` 表明，使用过度参数化的 QNN 模型可以消除大量局部极小值，从而改变优化格局。一方面，参数数量的增加会使训练更快、更容易，但另一方面，它可能会使模型过度拟合数据。这也与重复编码经典数据以实现计算的非线性密切相关。正因如此，受经典 DNN 的启发，我们可以考虑在 QNN 中应用某种 "dropout" 技术。这相当于在训练过程中随机丢弃一些（组）参数化门，以达到更好的泛化效果。

接下来我将通过下面样例了展示如何利用量子dropout来避免在量子机器学习算法在训练中出现的过拟合问题，我们将dropout掉的逻辑门的参数设置为0来进行dropout。

首先是导入相应包

.. code-block::

    import pyvqnet 
    from pyvqnet.qnn.vqc import *
    import numpy as np
    from pyvqnet import tensor
    from pyvqnet.qnn.vqc.qmeasure import expval
    from sklearn.model_selection import train_test_split
    from matplotlib import ticker
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler

搭建简单的量子线路

.. code-block::

    def embedding(x, wires, qmachine):
        # Encodes the datum multiple times in the register,
        for i in wires:
            ry(qmachine, i, tensor.asin(x[i]))
        for i in wires:
            rz(qmachine, i, tensor.acos(x[i] ** 2))


    def var_ansatz(
        theta, wires, qmachine, rotations=[ry, rz, rx], entangler=cnot, keep_rotation=None
    ):

        # the length of `rotations` defines the number of inner layers
        N = len(wires)
        wires = list(wires)

        counter = 0
        # keep_rotations contains a list per each inner_layer
        for rots in keep_rotation:
            # we cicle over the elements of the lists inside keep_rotation
            for qb, keep_or_drop in enumerate(rots):
                rot = rotations[counter]  # each inner layer can have a different rotation

                angle = theta[counter * N + qb]
                # conditional statement implementing dropout
                # if `keep_or_drop` is negative the rotation is dropped
                if keep_or_drop < 0:
                    angle_drop = tensor.QTensor(0.0)
                else:
                    angle_drop = angle
                    
                rot(qmachine, wires[qb], angle_drop)
            for qb in wires[:-1]:
                entangler(qmachine, wires=[wires[qb], wires[qb + 1]])
            counter += 1

    # quantum circuit qubits and params
    n_qubits = 5
    inner_layers = 3
    params_per_layer = n_qubits * inner_layers


    def qnn_circuit(x, theta, keep_rot, n_qubits, layers, qm):
        for i in range(layers):
            embedding(x, wires=range(n_qubits), qmachine=qm)

            keep_rotation = keep_rot[i]

            var_ansatz(
                theta[i * params_per_layer : (i + 1) * params_per_layer],
                wires=range(n_qubits),qmachine=qm,
                entangler=cnot,
                keep_rotation=keep_rotation,
            )
        
        return expval(qm, 0, PauliZ()) 

生成dropout列表，根据dropout列表来对量子线路中的逻辑门随机dropout

.. code-block::

    def make_dropout(rng, layer_drop_rate, rot_drop_rate, layers):
        drop_layers = []

        for lay in range(layers):
            out = np.random.choice(np.array(range(2)), p=np.array([1 - layer_drop_rate, layer_drop_rate]))

            if out == 1:  # 需dropout的层
                drop_layers.append(lay)

        keep_rot = []

        for i in range(layers):
            # 每个列表分为层
            # 这与我们使用的 QNN 相关
            keep_rot_layer = [list(range(n_qubits)) for j in range(1, inner_layers + 1)]

            if i in drop_layers:  # 如果需要在这一层应用 dropout
                keep_rot_layer = [] 
                inner_keep_r = [] 
                for param in range(params_per_layer):
                    # 每个旋转在层内有概率 p=rot_drop_rate 被丢弃
                    # 根据这个概率，我们为每个参数（旋转）采样
                    # 是否需要丢弃它
                    out = np.random.choice(np.array(range(2)), p=np.array([1 - rot_drop_rate, rot_drop_rate]))

                    if out == 0:  # 如果需要保留
                        inner_keep_r.append(param % n_qubits)  # % 是必须的，因为我们逐层工作
                    else:  # 如果旋转需要丢弃
                        inner_keep_r.append(-1)

                    if param % n_qubits == n_qubits - 1:  # 如果是寄存器的最后一个量子比特
                        keep_rot_layer.append(inner_keep_r)
                        inner_keep_r = []

            keep_rot.append(keep_rot_layer)

        return np.array(keep_rot)

    seed = 42
    layer_drop_rate = 0.5
    rot_drop_rate = 0.5
    layers = 5
    n_qubits = 4
    inner_layers = 3
    params_per_layer = 12

    result = make_dropout(seed, layer_drop_rate, rot_drop_rate, layers)

将量子线路添加至量子神经网络模块

.. code-block::

    class QNN(pyvqnet.nn.Module):
        
        def __init__(self, layers):
            super(QNN, self).__init__()
            self.qm = QMachine(n_qubits, dtype=pyvqnet.kcomplex64)
            self.para = Parameter((params_per_layer * layers,))
            
        def forward(self, x:QTensor, keep_rot):
            self.qm.reset_states(x.shape[0])
            x = qnn_circuit(x, self.para, keep_rot, n_qubits, layers, self.qm)
            
            return x

制作sin数据集

.. code-block::

    def make_sin_dataset(dataset_size=100, test_size=0.4, noise_value=0.4, plot=False):
        """1D regression problem y=sin(x*\pi)"""
        # x-axis
        x_ax = np.linspace(-1, 1, dataset_size)
        y = [[np.sin(x * np.pi)] for x in x_ax]
        np.random.seed(123)
        # noise vector
        noise = np.array([np.random.normal(0, 0.5, 1) for i in y]) * noise_value
        X = np.array(x_ax)
        y = np.array(y + noise)  # apply noise

        # split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=40, shuffle=True
        )

        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return X_train, X_test, y_train, y_test

    X, X_test, y, y_test = make_sin_dataset(dataset_size=20, test_size=0.25)


    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(y)
    y_test = scaler.transform(y_test)

    # reshaping for computation
    y = y.reshape(-1,)
    y_test = y_test.reshape(-1,)


    fig, ax = plt.subplots()
    plt.plot(X, y, "o", label="Training")
    plt.plot(X_test, y_test, "o", label="Test")

    plt.plot(
        np.linspace(-1, 1, 100),
        [np.sin(x * np.pi) for x in np.linspace(-1, 1, 100)],
        linestyle="dotted",
        label=r"$\sin(x)$",
    )
    plt.ylabel(r"$y = \sin(\pi\cdot x) + \epsilon$")
    plt.xlabel(r"$x$")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    plt.legend()

    plt.show()


.. image:: ./images/dropout_sin.png
   :width: 600 px
   :align: center

|

模型训练代码

.. code-block::

    epochs = 700

    n_run = 3
    seed =1234
    drop_rates = [(0.0, 0.0), (0.3, 0.2), (0.7, 0.7)]

    train_history = {}
    test_history = {}
    opt_params = {}
    layers = 3

    for layer_drop_rate, rot_drop_rate in drop_rates:
        costs_per_comb = []
        test_costs_per_comb = []
        opt_params_per_comb = []
        # 多次执行
        for tmp_seed in range(seed, seed + n_run):
            
            rng = np.random.default_rng(tmp_seed)
            assert len(X.shape) == 2  # X must be a matrix
            assert len(y.shape) == 1  # y must be an array
            assert X.shape[0] == y.shape[0]  # compatibility check

            # lists for saving single run training and test cost trend
            costs = []
            test_costs = []
            model = QNN(layers)
            optimizer = pyvqnet.optim.Adam(model.parameters(), lr=0.001)
            loss = pyvqnet.nn.loss.MeanSquaredError()
            
            for epoch in range(epochs):
                # 生成dropout列表
                keep_rot = make_dropout(rng, layer_drop_rate, rot_drop_rate, layers)
                # 更新rng
                rng = np.random.default_rng(tmp_seed)
                
                optimizer.zero_grad()
                data, label = QTensor(X,requires_grad=True,dtype=6), QTensor(y,
                                                    dtype=6,
                                                    requires_grad=False)

                result = model(data, keep_rot)
                cost = loss(label, result)
                costs.append(cost)
                cost.backward()
                optimizer._step()

                ## 测试
                keep_rot = np.array(
                    [
                        [list(range((n_qubits))) for j in range(1, inner_layers + 1)]
                        for i in range(layers)
                    ]
                )
                
                
                data_test, label_test = QTensor(X_test,requires_grad=True,dtype=6), QTensor(y_test,
                                                    dtype=6,
                                                    requires_grad=False)
                result_test = model(data_test, keep_rot)
                test_cost = loss(label_test, result_test)
                test_costs.append(test_cost)
                
                if epoch % 5 == 0:
                    print(
                        f"{layer_drop_rate:.1f}-{rot_drop_rate:.1f} ",
                        f"run {tmp_seed-seed} - epoch {epoch}/{epochs}",
                        f"--- Train cost:{cost}",
                        f"--- Test cost:{test_cost}",
                        end="\r",
                    )

            costs_per_comb.append(costs)
            test_costs_per_comb.append(test_costs)
            opt_params_per_comb.append(model.parameters())

        train_history[(layer_drop_rate, rot_drop_rate)] = costs_per_comb
        test_history[(layer_drop_rate, rot_drop_rate)] = test_costs_per_comb
        opt_params[(layer_drop_rate, rot_drop_rate)] = opt_params_per_comb

    ## 0.0-0.0  run 0 - epoch 695/700 --- Train cost:0.3917597 --- Test cost:0.2427316
    ## 0.0-0.0  run 1 - epoch 695/700 --- Train cost:0.3917596 --- Test cost:0.2349882
    ## 0.0-0.0  run 2 - epoch 695/700 --- Train cost:0.3917597 --- Test cost:0.2103992
    ## 0.3-0.2  run 0 - epoch 695/700 --- Train cost:0.3920721 --- Test cost:0.2155183
    ## 0.3-0.2  run 1 - epoch 695/700 --- Train cost:0.3932508 --- Test cost:0.2353068
    ## 0.3-0.2  run 2 - epoch 695/700 --- Train cost:0.392473 --- Test cost:0.20580922
    ## 0.7-0.7  run 0 - epoch 695/700 --- Train cost:0.3922218 --- Test cost:0.2057379

通过在训练时对模型的参数们进行随机dropout方式，能够预防模型的过拟合问题，不过需要对dropout的概率进行合适的设计，不然也会导致模型训练结果不佳。


在VQNet使用量子计算层进行模型训练
***************************************

以下是使用 ``QuantumLayer``, ``NoiseQuantumLayer``, ``VQCLayer`` 等VQNet接口实现量子机器学习的例子。

在VQNet中使用QuantumLayer进行模型训练
=========================================

.. code-block::

    from pyvqnet.nn.module import Module
    from pyvqnet.optim import sgd
    import numpy as np
    import os
    from pyvqnet.data import data_generator as dataloader
    from pyvqnet.nn.loss import CategoricalCrossEntropy

    from pyvqnet.tensor.tensor import QTensor
    import random
    import pyqpanda as pq
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyqpanda import *
    random.seed(1234)

    qvc_train_data = [0,1,0,0,1,
    0, 1, 0, 1, 0,
    0, 1, 1, 0, 0,
    0, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 1, 0, 0,
    1, 0, 1, 1, 1,
    1, 1, 0, 0, 0,
    1, 1, 0, 1, 1,
    1, 1, 1, 0, 1,
    1, 1, 1, 1, 0]
    qvc_test_data= [0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    0, 0, 1, 0, 1,
    0, 0, 1, 1, 0]

    def qvc_circuits(input,weights,qlist,clist,machine):

        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits)-1):
                cir.insert(pq.CNOT(nqubits[i],nqubits[i+1]))
            cir.insert(pq.CNOT(nqubits[len(nqubits)-1],nqubits[0]))
            return cir

        def build_circult(weights, xx, nqubits):
            
            def Rot(weights_j, qubits):
                circult = pq.QCircuit()
                circult.insert(pq.RZ(qubits, weights_j[0]))
                circult.insert(pq.RY(qubits, weights_j[1]))
                circult.insert(pq.RZ(qubits, weights_j[2]))
                return circult
            def basisstate():
                circult = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i]==1:
                        circult.insert(pq.X(nqubits[i]))
                return circult

            circult = pq.QCircuit()
            circult.insert(basisstate())

            for i in range(weights.shape[0]):
                
                weights_i = weights[i,:,:]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circult.insert(Rot(weights_j,nqubits[j]))
                cnots = get_cnot(nqubits)  
                circult.insert(cnots) 

            circult.insert(pq.Z(nqubits[0]))
            
            prog = pq.QProg() 
            
            prog.insert(circult)
            return prog

        weights = weights.reshape([2,4,3])
        prog = build_circult(weights,input,qlist)  
        prob = machine.prob_run_dict(prog, qlist[0], -1)
        prob = list(prob.values())

        return prob

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits,24,"cpu",4)

        def forward(self, x):
            return self.qvc(x)


    def get_data(dataset_str):
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)

        else:
            datasets = np.array(qvc_test_data)

        datasets = datasets.reshape([-1,5])
        data = datasets[:,:-1]
        label = datasets[:,-1].astype(int)
        label = np.eye(2)[label].reshape(-1,2)
        return data, label

    def get_accuary(result,label):
        result,label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
        return score

    def Run():

        model = Model()

        optimizer = sgd.SGD(model.parameters(),lr =0.5)
        batch_size = 3
        epoch = 10
        loss = CategoricalCrossEntropy()
        print("start training..............")
        model.train()

        datas,labels = get_data("train")
        print(datas)
        print(labels)
        print(datas.shape)
        for i in range(epoch):
            count=0
            sum_loss = 0
            accuary = 0
            t = 0
            for data,label in dataloader(datas,labels,batch_size,False):
                optimizer.zero_grad()
                data,label = QTensor(data), QTensor(label)

                result = model(data)

                loss_b = loss(label,result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count+=batch_size
                accuary += get_accuary(result,label)
                t = t + 1

            print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}")
        print("start testing..............")
        model.eval()
        count = 0
        test_data, test_label = get_data("test")
        test_batch_size = 1
        accuary = 0
        sum_loss = 0
        for testd,testl in dataloader(test_data,test_label,test_batch_size):
            testd = QTensor(testd)
            test_result = model(testd)
            test_loss = loss(testl,test_result)
            sum_loss += test_loss
            count+=test_batch_size
            accuary += get_accuary(test_result,testl)
        print(f"test:--------------->loss:{sum_loss/count} #####accuray:{accuary/count}")

    if __name__=="__main__":

        Run()


运行的loss以及accuracy结果：

.. code-block::
	
	start training..............
	epoch:0, #### loss:[0.20585182] #####accuray:0.6
	epoch:1, #### loss:[0.17479989] #####accuray:1.0
	epoch:2, #### loss:[0.12679021] #####accuray:1.0
	epoch:3, #### loss:[0.11088503] #####accuray:1.0
	epoch:4, #### loss:[0.10598478] #####accuray:1.0
	epoch:5, #### loss:[0.10482856] #####accuray:1.0
	epoch:6, #### loss:[0.10453037] #####accuray:1.0
	epoch:7, #### loss:[0.10445572] #####accuray:1.0
	epoch:8, #### loss:[0.10442699] #####accuray:1.0
	epoch:9, #### loss:[0.10442187] #####accuray:1.0
	epoch:10, #### loss:[0.10442089] #####accuray:1.0
	epoch:11, #### loss:[0.10442062] #####accuray:1.0
	epoch:12, #### loss:[0.10442055] #####accuray:1.0
	epoch:13, #### loss:[0.10442055] #####accuray:1.0
	epoch:14, #### loss:[0.10442055] #####accuray:1.0
	epoch:15, #### loss:[0.10442055] #####accuray:1.0
	epoch:16, #### loss:[0.10442055] #####accuray:1.0

	start testing..............
	[0.3132616580]
	test:--------------->loss:QTensor(0.3132616580, requires_grad=True) #####accuray:1.0

在VQNet中使用NoiseQuantumLayer进行模型训练
=============================================

使用 ``NoiseQuantumLayer`` 可以使用QPanda的噪声虚拟机构建含噪量子线路，并进行训练。

一个完整的含噪模型量子机器学习模型的例子如下：

.. code-block::

    import os
    import numpy as np

    from pyvqnet.nn.module import Module
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.conv import Conv2D

    from pyvqnet.nn import activation as F
    from pyvqnet.nn.pooling import MaxPool2D
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor

    import pyqpanda as pq
    from pyqpanda import *
    from pyvqnet.qnn.quantumlayer import NoiseQuantumLayer
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import time
    try:
        matplotlib.use('TkAgg')
    except:
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError('You should use Python 3.x')
    import os.path
    import gzip

    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }



    def _download(dataset_dir,file_name):
        file_path = dataset_dir + "/" + file_name
        
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as f:
                file_path_ungz = file_path[:-3].replace('\\', '/')
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz,"wb").write(f.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as f:
                    file_path_ungz = file_path[:-3].replace('\\', '/')
                    file_path_ungz = file_path_ungz.replace('-idx', '.idx')
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz,"wb").write(f.read())
        print("Done")
        
    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir,v)

    #use qpanda to create quantum circuits
    def circuit(weights,param,qubits,cbits,machine):

        circuit = pq.QCircuit()
        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.RY(qubits[0], weights[0]))
        prog = pq.QProg()
        prog.insert(circuit)
        prog << measure_all(qubits, cbits)
        result = machine.run_with_configuration(prog, cbits, 100)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        # Compute probabilities for each state
        probabilities = counts / 100
        # Get state expectation
        expectation = np.sum(states * probabilities)
        return expectation



    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.conv2 = Conv2D(input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.fc1 = Linear(input_channels=256, output_channels=64)
            self.fc2 = Linear(input_channels=64, output_channels=1)
            
            self.hybrid = NoiseQuantumLayer(circuit,1,"noise",1)
            self.fc3 = Linear(input_channels=1, output_channels=2)


        def forward(self, x):
            x = F.ReLu()(self.conv1(x))  
            x = self.maxpool1(x)
            x = F.ReLu()(self.conv2(x))  
            x = self.maxpool2(x)
            x = tensor.flatten(x, 1)  
            x = F.ReLu()(self.fc1(x))  
            x = self.fc2(x)    
            x = self.hybrid(x)
            x = self.fc3(x)

            return x

该模型为混合量子线路以及经典网络的模型，其中量子线路部分使用 ``NoiseQuantumLayer`` 对量子线路加噪声模型进行模拟。使用该模型对mnist数据库中中的0，1手写数字进行分类。

.. code-block::

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         # 下载数据
        import os, struct
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
        elif dataset == "testing_data":
            fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, 'rb')
        _, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, 'rb')
        _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        N = len(ind)
        images = np.zeros((N, rows, cols))
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def data_select(train_num, test_num):
        x_train, y_train = load_mnist("training_data")  
        x_test, y_test = load_mnist("testing_data")
        idx_train = np.append(np.where(y_train == 0)[0][:train_num],
                        np.where(y_train == 1)[0][:train_num])

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]
        
        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)

        # Test Leaving only labels 0 and 1
        idx_test = np.append(np.where(y_test == 0)[0][:test_num],
                        np.where(y_test == 1)[0][:test_num])

        x_test = x_test[idx_test]
        y_test = y_test[idx_test]
        x_test = x_test / 255
        y_test = np.eye(2)[y_test].reshape(-1, 2)
        
        return x_train, y_train, x_test, y_test

    if __name__=="__main__":
        x_train, y_train, x_test, y_test = data_select(100, 50)
        
        model = Net()
        optimizer = Adam(model.parameters(), lr=0.005)
        loss_func = CategoricalCrossEntropy()

        epochs = 10
        loss_list = []
        eval_loss_list = []
        train_acc_list = []
        eval_acc_list = []
        model.train()
        if not os.path.exists("./result"):
            os.makedirs("./result")
        else:
            pass
        eval_time = []
        F1 = open("./result/hqcnn_train_rlt.txt","w")
        F2 = open("./result/hqcnn_eval_rlt.txt","w")
        for epoch in range(1, epochs):
            total_loss = []
            iter  = 0
            correct = 0
            n_train = 0
            for x, y in data_generator(x_train, y_train, batch_size=1, shuffle=True):
                iter +=1
                start_time = time.time()
                x = x.reshape(-1, 1, 28, 28)
                optimizer.zero_grad()
                # Forward pass
                output = model(x)
                # Calculating loss
                loss = loss_func(y, output) 
                loss_np = np.array(loss.data)
                np_output = np.array(output.data, copy=False)
                mask = (np_output.argmax(1) == y.argmax(1))
                correct += np.sum(np.array(mask))
                n_train += 1
                
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer._step()
                total_loss.append(loss_np)
            print("##########################")
            print(f"Train Accuracy: {correct / n_train}")
            loss_list.append(np.sum(total_loss) / len(total_loss))
            train_acc_list.append(correct/n_train)
            print("epoch: ", epoch)
            print(100. * (epoch + 1) / epochs)
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
            F1.writelines(f"{epoch},{loss_list[-1]},{correct/n_train}\n")

            model.eval()
            correct = 0
            total_eval_loss = []
            n_eval = 0
            
            for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
                start_time1 = time.time()
                x = x.reshape(-1, 1, 28, 28)
                output = model(x)
                loss = loss_func(y, output)

                np_output = np.array(output.data, copy=False)
                mask = (np_output.argmax(1) == y.argmax(1))
                correct += np.sum(np.array(mask))
                n_eval += 1
                
                loss_np = np.array(loss.data)
                total_eval_loss.append(loss_np)
                eval_acc_list.append(correct/n_eval)
            print(f"Eval Accuracy: {correct / n_eval}")
            F2.writelines(f"{epoch},{np.sum(total_eval_loss) / len(total_eval_loss)},{correct/n_eval}\n")
        F1.close()
        F2.close()
		
对比含噪量子线路与理想量子线路的机器学习模型分类结果，其loss变化情况以及accuary变化情况如下：

.. code-block::

    Train Accuracy: 0.715
    epoch:  1
    1 loss is : 0.6519572449
    Eval Accuracy: 0.99
    ##########################
    Train Accuracy: 1.0
    epoch:  2
    2 loss is : 0.4458528900
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  3
    3 loss is : 0.3142367172
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  4
    4 loss is : 0.2259583092
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  5
    5 loss is : 0.1661866951
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  6
    6 loss is : 0.1306252861
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0
    epoch:  7
    7 loss is : 0.0996847820
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0
    epoch:  8
    8 loss is : 0.0801456261
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0
    epoch:  9
    9 loss is : 0.0649107647
    Eval Accuracy: 1.0

|

在VQNet中使用VQCLayer进行模型训练
=============================================

在本源量子的qpanda提供了 `VariationalQuantumCircuit <https://qpanda-tutorial.readthedocs.io/zh/latest/VQC.html#id1>`_ 。
线路中仅变化参数不变结构的门可以用pyQPanda的 ``VariationalQuantumGate`` 组成。
VQNet提供了封装类 ``VQC_wrapper`` ，用户使用普通逻辑门在函数 ``build_common_circuits`` 中构建模型中线路结构发生不定的子线路，
使用VQG在 ``build_vqc_circuits`` 构建结构不变，参数变化的子线路。使用 ``run`` 函数定义线路运行方式以及测量。

.. code-block::

    """
    using pyqpanda VQC api to build model and train VQNet model demo.

    """
    import sys,os
    import time

    from pyvqnet.data import data_generator as dataloader
    from pyvqnet.nn.module import Module
    from pyvqnet.optim import sgd
    import numpy as np
    from pyvqnet.nn.loss import CategoricalCrossEntropy

    from pyvqnet.tensor import QTensor
    import random

    from pyvqnet.qnn.quantumlayer import VQCLayer,VQC_wrapper,_array2var
    from pyqpanda import *
    import pyqpanda as pq

    random.seed(1234)
    qvc_train_data = [
        0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 0
    ]
    qvc_test_data = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]

    class QVC_demo(VQC_wrapper):

        def __init__(self):
            super(QVC_demo, self).__init__()

        def build_common_circuits(self,input,qlists,):
            qc = pq.QCircuit()
            for i in range(len(qlists)):
                if input[i]==1:
                    qc.insert(pq.X(qlists[i]))
            return qc

        def build_vqc_circuits(self,input,weights,machine,qlists,clists):

            def get_cnot(qubits):
                vqc = VariationalQuantumCircuit()
                for i in range(len(qubits)-1):
                    vqc.insert(pq.VariationalQuantumGate_CNOT(qubits[i],qubits[i+1]))
                vqc.insert(pq.VariationalQuantumGate_CNOT(qubits[len(qubits)-1],qubits[0]))
                return vqc

            def build_circult(weights, xx, qubits,vqc):

                def Rot(weights_j, qubits):
                    vqc = VariationalQuantumCircuit()
                    vqc.insert(pq.VariationalQuantumGate_RZ(qubits, weights_j[0]))
                    vqc.insert(pq.VariationalQuantumGate_RY(qubits, weights_j[1]))
                    vqc.insert(pq.VariationalQuantumGate_RZ(qubits, weights_j[2]))
                    return vqc

                for i in range(2):
                    weights_i = weights[i,:,:]
                    for j in range(len(qubits)):
                        weights_j = weights_i[j]
                        vqc.insert(Rot(weights_j,qubits[j]))
                    cnots = get_cnot(qubits)
                    vqc.insert(cnots)
                vqc.insert(pq.VariationalQuantumGate_Z(qubits[0]))  # pauli z(0)
                return vqc

            weights = weights.reshape([2,4,3])
            vqc = VariationalQuantumCircuit()
            return build_circult(weights, input,qlists,vqc)

        def run(self,vqc,input,machine,qlists,clists):
            """
            a function to get hamilton observable or measurment
            """
            prog = QProg()
            vqc_all = VariationalQuantumCircuit()
            # add encode circuits
            vqc_all.insert(self.build_common_circuits(input,qlists))
            vqc_all.insert(vqc)
            qcir = vqc_all.feed()
            prog.insert(qcir)
            prob = machine.prob_run_dict(prog, qlists[0], -1)
            prob = list(prob.values())

            return prob

    class Model(Module):
        def __init__(self,qvc_vqc):
            super(Model, self).__init__()
            self.qvc = VQCLayer(qvc_vqc,24,"cpu",4)

        def forward(self, x):
            return self.qvc(x)


    def get_data(dataset_str):
        """
        Tranform data to valid form
        """
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)

        else:
            datasets = np.array(qvc_test_data)

        datasets = datasets.reshape([-1, 5])
        data = datasets[:, :-1]
        label = datasets[:, -1].astype(int)
        label = np.eye(2)[label].reshape(-1, 2)
        return data, label



    def get_accuary(result,label):
        result,label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
        return score

    def Run():
        ### create class for VQC
        qvc_vqc = QVC_demo()
        model = Model(qvc_vqc)

        optimizer = sgd.SGD(model.parameters(),lr =0.5)
        batch_size = 3
        epoch = 20
        loss = CategoricalCrossEntropy()
        print("start training..............")
        model.train()
        PATH = os.path.abspath('train')
        datas,labels = get_data(PATH)
        for i in range(epoch):
            count=0
            sum_loss = 0
            accuary = 0

            for data,label in dataloader(datas,labels,batch_size,False):
                optimizer.zero_grad()
                data,label = QTensor(data), QTensor(label)
                result = model(data)
                loss_b = loss(label,result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count+=batch_size
                accuary += get_accuary(result,label)

            print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}")
        print("start testing..............")
        model.eval()
        count = 0
        test_data, test_label = get_data("test")
        test_batch_size = 1
        accuary = 0
        sum_loss = 0
        for testd,testl in dataloader(test_data,test_label,test_batch_size):
            testd = QTensor(testd)
            test_result = model(testd)
            test_loss = loss(testl,test_result)
            sum_loss += test_loss
            count+=test_batch_size
            accuary += get_accuary(test_result,testl)
        print(f"test:--------------->loss:{sum_loss/count} #####accuray:{accuary/count}")

    if __name__=="__main__":

        Run()

.. code-block::

    start training..............
    epoch:0, #### loss:0.22664549748102825 #####accuray:0.5333333333333333
    epoch:1, #### loss:0.20315084457397461 #####accuray:0.6666666666666666
    epoch:2, #### loss:0.1644243836402893 #####accuray:1.0
    epoch:3, #### loss:0.12654326359430948 #####accuray:1.0
    epoch:4, #### loss:0.11026077469189961 #####accuray:1.0
    epoch:5, #### loss:0.10584278305371603 #####accuray:1.0
    epoch:6, #### loss:0.10476383566856384 #####accuray:1.0
    epoch:7, #### loss:0.10450373888015747 #####accuray:1.0
    epoch:8, #### loss:0.10444082617759705 #####accuray:1.0
    epoch:9, #### loss:0.10442551374435424 #####accuray:1.0
    epoch:10, #### loss:0.10442176461219788 #####accuray:1.0
    epoch:11, #### loss:0.10442084868748983 #####accuray:1.0
    epoch:12, #### loss:0.10442061225573222 #####accuray:1.0
    epoch:13, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:14, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:15, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:16, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:17, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:18, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:19, #### loss:0.10442055265108745 #####accuray:1.0
    start testing..............
    [0.3132616580]
    test:--------------->loss:QTensor(0.3132616580, requires_grad=True) #####accuray:1.0

