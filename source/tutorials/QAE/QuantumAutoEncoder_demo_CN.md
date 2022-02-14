# 用于量子数据高效压缩的量子自动编码器演示

## 概述

经典的自动编码器是一种神经网络，可以在高维空间学习数据的高效低维表示。自动编码器的任务是，给定一个输入x，将x映射到一个低维点y，这样x就可以从y中恢复。可以选择底层自动编码器网络的结构，以便在较小的维度上表示数据，从而有效地压缩输入。受这一想法的启发，我们引入了量子自动编码器的模型来对量子数据执行类似的任务。量子自动编码器被训练来压缩量子态的特定数据集，而经典的压缩算法无法使用。量子自动编码器的参数采用经典优化算法进行训练。我们展示了一个简单的可编程线路的例子，它可以被训练成一个高效的自动编码器。我们在量子模拟的背景下应用我们的模型来压缩哈伯德模型和分子哈密顿量的基态。

代码可见[qae_test.py](../../pyVQNet/examples/qae_test.py)

![image-20210923140654983](.\QAE.png)

a） 具有3位潜在空间的6位自动编码器的图形表示。$\epsilon$将6位输入（红点）编码为3位中间状态（黄点），然后解码器$D$尝试在输出（绿点）处重建输入位。

b） 6-3-6量子自动编码器的线路实现。

## 量子自编码线路

![image-20210923140637113](.\QAE_Cir.png)

● 输入数据: $\left\{\left|\Psi_{\mathrm{i}}\right\rangle_{\mathrm{AB}}\right\}$是$n+k$量子位上的状态集合， 其中子系统$A$和$B$由$n$和$k$量子位组成。
		● 评估性能：测量与初始输入状态$\left|\Psi_{\mathrm{i}}\right\rangle$到输出$\rho_{\text {out }}^{i}$的偏差，我们计算期望的保真度公式为 $\mathrm{F}\left(\left|\Psi_{\mathrm{i}}\right\rangle, \rho_{\text {out }}^{i}\right)=\left\langle\Psi_{\mathrm{j}}\left|\rho_{\text {out }}^{\text {i }}\right| \Psi_{\mathrm{i}}\right\rangle$.
		● 对于成功的自动编码， $\mathrm{F}\left(\left|\Psi_{\mathrm{i}}\right\rangle, \rho_{\text {out }}^{i}\right)\approx1$ .
		● $U_{\text { }}^{p}$ 作用于$n+k$量子位的酉算子族, 其中$p=\left\{p_{1}, p_{2}, \ldots\right\}$ 是定义单一量子线路的一组参数。

● 其中$|\mathrm{a}\rangle_{\mathrm{B'}}$是$k$量子位的固定纯参考态。
		● 我们希望找到使平均保真度最大化的 $U_{\text { }}^{p}$ : 

$C_{1}(\vec{p})=\sum_{i} p_{i} \cdot F\left(\left|\psi_{i}\right\rangle, \rho_{i, \vec{p}}^{\text {out }}\right)$

也被称为损失函数，其中
$$
\rho_{i, \vec{p}}^{\text {out }}=\left(U^{\vec{p}}\right)_{A B^{\prime}}^{\dagger} \operatorname{Tr}_{B}\left[U_{A B}^{\vec{p}}\left[\psi_{i_{A B}} \otimes a_{B^{\prime}}\right]\left(U_{A B}^{\vec{p}}\right)^{\dagger}\right]\left(U^{\vec{p}}\right)_{A B^{\prime}}
$$
​		●最后一步，SWAP 操作。

## 可编程编码器线路

![image-20210923141513505](.\Quantum_circuit.png)

上图是一个可编程线路，可用作自动编码器模型。它包括一个量子位集合中所有可能的受控一般单量子位旋转（由$R_i$表示），加上单元开始和结束处的单量子位旋转。所有线路都用四量子位输入来描述。单元格由红色虚线分隔。

![image-20210923142001073](.\measure_circuit.png)

通过可编程线路的结果通过SWAP操作进行测量。根据该结果判断结果的逼真度。

![image-20210923141717162](.\quantum_classical.png)

上图是用于训练量子自编码器的混合方案的示意图。

## 流程

### 数据预处理

- 使用MNIST数据集（图片为手写数字），其中包含60.000张28 x 28像素的图片。
- 将所选图片的大小调整为较小的尺寸（2x{2,3,4}像素），然后展平这些值以获得值[b1，…，bn]，其中n是4,6或8。
- 为了将它们转换为量子比特，它们使用旋转门（此处选择$Rx$门）进行编码。
- 最终量子位寄存器为：$$
  \left|00 . .0 q_{1} \ldots q_{n}\right\rangle
  $$.

### 量子线路定义

![QAE_Quantum_Cir](.\QAE_Quantum_Cir.png)

### 构建量子线路

```python
# parameter quantum circuits
import sys
import os
from vqnet.nn.module import Module, Parameter
import pyqpanda.pyQPanda as pq
from pyqpanda import *
import numpy as np
from vqnet.tensor.tensor import QTensor
from pyqpanda.Visualization.circuit_draw import *
from pyvqnet._core import Tensor as CoreTensor
import re


class QAElayer(Module):
    """
    parameterized quantum circuit Layer.It contains paramters can be trained.

    """
    def __init__(self,trash_qubits_number:int=2,total_qubits_number:int=7,machine:str="cpu"):

        """

        trash_qubits_number: 'int' - should tensor's gradient be tracked, defaults to False
        total_qubits_number: 'int' - Ansatz circuits repeat block times
        machine: 'str' - compute machine
        """
        super().__init__()

        self.machine = machine
        if machine!="cpu":
            raise ValueError("machine only tested on cpu simulation")
        self.machine = pq.CPUQVM()# outside
        self.machine.init_qvm()# outside
        self.qlist = self.machine.qAlloc_many(total_qubits_number)
        aux_qubits_number = 1
        self.clist = self.machine.cAlloc_many(aux_qubits_number)

        self.history_prob = []

        self.n_qubits = total_qubits_number
        self.n_aux_qubits = aux_qubits_number
        self.n_trash_qubits = trash_qubits_number

        training_qubits_size = self.n_qubits - self.n_aux_qubits - self.n_trash_qubits

        # 12  + 36  + 12 params ,if training_qubits_size =4 (default)
        weight_shapes = {"params_rot_begin": training_qubits_size * 3,
                     "params_crot": training_qubits_size * (training_qubits_size - 1) * 3,
                     "params_rot_end":  training_qubits_size * 3}


        self.weights = Parameter(weight_shapes['params_rot_begin']
                                + weight_shapes['params_crot']
                                + weight_shapes['params_rot_end'])



    def forward(self, x):
        """
            forward function
        """
        self.history_prob = []
        batchsize = x.shape[0]
        batch_measure = np.zeros([batchsize])

        if batchsize == 1:
            prog = paramterized_quautum_circuits(x.data,
                                                self.weights.data,
                                                self.qlist,
                                                self.clist,
                                                self.n_qubits,
                                                self.n_aux_qubits,
                                                self.n_trash_qubits)

            result = self.machine.run_with_configuration(prog, self.clist, 100)
            counts = result['0']
            probabilities = counts / 100
            print("probabilities", probabilities)
            batch_measure[0] = probabilities
        else:
            for b in range(batchsize):
                xdata = x.data
                b_str = str(b)
                xdata_ = xdata.select([b_str, ":"])
                prog = paramterized_quautum_circuits(xdata_,
                                                    self.weights.data,
                                                    self.qlist,
                                                    self.clist,
                                                    self.n_qubits,
                                                    self.n_aux_qubits,
                                                    self.n_trash_qubits)

                result = self.machine.run_with_configuration(prog, self.clist, 100)
                counts = result['0']
                # print(f'result {result}')
                probabilities = counts / 100
                batch_measure[b] = probabilities


        requires_grad = (x.requires_grad or self.weights.requires_grad) and not QTensor.NO_GRAD
        nodes = []
        if x.requires_grad:
            nodes.append(QTensor.GraphNode(tensor=x, df=lambda g: 1))
        if self.weights.requires_grad:
            nodes.append(
                QTensor.GraphNode(
            tensor = self.weights, df=lambda g: _grad(g,
                                                     paramterized_quautum_circuits,
                                                     x.data,
                                                     self.weights.data,
                                                     self.machine,
                                                     self.qlist,self.clist,
                                                     self.n_qubits,
                                                     self.n_aux_qubits,
                                                     self.n_trash_qubits
                                                     )
                                            )
                    )
        return QTensor(data = [batch_measure],requires_grad = requires_grad,nodes =nodes)




# use finit difference to calc delta
def _grad(g, ForwardFunc, x:CoreTensor, weights:CoreTensor,
                    machine, qlists, clists,
                                        n_qubits,
                                        n_aux_qubits,
                                        n_trash_qubits,
                    delta:float = np.pi/2):

    num_para = 1
    g = np.array(g)
    batchsize = x.shape[0]
    for _ in weights.shape:
        num_para = num_para * _
    grad = np.zeros_like(weights)

    #loop every params
    for _ in range(num_para):
        #loop every sample (average)
        for b in range(batchsize):
            b_str = str(b)
            x_ = x.select([b_str, ":"])
            weights_theta = np.array(weights)
            weights_theta[_] = weights_theta[_] + delta
            weights_theta = CoreTensor(weights_theta)

            prog_add_theta = ForwardFunc(x_,
                                        weights_theta,
                                        qlists,
                                        clists,
                                        n_qubits,
                                        n_aux_qubits,
                                        n_trash_qubits
                                        )

            result = machine.run_with_configuration(prog_add_theta, clists, 100)
            counts = result['0']
            # print(result)
            prob_add_theta = counts / 100
            weights_theta = np.array(weights)
            weights_theta[_] = weights_theta[_] - delta
            weights_theta = CoreTensor(weights_theta)

            prob_sub_theta = ForwardFunc(x_,
                                        weights_theta,
                                        qlists,
                                        clists,
                                        n_qubits,
                                        n_aux_qubits,
                                        n_trash_qubits
                                        )

            result = machine.run_with_configuration(prob_sub_theta, clists, 100)
            counts = result['0']
            # print(result)
            prob_sub_theta = counts / 100
            grad[_] += g[b]*(prob_add_theta - prob_sub_theta) / (2*delta)
        grad[_] /= (batchsize )
    grad = CoreTensor(grad)
    return grad


def SWAP_CIRCUITS(input,
                  param,
                  qubits,
                  n_qubits:int  =7,
                  n_aux_qubits = 1,
                  n_trash_qubits:int = 2,):
    circuit = pq.QCircuit()
    for i in range(n_aux_qubits):
        circuit.insert(pq.H(qubits[i]))
    for i in range(n_trash_qubits):
        circuit_tmp = pq.QCircuit()
        # print([i, i + n_aux_qubits , 2 * n_trash_qubits - i])
        circuit_tmp.insert(pq.SWAP(qubits[i + n_aux_qubits], qubits[2 * n_trash_qubits - i]))
        circuit_tmp.set_control(qubits[i])
        circuit.insert(circuit_tmp)
    for i in range(n_aux_qubits):
        circuit.insert(pq.H(qubits[i]))

    return circuit

def paramterized_quautum_circuits(input:CoreTensor,
                                  param:CoreTensor,
                                  qubits,
                                  clist,
                                  n_qubits:int  =7,
                                  n_aux_qubits = 1,
                                  n_trash_qubits:int = 2):
    """
    use qpanda to define circuit

    """
    input = np.array(input)
    input = input.squeeze()
    param = np.array(param)
    param = param.squeeze()

    training_qubits_size = n_qubits - n_trash_qubits - n_aux_qubits
    params_rot_begin = 0  # ( training_qubits_size )*3
    params_crot = (training_qubits_size )*3  # training_qubits_size * (training_qubits_size - 1) * 3,
    params_rot_end = training_qubits_size * 3 + training_qubits_size * (training_qubits_size - 1) * 3

    circuit = pq.QCircuit()
    # 3~7 angle_embedding
    for i in range(n_trash_qubits + n_aux_qubits, n_qubits):
        circuit.insert(pq.RX(qubits[i], input[i]))

    # Add the first rotational gates:
    #qubits 3 ~7
    for i in range(n_aux_qubits + n_trash_qubits, n_qubits):
        # qml.Rot(phi, theta, omega, wire) :rz ry rz inverse
        circuit.insert(pq.RZ(qubits[i], param[params_rot_begin+2]))  # 0 3 6 9
        circuit.insert(pq.RY(qubits[i], param[params_rot_begin+1]))  # 1 4 7 10
        circuit.insert(pq.RZ(qubits[i], param[params_rot_begin]))  # 2 5 8 11
        params_rot_begin += 3

    for i in range(n_aux_qubits + n_trash_qubits, n_qubits):  # 3 - 7
        for j in range(n_aux_qubits + n_trash_qubits, n_qubits):  # 3 - 7
            if i != j:
                circuit_sub = pq.QCircuit()
                circuit_sub.insert(pq.RZ(qubits[j], param[params_crot+2]))  # 0 3 6 9
                circuit_sub.insert(pq.RY(qubits[j], param[params_crot+1]))  # 1 4 7 10
                circuit_sub.insert(pq.RZ(qubits[j], param[params_crot]))  # 2 5 8 11
                circuit_sub.set_control(qubits[i])
                params_crot += 3
                circuit = circuit.insert(circuit_sub)

    for i in range(n_aux_qubits + n_trash_qubits, n_qubits):
        # qml.Rot(phi, theta, omega, wire) :rz ry rz inverse
        circuit.insert(pq.RZ(qubits[i], param[params_rot_end+2]))  # 0 3 6 9
        circuit.insert(pq.RY(qubits[i], param[params_rot_end+1]))  # 1 4 7 10
        circuit.insert(pq.RZ(qubits[i], param[params_rot_end]))  # 2 5 8 11
        params_rot_end += 3

    circuit.insert(SWAP_CIRCUITS(input, param, qubits, n_qubits, n_aux_qubits, n_trash_qubits))

    prog = pq.QProg()
    prog.insert(circuit)
    for m in range(n_aux_qubits):
        prog.insert(pq.Measure(qubits[m], clist[m]))

    # draw_qprog(prog, 'pic', filename='D:/test_cir_draw.png')
    return prog
```

### 训练和测试

```python
import sys
import os
sys.path.insert(0, os.path.abspath('..\\pyVQNet\\'))
# print(sys.path)
import numpy as np
from vqnet.nn.module import Module
from vqnet.nn.loss import CategoricalCrossEntropy, fidelityLoss
from vqnet.optim.adam import Adam
from vqnet.optim.sgd import SGD
from vqnet.data.data import data_generator

from vqnet.qnn.qae.qae import QAElayer
from vqnet.nn.loss import Loss

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def cal_size(tup):
    sz = 1
    for i in tup:
        sz*=i
    return sz

class Model(Module):

    def __init__(self, trash_num: int = 2, total_num: int = 7):
        super().__init__()
        self.pqc = QAElayer(trash_num, total_num)
        
    def forward(self, x):
        
        x = self.pqc(x)
        #x = self.fc(x)
        return x


def load_mnist(dataset="training_data", digits=np.arange(2), path="./dataset/MNIST_data"):      
    import os, struct
    from array import array as pyarray
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
    x_train, y_train = load_mnist("training_data")                    
    x_train = x_train / 255                                            
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

    model = Model(trash_qubits, total_qubits)

    optimizer = Adam(model.parameters(), lr=0.005)                        
    model.train()

    loss_list = []
    loss_list_test = []
    fidelity_train = []
    fidelity_val = []

    for epoch in range(1, 40):
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

            output_data = np.array(output.data)
            floss = fidelityLoss()
            loss = floss(output)
            loss_data = np.array(loss.data)
            loss.backward()
            running_fidelity_train += output_data[0]
            optimizer._step()
            full_loss += loss_data[0]
            n_loss += batch_size
            # print("Epoch:", epoch, "batch_processed:", n_loss, "loss:", loss.item())
            np_output = np.array(output.data, copy=False)
            # print("--------------------------------")
            mask = np_output.argmax(1) == y.argmax(1)

            correct += sum(mask)

        # print(f"Train Accuracy: {correct/n_loss}%")
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

            np_output = np.array(output.data, copy=False)
            mask = np_output.argmax(1) == y.argmax(1)
            correct += sum(mask)
            n_eval += 1
            n_loss += 1


        loss_output = full_loss / n_loss
        print(f"Epoch: {epoch}, Loss: {loss_output}")
        loss_list_test.append(loss_output)

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
```

结果展示：

![train_loss](.\train_loss.png)

训练和测试的保真度都是接近于1的。

