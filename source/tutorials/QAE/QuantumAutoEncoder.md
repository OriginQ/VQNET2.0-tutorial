# Demo of Quantum autoencoders for efficient compression of quantum data

## Overview

Classical autoencoders are neural networks that can learn efficient low dimensional representations of data in higher dimensional space. The task of an autoencoder is, given an input x, is to map x to a lower dimensional point y such that x can likely be recovered from y. The structure of the underlying autoencoder network can be chosen to represent the data on a smaller dimension, effectively compressing the input. Inspired by this idea, we introduce the model of a quantum autoencoder to perform similar tasks on quantum data. The quantum au-
toencoder is trained to compress a particular dataset of quantum states, where a classical compression algorithm cannot be employed. The parameters of the quantum autoencoder are trained using classical optimization algorithms. We show an example of a simple programmable circuit that can be trained as an efficient autoencoder. We apply our model in the context of quantum simulation to compress ground states of the Hubbard model and molecular Hamiltonians.

![image-20210923140654983](.\QAE.png)

a) A graphical representation of a 6-bit autoencoder with a 3-bit latent space. The map $\epsilon$encodes a 6-bit input (red dots) into a 3-bit intermediate state (yellow dots), after which the decoder $D$ attempts to reconstruct the input bits at the output (green dots). 

b) Circuit implementation of a 6-3-6 quantum autoencoder.

## Quantum Autoencoder Circuit

![image-20210923140637113](.\QAE_Cir.png)

● input data: $\left\{\left|\Psi_{\mathrm{i}}\right\rangle_{\mathrm{AB}}\right\}$is an ensemble of states on $n+k$ qubits, where subsystems $A$ and$B$are comprised of $n$ and $k$ qubits.
		● evaluating the performance: measuring the deviation from the initial input state$\left|\Psi_{\mathrm{i}}\right\rangle$to the output$\rho_{\text {out }}^{i}$, for which we compute the expected  fidelity $\mathrm{F}\left(\left|\Psi_{\mathrm{i}}\right\rangle, \rho_{\text {out }}^{i}\right)=\left\langle\Psi_{\mathrm{j}}\left|\rho_{\text {out }}^{\text {i }}\right| \Psi_{\mathrm{i}}\right\rangle$.
		● for a successful autoencoding, $\mathrm{F}\left(\left|\Psi_{\mathrm{i}}\right\rangle, \rho_{\text {out }}^{i}\right)\approx1$ .
		● $U_{\text { }}^{p}$ is a family of unitary operators acting on $n+k$ qubits, where $p=\left\{p_{1}, p_{2}, \ldots\right\}$ is some set of parameters defining a unitary quantum circuit. 

● We introduce$|\mathrm{a}\rangle_{\mathrm{B'}}$ is a fixed pure reference state of $k$ qubits
		● We wish to find the unitary $U_{\text { }}^{p}$ which maximizes the average fidelity: 

$C_{1}(\vec{p})=\sum_{i} p_{i} \cdot F\left(\left|\psi_{i}\right\rangle, \rho_{i, \vec{p}}^{\text {out }}\right)$

also named the cost function, where
$$
\rho_{i, \vec{p}}^{\text {out }}=\left(U^{\vec{p}}\right)_{A B^{\prime}}^{\dagger} \operatorname{Tr}_{B}\left[U_{A B}^{\vec{p}}\left[\psi_{i_{A B}} \otimes a_{B^{\prime}}\right]\left(U_{A B}^{\vec{p}}\right)^{\dagger}\right]\left(U^{\vec{p}}\right)_{A B^{\prime}}
$$
​		● Last step,  the SWAP operation.

## The Programmable Encoder Circuit

![image-20210923141513505](.\Quantum_circuit.png)

The above figure is a programmable circuit that can be used as an automatic encoder model. It includes all possible controlled general single qubit rotations in a qubit set (represented by $R_i$), plus single qubit rotations at the beginning and end of the cell. All circuits are described with a four qubit input. The cells are separated by red dashed lines.

![image-20210923142001073](.\measure_circuit.png)

The results through the programmable line are measured through swap operation. Judge the fidelity of the result according to this result.

![image-20210923141717162](.\quantum_classical.png)

The above figure is a schematic diagram of a hybrid scheme for training quantum self encoder.

## Pipeline

### Dataset Preprocessing

- Used the MNIST dataset (pictures are of handwritten digits), which contains 60.000 pictures of 28 x 28 pixels.
- Resized the selected pictures to a smaller dimension (2 x {2, 3, 4} pixels) and then flatten the values to obtain the values [b1, ..,bn], where n is 4, 6 or 8.
- In order to transform them to quantum bits, they are encoded using rotational gates (here chosen the $Rx$gate).
- The final qubit register is ：$$
  \left|00 . .0 q_{1} \ldots q_{n}\right\rangle
  $$.

### quantum circuits definition

![QAE_Quantum_Cir](.\QAE_Quantum_Cir.png)

### Quantum circuit construction

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

### train and eval

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

