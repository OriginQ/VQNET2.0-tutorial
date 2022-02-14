# Quantum transfer learning

## Overview
We apply a machine learning method called transfer learning to image classifier based on hybrid classical quantum network. We will code up a simple example that integrates **Pyqpanda** with **VQNET**. 

Transfer learning is a well-established technique for training artificial neural networks, which is based on the general intuition that if a pre-trained network is good at solving a given problem, then, with just a bit of additional training, it can be used to also solve a different but related problem. this idea can be formalized in terms of two abstract networks $A$ and $B$, independently from their quantum or classical physical nature.

<img src=".\transfer_learning_general.png" alt="transfer_learning_general" style="zoom:50%;" />

As sketched in the above figure, one can give the following **general definition of the transfer learning method**:

1. Take a network $A$ that has been pre-trained on a dataset $D_A$ and for a given task $T_A$.
2. Remove some of the final layers. In this way, the resulting truncated network $A′$ can be used as a feature extractor.
3. Connect a new trainable network $B$ at the end of the pre-trained network $A′$.
4. Keep the weights of $A′$ constant, and train the final block $B$ with a new dataset $D_B$ and/or for a new task of interest $T_B$.

When dealing with hybrid systems, depending on the physical nature (classical or quantum) of the networks $A$ and $B$, one can have different implementations of transfer learning as

## Classical-to-quantum transfer learning

1. As pre-trained network AA we use **CNN**.
2. After removing its final layer we obtain $A′$, a pre-processing block which maps any input high-resolution image into 128 abstract features.
3. Such features are classified by a 4-qubit “dressed quantum circuit” $B$, a variational quantum circuit sandwiched between two classical layers.
4. The hybrid model is trained, keeping $A′$ constant, on the  MNIST dataset.

A graphical representation of the full data processing pipeline is given in the figure below.

![flow_chart](F:\vqnet\vqnet\tutorials\QTransferLearning\flow_chart.png)



## Pipeline

### data preparation 
We will create a simple hybrid neural network from [MNIST datasets](http://yann.lecun.com/exdb/mnist/) . We first load MNIST . These will be used as the input of our neural network for classification.

```python
def load_mnist(dataset="training_data", digits=np.arange(2), path="..//..//data//MNIST_data"):  # 下载数据
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

def data_select(train_num, test_num):
    x_train, y_train = load_mnist("training_data")  # 下载训练数据
    # print(x_train.shape, y_train.shape)
    x_test, y_test = load_mnist("testing_data")
    # print(x_test.shape, y_test.shape)

    # Train Leaving only labels 0 and 1
    idx_train = np.append(np.where(y_train == 0)[0][:train_num],
                          np.where(y_train == 1)[0][:train_num])

    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    # print(x_train.shape, y_train.shape)
    x_train = x_train / 255
    y_train = np.eye(2)[y_train].reshape(-1, 2)

    # Test Leaving only labels 0 and 1
    idx_test = np.append(np.where(y_test == 0)[0][:test_num],
                         np.where(y_test == 1)[0][:test_num])

    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    x_test = x_test / 255
    y_test = np.eye(2)[y_test].reshape(-1, 2)
    # print(x_train.shape, y_train.shape)
    return x_train, y_train, x_test, y_test
```

### quantum circuits definition

This experiment is a multi classification task, Four qubits are used..

![test_cir](.\QTransferLearning_cir.png)

```python
    n_qubits = 4  # Number of qubits
    q_depth = 6  # Depth of the quantum circuit (number of variational layers)
    q_delta = 0.01  # Initial spread of random quantum weights

    def Q_H_layer(qubits, nqubits):
        """Layer of single-qubit Hadamard gates.
        """
        circuit = pq.QCircuit()
        for idx in range(nqubits):
            circuit.insert(pq.H(qubits[idx]))
        return circuit

    def Q_RY_layer(qubits, w):
        """Layer of parametrized qubit rotations around the y axis.
        """
        circuit = pq.QCircuit()
        for idx, element in enumerate(w):
            circuit.insert(pq.RY(qubits[idx], element))
        return circuit

    def Q_entangling_layer(qubits, nqubits):
        """Layer of CNOTs followed by another shifted layer of CNOT.
        """
        circuit = pq.QCircuit()
        for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        return circuit

    def Q_quantum_net(q_input_features, q_weights_flat, qubits, cubits, machine):
        """
        The variational quantum circuit.
        """
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(n_qubits)
        circuit = pq.QCircuit()

        # Reshape weights
        # q_weights = q_weights_flat.reshape(q_depth, n_qubits)
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
        # print(prog)
        # draw_qprog(prog, 'pic', filename='D:/test_cir.png')

        exp_vals = []
        for position in range(n_qubits):
            pauli_str = "Z" + str(position)
            pauli_map = pq.PauliOperator(pauli_str, 1)
            hamiltion = pauli_map.toHamiltonian(True)
            exp = machine.get_expectation(prog, hamiltion, qubits)
            exp_vals.append(exp)

        return exp_vals
```
### Creating the Hybrid Neural Network

Here, we first extract the features of the 28 * 28 data through the CNN layer, while keeping the parameters of the CNN feature extractor unchanged. Change the last layer of CNN, first connect the full connection layer, and convert the 128 feature data of the penultimate layer of CNN into 4 feature data. Then connect the quantum processing circuit. Finally, it is transformed into 10-dimensional output through full connection layer to adapt to 10 classification tasks.

```python
class Q_DressedQuantumNet(Module):
    """
        VQNET module implementing the *dressed* quantum net.
        """

    def __init__(self):
        """
            Definition of the *dressed* layout.
            """

        super().__init__()
        self.pre_net = Linear(128, n_qubits)
        self.post_net = Linear(n_qubits, 10)
        self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

        def forward(self, input_features):
            """
            Defining how tensors are supposed to move through the *dressed* quantum
            net.
            """

            # obtain the input features for the quantum circuit
            # by reducing the feature dimension from 512 to 4
            pre_out = self.pre_net(input_features)
            q_in = tensor.tanh(pre_out) * np.pi / 2.0
            q_out_elem = self.temp_Q(q_in)
            result = q_out_elem
            # return the two-dimensional prediction from the postprocessing layer
            return self.post_net(result)
        
# classical CNN
class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = Conv2D(input_channels=1, output_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.BatchNorm2d1 = BatchNorm2d(16)
        self.Relu1 = F.ReLu()

        self.conv2 = Conv2D(input_channels=16, output_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.BatchNorm2d2 = BatchNorm2d(32)
        self.Relu2 = F.ReLu()
        self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")

        self.conv3 = Conv2D(input_channels=32, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.BatchNorm2d3 = BatchNorm2d(64)
        self.Relu3 = F.ReLu()

        self.conv4 = Conv2D(input_channels=64, output_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.BatchNorm2d4 = BatchNorm2d(128)
        self.Relu4 = F.ReLu()
        self.maxpool4 = MaxPool2D([2, 2], [2, 2], padding="valid")

        self.fc1 = Linear(input_channels=128 * 4 * 4, output_channels=1024)
        self.fc2 = Linear(input_channels=1024, output_channels=128)
        self.fc3 = Linear(input_channels=128, output_channels=10)


    def forward(self, x):

        # x = self.Relu1(self.BatchNorm2d1(self.conv1(x)))
        x = self.Relu1(self.conv1(x))

        # x = self.maxpool2(self.Relu2(self.BatchNorm2d2(self.conv2(x))))
        x = self.maxpool2(self.Relu2(self.conv2(x)))

        # x = self.Relu3(self.BatchNorm2d3(self.conv3(x)))
        x = self.Relu3(self.conv3(x))

        # x = self.maxpool4(self.Relu4(self.BatchNorm2d4(self.conv4(x))))
        x = self.maxpool4(self.Relu4(self.conv4(x)))


        x = tensor.flatten(x, 1)  # view(1, -1)  # 1 256

        x = F.ReLu()(self.fc1(x))  # 1 64
        x = F.ReLu()(self.fc2(x))  # 1 64
        x = self.fc3(x)  # 1 1

        return x
```
### Preparation of classical CNN network feature extractor

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath('..\\'))

# print(sys.path)
import pyvqnet
from pyvqnet.data import mnist
from pyvqnet.nn.module import Module
from pyvqnet.nn.linear import Linear
from pyvqnet.nn.conv import Conv2D
from pyvqnet.utils.storage import load_parameters, save_parameters

from pyvqnet.nn import activation as F
from pyvqnet.nn.pooling import MaxPool2D
from pyvqnet.nn.dropout import Dropout
from pyvqnet.nn.batch_norm import BatchNorm2d
from pyvqnet.nn.loss import CategoricalCrossEntropy,SoftmaxCrossEntropy
from pyvqnet.nn.loss import BinaryCrossEntropy
from pyvqnet.optim.sgd import SGD
from pyvqnet.optim.adam import Adam
from pyvqnet.data.data import data_generator
from pyvqnet.tensor import tensor
from pyvqnet.tensor.tensor import QTensor
import pyqpanda as pq
from pyqpanda import *
import matplotlib
from pyvqnet.nn.module import *
from pyvqnet.utils.initializer import *
from pyvqnet.utils import initializer
from pyvqnet.qnn.quantumlayer import QuantumLayer

try:
    matplotlib.use('TkAgg')
except:
    pass
import time

def classcal_cnn_model_making():
    # load train data
    x_train, y_train = load_mnist("training_data", digits=np.arange(10))
    x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np.eye(10)[y_train].reshape(-1, 10)
    y_test = np.eye(10)[y_test].reshape(-1, 10)

    model = CNN()
    optimizer = SGD(model.parameters(), lr=0.005)
    loss_func = SoftmaxCrossEntropy()

    epochs = 200
    loss_list = []
    model.train()

    eval_time = []
    SAVE_FLAG = True
    temp_loss = 0
    for epoch in range(1, epochs):
        total_loss = []
        for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):

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

        if SAVE_FLAG:
            temp_loss = loss_list[-1]
            save_parameters(model.state_dict(), "CNN_TL.model")
            SAVE_FLAG = False
        else:
            if temp_loss > loss_list[-1]:
                temp_loss = loss_list[-1]
                save_parameters(model.state_dict(), "CNN_TL.model")

    plt.plot(loss_list)
    plt.title('VQNET QCNN Transfer Learning')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.savefig("cnn_transfer_learning_classical")
    plt.show()

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
    print(f"Eval Accuracy: {correct / (n_eval*4)}")

    n_samples_show = 6
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    model.eval()
    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        if count == n_samples_show:
            break
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        pred = QTensor.argmax(output, [1])
        axes[count].imshow(x[0].squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
        count += 1
    plt.show()
   
if __name__ == "__main__":
    classcal_cnn_model_making()
```

Save the trained CNN extractor as "cnn_tl.model".

### Constructing quantum transfer learning CNN hybrid model

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath('..\\'))

# print(sys.path)
import pyvqnet
from pyvqnet.data import mnist
from pyvqnet.nn.module import Module
from pyvqnet.nn.linear import Linear
from pyvqnet.nn.conv import Conv2D
from pyvqnet.utils.storage import load_parameters, save_parameters

from pyvqnet.nn import activation as F
from pyvqnet.nn.pooling import MaxPool2D
from pyvqnet.nn.dropout import Dropout
from pyvqnet.nn.batch_norm import BatchNorm2d
from pyvqnet.nn.loss import CategoricalCrossEntropy,SoftmaxCrossEntropy
from pyvqnet.nn.loss import BinaryCrossEntropy
from pyvqnet.optim.sgd import SGD
from pyvqnet.optim.adam import Adam
from pyvqnet.data.data import data_generator
from pyvqnet.tensor import tensor
from pyvqnet.tensor.tensor import QTensor
import pyqpanda as pq
from pyqpanda import *
import matplotlib
from pyvqnet.nn.module import *
from pyvqnet.utils.initializer import *
from pyvqnet.utils import initializer
from pyvqnet.qnn.quantumlayer import QuantumLayer

try:
    matplotlib.use('TkAgg')
except:
    pass
import time

def quantum_cnn_TransferLearning():

    n_qubits = 4  # Number of qubits
    q_depth = 6  # Depth of the quantum circuit (number of variational layers)
    q_delta = 0.01  # Initial spread of random quantum weights

    def Q_H_layer(qubits, nqubits):
        """Layer of single-qubit Hadamard gates.
        """
        circuit = pq.QCircuit()
        for idx in range(nqubits):
            circuit.insert(pq.H(qubits[idx]))
        return circuit

    def Q_RY_layer(qubits, w):
        """Layer of parametrized qubit rotations around the y axis.
        """
        circuit = pq.QCircuit()
        for idx, element in enumerate(w):
            circuit.insert(pq.RY(qubits[idx], element))
        return circuit

    def Q_entangling_layer(qubits, nqubits):
        """Layer of CNOTs followed by another shifted layer of CNOT.
        """
        circuit = pq.QCircuit()
        for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        return circuit

    def Q_quantum_net(q_input_features, q_weights_flat, qubits, cubits, machine):
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
        # print(prog)
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
        VQNET module implementing the *dressed* quantum net.
        """

        def __init__(self):
            """
            Definition of the *dressed* layout.
            """

            super().__init__()
            self.pre_net = Linear(128, n_qubits)
            self.post_net = Linear(n_qubits, 10)
            self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

        def forward(self, input_features):
            """
            Defining how tensors are supposed to move through the *dressed* quantum
            net.
            """
            # obtain the input features for the quantum circuit
            # by reducing the feature dimension from 512 to 4
            pre_out = self.pre_net(input_features)
            q_in = tensor.tanh(pre_out) * np.pi / 2.0
            q_out_elem = self.temp_Q(q_in)
            result = q_out_elem
            # return the two-dimensional prediction from the postprocessing layer
            return self.post_net(result)

    class quantum_transfer_layer(Module):
        def __init__(self, ):
            super().__init__()
            self.pre_net = Linear(128, n_qubits)
            self.quantum_layer = Hybrid()
            self.post_net = Linear(n_qubits, 10)

        def forward(self, x):
            pre_out = self.pre_net(x)
            q_in = tensor.tanh(pre_out) * np.pi / 2.0
            result = self.quantum_layer(q_in)
            return self.post_net(result)


    x_train, y_train = load_mnist("training_data", digits=np.arange(10))  # 下载训练数据
    x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
    x_train = x_train[:2000]
    y_train = y_train[:2000]
    x_test = x_test[:400]
    y_test = y_test[:400]

    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np.eye(10)[y_train].reshape(-1, 10)
    y_test = np.eye(10)[y_test].reshape(-1, 10)

    model = CNN()
    model_param = load_parameters("CNN_TL.model")
    model.load_state_dict(model_param)

    loss_func = SoftmaxCrossEntropy()

    epochs = 100
    loss_list = []
    acces = []
    eval_losses = []
    eval_acces = []

    model_hybrid = model
    print(model_hybrid)

    for param in model_hybrid.parameters():
        param.requires_grad = False
    # Replace the full connection layer with quantum circuit
    model_hybrid.fc3 = Q_DressedQuantumNet()

    optimizer_hybrid = Adam(model_hybrid.fc3.parameters(), lr=0.001)
    model_hybrid.train()
    eval_time = []
    SAVE_FLAG = True
    temp_loss = 0
    for epoch in range(1, epochs):
        total_loss = []
        for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):
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
        if SAVE_FLAG:
            temp_loss = loss_list[-1]
            # only save FC3
            save_parameters(model_hybrid.fc3.state_dict(), "QCNN_TL_FC3.model")
            # Save the entire model
            save_parameters(model_hybrid.state_dict(), "QCNN_TL_ALL.model")
            SAVE_FLAG = False
        else:
            if temp_loss > loss_list[-1]:
                temp_loss = loss_list[-1]
                # only save FC3
                save_parameters(model_hybrid.fc3.state_dict(), "QCNN_TL_FC3.model")
                # Save the entire model
                save_parameters(model_hybrid.state_dict(), "QCNN_TL_ALL.model")

        correct = 0
        n_eval = 0
        loss_temp =[]
        for x1, y1 in data_generator(x_test, y_test, batch_size=4, shuffle=True):
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

    plt.title('model loss')
    plt.plot(loss_list, color='green', label='train_losses')
    plt.plot(eval_losses, color='red', label='eval_losses')
    plt.ylabel('loss')
    plt.legend(["train_losses", "eval_losses"])
    plt.savefig("qcnn_transfer_learning_classical")
    plt.show()
    plt.close()


    n_samples_show = 6
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    model_hybrid.eval()
    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        if count == n_samples_show:
            break
        x = x.reshape(-1, 1, 28, 28)
        output = model_hybrid(x)
        pred = QTensor.argmax(output, [1])
        axes[count].imshow(x[0].squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
        count += 1
    plt.show()
   
if __name__ == "__main__":
    quantum_cnn_TransferLearning()
```

The model can save the whole model or only the trained model. The difference is that all models are reserved, and only one model needs to be loaded when forecasting calls. Save the model of FC3 layer. Two models need to be called during prediction.

### optimizer definition

Use Adam for this task is enough, model_hybrid.fc3.parameters() is parameters need to be calculated. Because the feature extraction layer CNN remains unchanged, only the FC3 layer replaced by the quantum circuit layer needs to be updated.

```
    optimizer_hybrid = Adam(model_hybrid.fc3.parameters(), lr=0.001)
```

### Training the Network 

```
model contains quantum circuits or classic data layer 
SoftmaxCrossEntropy() is loss function
backward() calculates model.parameters gradients 
```

```python
if __name__ == "__main__":
    # classcal_cnn_model_making()
    quantum_cnn_TransferLearning()
```

### Training loss result chart

![qcnn_transfer_learning_classical](.\qcnn_transfer_learning_classical.png)

### Prediction network

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath('..\\'))

# print(sys.path)
import pyvqnet
from pyvqnet.data import mnist
from pyvqnet.nn.module import Module
from pyvqnet.nn.linear import Linear
from pyvqnet.nn.conv import Conv2D
from pyvqnet.utils.storage import load_parameters, save_parameters

from pyvqnet.nn import activation as F
from pyvqnet.nn.pooling import MaxPool2D
from pyvqnet.nn.dropout import Dropout
from pyvqnet.nn.batch_norm import BatchNorm2d
from pyvqnet.nn.loss import CategoricalCrossEntropy,SoftmaxCrossEntropy
from pyvqnet.nn.loss import BinaryCrossEntropy
from pyvqnet.optim.sgd import SGD
from pyvqnet.optim.adam import Adam
from pyvqnet.data.data import data_generator
from pyvqnet.tensor import tensor
from pyvqnet.tensor.tensor import QTensor
import pyqpanda as pq
from pyqpanda import *
import matplotlib
from pyvqnet.nn.module import *
from pyvqnet.utils.initializer import *
from pyvqnet.utils import initializer
from pyvqnet.qnn.quantumlayer import QuantumLayer

try:
    matplotlib.use('TkAgg')
except:
    pass
import time

def quantum_cnn_TransferLearning_predict():

    n_qubits = 4  # Number of qubits
    q_depth = 6  # Depth of the quantum circuit (number of variational layers)
    q_delta = 0.01  # Initial spread of random quantum weights

    def Q_H_layer(qubits, nqubits):
        """Layer of single-qubit Hadamard gates.
        """
        circuit = pq.QCircuit()
        for idx in range(nqubits):
            circuit.insert(pq.H(qubits[idx]))
        return circuit

    def Q_RY_layer(qubits, w):
        """Layer of parametrized qubit rotations around the y axis.
        """
        circuit = pq.QCircuit()
        for idx, element in enumerate(w):
            circuit.insert(pq.RY(qubits[idx], element))
        return circuit

    def Q_entangling_layer(qubits, nqubits):
        """Layer of CNOTs followed by another shifted layer of CNOT.
        """
        circuit = pq.QCircuit()
        for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        return circuit

    def Q_quantum_net(q_input_features, q_weights_flat, qubits, cubits, machine):
    # def Q_quantum_net(q_input_features, q_weights_flat):
        """
        The variational quantum circuit.
        """
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(n_qubits)
        circuit = pq.QCircuit()

        # Reshape weights
        # q_weights = q_weights_flat.reshape(q_depth, n_qubits)
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
        # print(prog)
        # draw_qprog(prog, 'pic', filename='D:/test_cir.png')

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
        VQNET module implementing the *dressed* quantum net.
        """

        def __init__(self):
            """
            Definition of the *dressed* layout.
            """

            super().__init__()
            self.pre_net = Linear(128, n_qubits)
            self.post_net = Linear(n_qubits, 10)
            self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

        def forward(self, input_features):
            """
            Defining how tensors are supposed to move through the *dressed* quantum
            net.
            """

            # obtain the input features for the quantum circuit
            # by reducing the feature dimension from 512 to 4
            pre_out = self.pre_net(input_features)
            q_in = tensor.tanh(pre_out) * np.pi / 2.0
            q_out_elem = self.temp_Q(q_in)

            result = q_out_elem
            # return the two-dimensional prediction from the postprocessing layer
            return self.post_net(result)



    class quantum_transfer_layer(Module):
        def __init__(self, ):
            super().__init__()
            self.pre_net = Linear(128, n_qubits)
            self.quantum_layer = Hybrid()
            self.post_net = Linear(n_qubits, 10)

        def forward(self, x):
            pre_out = self.pre_net(x)
            q_in = tensor.tanh(pre_out) * np.pi / 2.0
            result = self.quantum_layer(q_in)
            return self.post_net(result)

    x_train, y_train = load_mnist("training_data", digits=np.arange(10))  # 下载训练数据
    x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
    x_train = x_train[:2000]
    y_train = y_train[:2000]
    x_test = x_test[:500]
    y_test = y_test[:500]

    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np.eye(10)[y_train].reshape(-1, 10)
    y_test = np.eye(10)[y_test].reshape(-1, 10)

    # # The first method: store separately and read separately
    # model = CNN()
    # model_param = load_parameters("QCNN_TL_1.model")
    # model.load_state_dict(model_param)
    # model_hybrid = model
    # for param in model_hybrid.parameters():
    #     param.requires_grad = False
    # model_hybrid.fc3 = Q_DressedQuantumNet()
    # model_param_quantum = load_parameters("QCNN_TL_FC3.model")
    # model_hybrid.fc3.load_state_dict(model_param_quantum)
    # model_hybrid.eval()

    # The second method: unified storage and unified reading
    model = CNN()
    model_hybrid = model
    model_hybrid.fc3 = Q_DressedQuantumNet()
    for param in model_hybrid.parameters():
        param.requires_grad = False
    model_param_quantum = load_parameters("QCNN_TL_ALL.model")
    # print("model_param_quantum: ", model_param_quantum)
    model_hybrid.load_state_dict(model_param_quantum)
    model_hybrid.eval()

    loss_func = SoftmaxCrossEntropy()
    eval_losses = []

    correct = 0
    n_eval = 0
    loss_temp =[]
    eval_batch_size = 4
    for x1, y1 in data_generator(x_test, y_test, batch_size=eval_batch_size, shuffle=True):
        x1 = x1.reshape(-1, 1, 28, 28)
        output = model_hybrid(x1)
        loss = loss_func(y1, output)
        np_loss = np.array(loss.data)
        np_output = np.array(output.data, copy=False)
        mask = (np_output.argmax(1) == y1.argmax(1))
        correct += np.sum(np.array(mask))

        n_eval += 1
        loss_temp.append(np_loss)
    # print("correct: ", correct)
    # print("n_eval: ", n_eval*4)
    eval_losses.append(np.sum(loss_temp) / n_eval)
    print(f"Eval Accuracy: {correct / (eval_batch_size*n_eval)}")


    n_samples_show = 6
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    model_hybrid.eval()
    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        if count == n_samples_show:
            break
        x = x.reshape(-1, 1, 28, 28)
        output = model_hybrid(x)
        pred = QTensor.argmax(output, [1])
        axes[count].imshow(x[0].squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
        count += 1
    plt.show()


if __name__ == "__main__":
    # classcal_cnn_model_making()
    # quantum_cnn_TransferLearning()
    quantum_cnn_TransferLearning_predict()

```

### Forecast result chart

![qcnn_predict](.\qcnn_transfer_learning_predict.png)

At present, only 2000 data are used for training. If you want better data effect, you need to increase the data to continue training. MNIST has 60000 data training sets and 10000 test sets. At present, it mainly shows the feasibility and effect of transfer learning under VQNet framework.