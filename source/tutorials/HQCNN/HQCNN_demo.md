## demo of Hybrid quantum-classical Neural Networks with VQNET and Pyqpanda

## Overview
we explore how a classical neural network can be partially quantized to create a hybrid quantum-classical neural network. We will code up a simple example that integrates **Pyqpanda** with **VQNET**. 

## Pipeline

### data preparation 
We will create a simple hybrid neural network from [MNIST datasets](http://yann.lecun.com/exdb/mnist/) Two types of digital (0 or 1) images are classified. We first load MNIST and filter images containing 0 and 1. These will be used as the input of our neural network for classification.

```python
def load_mnist(dataset="training_data", digits=np.arange(2), path="../../../dataset/MNIST_data"):        
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
    x_train, y_train = load_mnist("training_data")  
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

This experiment is a binary classification task, so one quantum can be used.

```python
          ┌─┐ ┌────────────┐ ┌─┐ 
q_0:  |0>─┤H├ ┤RY(3.141593)├ ┤M├ 
          └─┘ └────────────┘ └╥┘ 
 c_0:  0 ═════════════════════╩═
```

```python
def circuit(weights):
    num_qubits = 1
    machine = pq.CPUQVM()
    machine.init_qvm()
    qubits = machine.qAlloc_many(num_qubits)
    cbits = machine.cAlloc_many(num_qubits)
    circuit = pq.QCircuit()

    circuit.insert(pq.H(qubits[0]))
    circuit.insert(pq.RY(qubits[0], weights[0]))

    prog = pq.QProg()
    prog.insert(circuit)
    prog << measure_all(qubits, cbits)

    result = machine.run_with_configuration(prog, cbits, 100)
    counts = np.array(list(result.values()))
    states = np.array(list(result.keys())).astype(float)
    probabilities = counts / 100
    expectation = np.sum(states * probabilities)
    return expectation
```
### Creating the Hybrid Neural Network

```python
class Hybrid(Module):
    """ Hybrid quantum - Quantum layer definition """

    def __init__(self, shift):
        super(Hybrid, self).__init__()
        self.shift = shift

    def forward(self, input): #input is weights
        self.input = input
        expectation_z = circuit(np.array(input.data))
        result = [[expectation_z]]
        requires_grad = input.requires_grad and not QTensor.NO_GRAD

        def _backward_mnist(g, input):
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
            nodes.append(QTensor.GraphNode(tensor=input, df=lambda g: _backward_mnist(g, input)))
        # return result
        return QTensor(data=result, requires_grad=requires_grad, nodes=nodes)


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
```
### optimizer definition

Use Adam for this task is enough,model.parameters() is parameters need to be calculated

```
optimizer = Adam(model.parameters(), lr=0.005)
```

### Training the Network 

```
model contains quantum circuits or classic data layer 
CategoricalCrossEntropy() is loss function
backward() calculates model.parameters gradients 
```

```python
if __name__=="__main__":
    x_train, y_train, x_test, y_test = data_select(100, 50) # load 100 train data，50 test data
    model = Net()  
    optimizer = Adam(model.parameters(), lr=0.005)
    loss_func = CategoricalCrossEntropy()
    
    epochs = 50 
    loss_list = []
    model.train()

    for epoch in range(1, epochs):
        total_loss = []
        for x, y in data_generator(x_train, y_train, batch_size=1, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            optimizer.zero_grad()
            # Forward pass
            output = model(x)
            # Calculating loss
            loss = loss_func(y, output)  # target output
            loss_numpy = np.array(loss.data)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer._step()
            total_loss.append(loss_numpy)

        loss_list.append(np.sum(total_loss) / len(total_loss))
        print("loss_list: ", loss_list[-1])
       
    plt.plot(loss_list)
    plt.title('VQNET NN Training')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.show()
```

```python
1 loss is : 0.6388268280
2 loss is : 0.4221732330
3 loss is : 0.2906876373
4 loss is : 0.2026059341
5 loss is : 0.1447878075
6 loss is : 0.1102785969
7 loss is : 0.0856013203
8 loss is : 0.0681168079
9 loss is : 0.0566613102
10 loss is : 0.0458295727
11 loss is : 0.0379011559
12 loss is : 0.0323327708
13 loss is : 0.0272748518
14 loss is : 0.0234790444
15 loss is : 0.0202727675
16 loss is : 0.0175541949
17 loss is : 0.0153800941
18 loss is : 0.0134823751
19 loss is : 0.0118615890
20 loss is : 0.0104916596
21 loss is : 0.0093538606
22 loss is : 0.0082624024
23 loss is : 0.0073458111
24 loss is : 0.0064947999
25 loss is : 0.0058320701
26 loss is : 0.0052365732
27 loss is : 0.0047013408
28 loss is : 0.0042242950
29 loss is : 0.0037678862
30 loss is : 0.0033725163
31 loss is : 0.0030380586
32 loss is : 0.0027178711
33 loss is : 0.0024608523
34 loss is : 0.0022468807
35 loss is : 0.0020052427
36 loss is : 0.0018217954
37 loss is : 0.0016216293
38 loss is : 0.0014647576
39 loss is : 0.0013285525
40 loss is : 0.0012274934
41 loss is : 0.0011098573
42 loss is : 0.0009783304
43 loss is : 0.0008919446
44 loss is : 0.0008021845
45 loss is : 0.0007247931
46 loss is : 0.0006590705
47 loss is : 0.0005963836
48 loss is : 0.0005356621
49 loss is : 0.0004881200
```

### eval
```python
    model.eval()
    correct = 0
    total_loss1 = []
    n_eval = 0
    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        loss = loss_func(y, output)
        loss_numpy = np.array(loss.data)
        np_output = np.array(output.data, copy=False)
        mask = (np_output.argmax(1) == y.argmax(1))
        correct += np.sum(np.array(mask))
        n_eval += 1
    print(f"Eval Accuracy: {correct / n_eval}")
```


### result

![image-20210918160816220](.\train_loss.png)

test result:

```python
Eval Accuracy: 1.0
```

```python
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
```

![image-20210918160914177](.\eval_test.png)