# 带参数的量子线路示例

## Overview
带参量子线路是用来进行量子机器学习应用的一种方法. VQNet里可以使用构建带参量子线路, 使用自动微分和计算量子线路的梯度，通过优化量子线路参数，达到最小化模型的损失函数的目标。具体代码示例为 [pqc_test.py](../../pyVQNet/examples/pqc_test.py)。

整个过程有以下步骤：

    1.通过QPanda初始化量子线路模拟器machine，开辟量子比特qubit。
    
    2.根据具体任务与设计, 添加含参量子门以及非参量子逻辑门到线路中并构建量子程序prog。
    
    3.使用哈密特量观测特定量子比特上的期望值作为量子线路的输出。
    
    4.使用梯度下降法反向传播梯度并优化参数。


## Pipeline

### data preparation

此示例使用iris数据库进行分类任务，这里取了2类，共100个样本。其中80个作为训练，20个作为验证集。

```python
x_train,x_test,y_train,y_test = load_iris("training_data")    
```

### PQC construct

Module 是一个抽象类，是VQNet自动微分的基类。

```python
class Model(Module):
    def __init__(self):
        super().__init__()
        self.pqc = PQCLayer()
    def forward(self, x):
        x = self.pqc(x)
        return x
```

这里我们定义一个PQCLayer作为实现参量子线路的运算，我们首先使用QPanda分配量子比特qlist以及运算机器machine，forward函数为前传函数，构建了量子线路，并定义量子线路的哈密顿量作为输出。反向函数也需要定义，其使用parameter-shift方法进行量子线路梯度的计算。
```python
class PQCLayer(Module):
    """
    parameterized quantum circuit Layer.It contains paramters can be trained.

    """
    def __init__(self,machine:str="cpu",quantum_number:int=4,rep:int=3,measure_qubits:str="Z0 Z1"):

        """
        machine: 'str' - compute machine
        quantum_number: 'int' - should tensor's gradient be tracked, defaults to False
        rep: 'int' - Ansatz circuits repeat block times
        measure_qubits: 'str' - measure qubits
        """
        super().__init__()

        self.machine = machine
        if machine!="cpu":
            raise ValueError("machine onyl tested on cpu simulation")

        self.machine = pq.CPUQVM()
        self.machine.init_qvm()
        self.qlist = self.machine.qAlloc_many(quantum_number)

        self.history_expectation = []

        self.weights = Parameter(quantum_number*rep +quantum_number,zeros)

        measure_qubits = re.split(r'\s',measure_qubits)

        self.measure_qubits = measure_qubits
        self._rep = rep

    def forward(self, x):
        """
            forward function 
        """
        self.history_expectation = []
        batchsize = x.shape[0]
        batch_exp = np.zeros([batchsize,len(self.measure_qubits)])

        if batchsize == 1:
            prog = paramterized_quautum_circuits(x.data,self.weights.data,self.qlist,self._rep)
            for i,mq in enumerate(self.measure_qubits):
                hamiltion = Hamiltonian(mq)
                exp = self.machine.get_expectation(prog, hamiltion, self.qlist)      
                self.history_expectation.append(exp)
                batch_exp[0,i] = exp
        else:
            for b in range(batchsize):
                xdata = x.data
                b_str = str(b)
                x_ = xdata.select([b_str, ":"])
                prog = paramterized_quautum_circuits(x_,self.weights.data,self.qlist,self._rep)

                for i,mq in enumerate(self.measure_qubits):
                    hamiltion = Hamiltonian(mq)
                    exp = self.machine.get_expectation(prog, hamiltion, self.qlist)      
                    self.history_expectation.append(exp)
                    batch_exp[b,i] = exp
            
        requires_grad = (x.requires_grad or self.weights.requires_grad) and not QTensor.NO_GRAD
        nodes = []
        if x.requires_grad:
            nodes.append(QTensor.GraphNode(tensor=x, df=lambda g: 1))
        if  self.weights.requires_grad:
            nodes.append(
                QTensor.GraphNode(
            tensor = self.weights, df=lambda g: _parashift_grad(g,
                                                      paramterized_quautum_circuits,
                                                     x.data, 
                                                     self.weights.data, 
                                                     self.machine,
                                                     self.qlist,
                                                     self._rep,
                                                     self.measure_qubits,
                                                     self.history_expectation)
                                            )
                    )
        return QTensor(data = batch_exp,requires_grad = requires_grad,nodes =nodes)


def _parashift_grad(g,ForwardFunc,x:CoreTensor,weights:CoreTensor,
                machine,qlists,rep,measure_qubits:str,history_expectation:list,delta:float = 1e-2):
    num_para = 1
    g = np.array(g)
    batchsize = x.shape[0]
    for _ in weights.shape:
        num_para = num_para * _
    grad = np.zeros_like(weights)

    for _ in range(num_para):
            iter = 0

            for b in range(batchsize):
                b_str = str(b)
                x_ = x.select([b_str, ":"])


                for i,mq in enumerate(measure_qubits):
                    hamiltion = Hamiltonian(mq)
                    weights_theta = np.array(weights)
                    weights_theta[_] = weights_theta[_] + delta
                    weights_theta = CoreTensor(weights_theta)
                
                    prog = ForwardFunc(x_,weights_theta,qlists,rep)
                    exp2 = machine.get_expectation(prog, hamiltion, qlists)   # exp w.r.t measure_qubits

                    weights_theta = np.array(weights)
                    weights_theta[_] = weights_theta[_] - delta
                    weights_theta = CoreTensor(weights_theta)
                
                    prog = ForwardFunc(x_,weights_theta,qlists,rep)
                    exp3 = machine.get_expectation(prog, hamiltion, qlists)   # exp w.r.t measure_qubits
                    grad[_] += g[b,i]*(exp2 - exp3) / 2

    grad = CoreTensor(grad)

    return grad
```

### quantum circuits definition

这里的带参量子线路使用QPanda定义 [qpanda](https://pyqpanda-toturial.readthedocs.io/zh/latest/).
我们一般首先用了一个经典数据编码量子线路将经典数据编码为量子态。经典数据编码线路不固定，这里使用的编码线路为IQP encoding (Instantaneous Quantum Polynomial encoding)。

我们有这里7个RX量子逻辑门，有7个经典数据特征分别编码到4个量子比特上。
![loss](cir1.png)

```python
def paramterized_quautum_circuits(input:CoreTensor,param:CoreTensor,qubits,rep:int):
    """
    use qpanda to define circuit

    """
    w = input
    w = np.array(w)
    w = w.squeeze()

    circuit = pq.QCircuit()

    circuit.insert(pq.H(qubits[0]))
    circuit.insert(pq.H(qubits[1])) 
    circuit.insert(pq.H(qubits[2]))
    circuit.insert(pq.H(qubits[3]))    

    circuit.insert(pq.RZ(qubits[0],w[0]))  
    circuit.insert(pq.RZ(qubits[1],w[1])) 
    circuit.insert(pq.RZ(qubits[2],w[2]))
    circuit.insert(pq.RZ(qubits[3],w[3]))

    circuit.insert(pq.CNOT(qubits[0],qubits[1]))
    circuit.insert(pq.RZ(qubits[1],w[4]))  
    circuit.insert(pq.CNOT(qubits[0],qubits[1]))

    circuit.insert(pq.CNOT(qubits[1],qubits[2]))
    circuit.insert(pq.RZ(qubits[2],w[5]))  
    circuit.insert(pq.CNOT(qubits[1],qubits[2]))

    circuit.insert(pq.CNOT(qubits[2],qubits[3]))
    circuit.insert(pq.RZ(qubits[3],w[6]))  
    circuit.insert(pq.CNOT(qubits[2],qubits[3]))
    
    cir = CNOT_RZ_REP_CIRCUITS(qubits,param,rep)
    
    circuit.insert(cir)
    prog = pq.QProg()    
    prog.insert(circuit)    
   
    return prog
```

接下来是参数可变的量子线路部分。我们这里使用HardwareEfficientAnsatz，首先是4个量子比特上RX门，接下来插入多个3个Z门以及4个RZ门的组合模块，具体结构如下：
![loss](cir2.png)

```python
###circuits define
def CNOT_RZ_REP_CIRCUITS(qubits,param,rep:int = 3):
    
    para = param
    para = np.array(para)
    para = para.squeeze()

    circuit = pq.QCircuit()

    for q in range(len(qubits)):
        circuit.insert(pq.RX(qubits[q],para[q]))    

    for r in range(rep):
        for q in range(len(qubits) - 1):
            circuit.insert(pq.CNOT(qubits[q],qubits[q+1]))
            
        for q in range(len(qubits)):
            #print(q)
            circuit.insert(pq.RZ(qubits[q],para[(r+1)*len(qubits) + q]))
    
    return circuit

```

我们在PQCLayer构建时候定义了，第1位第2位的量子比特执行泡利Z算符测量，构建对应的哈密顿量。若第一个哈密顿量更大，则会将此样本归类到标签为“0”的类，同理，若第2个测量值更大，则会将此样本归类到标签为“1”的类。通过模型的训练，量子线路的参数会不断更新，使得训练集上更多的样本被预测正确。

```

def Hamiltonian(input:str):
    """
        Interchange two axes of an array.

        :param input: expect measure qubits.
        :return: hamiltion operator
    
        Examples::
        Hamiltonian("Z0 Z1" )
    """
     
    pauli_str = input  
    pauli_map = PauliOperator(pauli_str, 1)    
    hamiltion = pauli_map.toHamiltonian(True)
    return hamiltion  
```
### optimizer definition

使用SGD随机梯度下降优化器进行参数优化，``model.parameters()``代表待训练的参数，需要在model中以Parameter进行构建。
```
    optimizer = SGD(model.parameters(),lr=0.1)                        
```

### train

模型的损失函数使用``CategoricalCrossEntropy``计算数据和标签之间的softmax交叉熵。使用交叉熵损失测量输入（使用softmax函数计算）的概率和目标之间的分布误差，其中类是互斥的（只有一个类是正的）。
``backward()``开始计算量子线路的梯度

```

    for epoch in range(1,10):
        model.train()
        batch_size = 5
        for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True):

            x0 = (np.pi/2-x[:,:-1])*(np.pi/2-x[:,1:])
            x = np.concatenate([x,x0],axis = 1)
            optimizer.zero_grad()
            output = model(x)
            CCEloss = CategoricalCrossEntropy()
            loss = CCEloss( y,output)
            loss.backward()
            optimizer._step()
            
```

### eval
``` 
    # Evaluation
        model.eval()
        correct = 0
        full_loss = 0
        n_loss = 0
        n_eval = 0
        batch_size = 1
        for x, y in data_generator(x_test, y_test, batch_size=batch_size, shuffle=True):
            x0 = (np.pi/2-x[:,:-1])*(np.pi/2-x[:,1:])
            x = np.concatenate([x,x0],axis = 1)
            output = model(x)
            CCEloss = CategoricalCrossEntropy()            
            loss = CCEloss( y,output)
            full_loss += loss.item()
            print("Epoch:", epoch, "iter:", n_loss, "loss:", loss.item())
            np_output = np.array(output.data,copy=False)
            mask  = np_output.argmax(1) == y.argmax(1)
            correct += sum(mask)
            n_eval += 1
            n_loss += 1

        print(f"Eval Accuracy: {correct/n_eval}")
        F1.write(f"{full_loss / n_loss}\t{correct/n_eval}\n")
    F1.close()
    print("\ndone\n")
    del model

```

### results

下图分别为训练集train以及验证集valid的损失函数以及准确率随着训练次数的变化图，可见准确率在增加，损失在降低。

   ![loss](Figure_1.png)
   ![acc](Figure_2.png)

