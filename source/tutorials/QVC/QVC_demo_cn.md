## 																											变分量子分类器实例
## 概述

这个例子旨在确定数据中1的个数，本质上是一个二分类任务。不同与其他相关任务，变分量子分类器（Quantum Variable Classifier，QVC）可以利用Pauliz门去改变量子线路的初态，例如将|0000>态映射到|0101>态，这里主要是根据输出的参数。

大致的步骤如下：

    1.初始化量子线路和参数.
    
    2.根据任务的要求，为量子线路添加参数化或者非参数化量子逻辑门.
    
    3.采用QVC原理方法去实现量子态转换.
    
    4.利用sgd优化器去实现量子线路参数最优化.

具体例子可见 [qvc_test.py](../../pyVQNet/examples/qvc_test.py)。

## 步骤

### 数据准备

该实例的数据量较小，存放在qvc_data.txt和qvc_data_test.txt文件中，总共16个例子，每个例子由0，1组成（共5个），前四个数字用来训练（测试）网络，最后一个则是训练（测试）数据的标签。如果设定了shuffle = true，模型则通过打乱次序的方式来接受数据，这里的打乱是以batch为单位而不是所有的数据。

```python

def get_data(PATH):
    datasets = np.loadtxt(PATH) 
    data = datasets[:,:-1]	
    label = datasets[:,-1].astype(int)
    label = np.eye(2)[label].reshape(-1,2)
    return data, label

def dataloader(data,label,batch_size, shuffle = True)->np:
    if shuffle:
        for _ in range(len(data)//batch_size):
            random_index = np.random.randint(0, len(data), (batch_size, 1))
            yield data[random_index].reshape(batch_size,-1),label[random_index].reshape(batch_size,-1)
    else:
        for i in range(0,len(data)-batch_size+1,batch_size):
            yield data[i:i+batch_size], label[i:i+batch_size]

```

### QVC 线路结构

Module 是autograd层的抽象类，将Qvc线路作为一个成员变量。
```python
class Model(Module):
    def __init__(self,shape):
        self.qvc = Qvc(shape)
        self.fc1 = Linear(4,16)
        self.fc2 = Linear(16,2)
    def forward(self, x):
        return self.qvc(x)
        # x = self.fc1(x)
        # return self.fc2(x)

```
利用PyQpanda建立的参数化量子线路，其中参数的维度为4
```python
class Qvc(Module):
    def __init__(self,shape,  q_delta = 1e-2):
        super().__init__()
        self.weights = Parameter(shape) #2*4*3
        self.delta = q_delta
        self.machine  = pq.CPUQVM()
        self.machine.init_qvm()
        self.n_qubits = self.machine.qAlloc_many(4)
        #self.params = Parameter(num_layers*3)
        self.bias = Parameter([1])
        #self.measure_qubits = re.split('\s',"Z0 Z1")
        self.last = []
    def forward(self,x:QTensor):#5*4, all x[i] must =0 or 1
        nodes = []
        self.last.clear()
        requires_grad = x.requires_grad and not QTensor.NO_GRAD
        batch_size = x.data.shape[0]
        data = np.zeros((batch_size,2))
        for mini in range(batch_size):
            xx = x.data.select([str(mini)])#xx.shape[1,4]
            prog = build_circult(self.weights.data, xx, self.n_qubits)
            prob = self.machine.prob_run_dict(prog, self.n_qubits[0], -1)
            prob = list(prob.values())
            for i in prob:
                self.last.append(i)
            data[mini] = prob
        if self.weights.requires_grad:
            nodes.append(QTensor.GraphNode(tensor=self.weights,df = lambda g:_get_grad(g, x.data, self.weights.data, build_circult,self.delta,self.machine,self.n_qubits,self.last)
            ))
        return QTensor(data,requires_grad,nodes)
def _get_grad(g:CoreTensor, x:CoreTensor, weights:CoreTensor, forward_circult,delta,machine,nqubits,last):
    num = 1
    batch_size = x.shape[0]
    gn = np.array(g)
    grad = tensor.zeros(weights.shape).data
    gg = grad.getdata()
    gg = gg.flatten()
    for i in weights.shape:
        num *= i
    for i in range(num):
        iter = 0
        paras = weights.clone()
        para = paras.getdata()
        para = para.flatten()
        para[i] += delta
        para = CoreTensor(para).reshape(paras.shape)
        for j in range(batch_size):
            xx = x.select([str(j)])
            prog = forward_circult(para, xx, nqubits)
            prob = machine.prob_run_dict(prog, nqubits[0], -1)
            prob = list(prob.values())
            for m in range(len(prob)):
                gg[i] += gn[j,m]*(prob[m]-last[iter])/delta
                iter += 1
        gg[i]/=batch_size

    gg = gg.reshape(grad.shape)
    grad = CoreTensor(gg)
    return grad

```

### 量子线路定义

量子线路每个量子比特上有两个Rot操作（由RX，RY，RZ组成），通过测量第一个量子比特以获得最终的概率值。根据量子门参数的不同可以通过增加Pauliz门去改变量子的初态。

![qvc_circuit](qvc_circuit.png)

```python

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

	circult.insert(pq.Z(nqubits[0]))#pauli z(0)
	
	prog = pq.QProg()  
	prog.insert(circult)
	return prog

```

### 优化器定义

本实例中使用的是sgd优化器，`model.parameters()`即需要更新的参数。

```python
optimizer = sgd.SGD(model.parameters(),lr =0.5)
```

### 训练

model可以兼容量子线路和经典数据层，CategoricalCrossEntropy()是一个损失函数，利用backward() 去计算模型参数的梯度

```python
nqubits = 4
num_layer = 2
model = Model([num_layer,nqubits,3])

optimizer = sgd.SGD(model.parameters(),lr =0.5)
batch_size = 3
epoch = 10
loss = CategoricalCrossEntropy()
print("start training..............")
model.train()
PATH = os.path.abspath('./dataset/qvc_data.txt')
datas,labels = get_data(PATH)
for i in range(epoch):
    count=0
    sum_loss = 0
    accuary = 0
    for data,label in dataloader(datas,labels,batch_size):
        optimizer.zero_grad()
        data,label = QTensor(data,requires_grad=True), QTensor(label)
        result = model(data)

        loss_b = loss(label,result)
        loss_b.backward()
        optimizer._step()
        sum_loss += loss_b.item()
        count+=batch_size
        accuary += get_accuary(result,label)

    print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}")
print("start testing..............")

```


### 测试
```python
model.eval()
count = 0
test_PATH = os.path.abspath('./dataset/qvc_data_test.txt')
test_data, test_label = get_data(test_PATH)
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
```

### 结果


![qvc_loss](qvc_loss.png)

![qvc_accuracy](qvc_accuracy.png)