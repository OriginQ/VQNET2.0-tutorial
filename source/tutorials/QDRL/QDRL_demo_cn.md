## 												量子数据重上传实例

## 概述
在神经网络中，每一个神经元都接受来自上层所有神经元的信息（图a）。与之相对的，单比特量子分类器接受上一个的信息处理单元和输入（图b）。通俗地来说，对于传统的量子线路来说，当数据上传完成，可以直接通过若干幺正变换 $$U(\theta_1,\theta_2,\theta_3)$$直接得到结果，但是在量子数据重上传（Quantum Data Re upLoading，QDRL）任务中，数据在幺正变换之前需要进行重新上传操作。具体例子可见[qdrl_test.py](../../pyVQNet/examples/qdrl_test.py)

### ![qdrl](qdrl.png)

大致流程如下:

    1.重上传: 创建一个单比特的量子线路，它可以实现若干幺正变换并且决定是否需要根据网络训练的参数去增加额外的量子逻辑门
    
    2.分类: Bloch球体表面有足够的空间，因此可以定义一组表示不同标签的目标状态，并以最大正交的方式选择线路最终的量子态

## 步骤

### 数据准备

###                                                                 ![points](points.png)

共有两类点**蓝色**记为(1,0)，**红色**记为(0,1)，以坐标（0，0）为圆心，以$$\sqrt{\frac{1}{2}}$$半径，在该圆内为蓝色，否则为红色

```python
def circle(samples:int,  reps =  np.sqrt(1/2)) :
    data_x, data_y = [], []
    for i in range(samples):
        x = np.random.rand(2)
        y = [0,1]
        if np.linalg.norm(x) < reps:
            y = [1,0]
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)
```



### QDRL 线路结构

Module 是autograd层的抽象类

```python
class Model(Module):

    def __init__(self):
        super().__init__()
        self.pqc = vmodel(params.shape)
        self.fc2 = Linear(2,2)
    def forward(self, x):
        x = self.pqc(x)
        return x
```
利用PyQpanda建立的参数化量子线路，其中参数的维度为4
```python
class vmodel(Module):
    def __init__(self,shape, num_layers = 3, q_delta = 1e-4):
        super().__init__()
        self.delta = q_delta
        self.num_layers = num_layers
        self.machine  = pq.CPUQVM()
        self.machine.init_qvm()
        self.n_qubits = self.machine.qAlloc_many(1)
        self.params = Parameter(num_layers*3) 
        self.last = []
  
    def forward(self,x):#x:QTensor
        self.last = []
        nodes = []
        requires_grad = (x.requires_grad or self.params.requires_grad) and not QTensor.NO_GRAD
        batch_save = np.zeros((x.data.shape[0],2))
        for i in range(x.data.shape[0]):
            xx = x.data.select([str(i),":"])
            prog = build_circult(self.params.data,xx, self.n_qubits)
            prob = self.machine.prob_run_dict(prog, self.n_qubits, -1)
            prob = list(prob.values())
            batch_save[i] = prob
            for ii in prob:
                self.last.append(ii)
        if x.requires_grad:
            nodes.append(QTensor.GraphNode(tensor=x,df=lambda g:1))
        if self.params.requires_grad:
            nodes.append(QTensor.GraphNode(tensor=self.params, df=lambda g:get_grad(g, x.data, self.params.data, build_circult,self.delta,self.machine,self.n_qubits,self.last)))
        
        return QTensor(data = batch_save,requires_grad = requires_grad,nodes=nodes)
def get_grad(g:CoreTensor, x:CoreTensor, params:CoreTensor, forward_circult,delta,machine,nqubits,last):
        num_params  = 1
        g = np.array(g)
        batch_size = x.shape[0]
        for _ in params.shape:
            num_params *= _
        grad = np.zeros(num_params)
        
        for i in range(num_params):
            iter = 0
            params_ = np.array(params)
            params_[i] += delta
            params_ = CoreTensor(params_)
            for tinybatch in range(batch_size):
                xx = x.select([str(tinybatch),":"])
                prog = forward_circult(params_,xx,nqubits)
                
                prob = machine.prob_run_dict(prog, nqubits, -1)
                prob = list(prob.values())
                for m in range(len(prob)):
                    grad[i] += g[tinybatch,m]* (prob[m]-last[iter])/delta
                    iter+=1
            grad[i] /= batch_size
        return CoreTensor(grad)

```
### 量子线路定义
量子线路的单量子比特上有6个Rot操作（由RX，RY，RZ组成），通过测量第一个量子比特以获得最终的概率值（两个）![qdrl_circuit](qdrl_circuit.png)
```python
def build_circult(param, x, n_qubits):#param 3*3, x=[1,2,3], x[i]   
    
        x1 = np.array(x).squeeze()
        param1 = np.array(param).squeeze()
        circult = pq.QCircuit()
        circult.insert(pq.RZ(n_qubits[0], x1[0]))
        circult.insert(pq.RY(n_qubits[0], x1[1]))
        circult.insert(pq.RZ(n_qubits[0], x1[2]))
        circult.insert(pq.RZ(n_qubits[0], param1[0]))
        circult.insert(pq.RY(n_qubits[0], param1[1]))
        circult.insert(pq.RZ(n_qubits[0], param1[2]))
        circult.insert(pq.RZ(n_qubits[0], x1[0]))
        circult.insert(pq.RY(n_qubits[0], x1[1]))
        circult.insert(pq.RZ(n_qubits[0], x1[2]))
        circult.insert(pq.RZ(n_qubits[0], param1[3]))
        circult.insert(pq.RY(n_qubits[0], param1[4]))
        circult.insert(pq.RZ(n_qubits[0], param1[5]))
        circult.insert(pq.RZ(n_qubits[0], x1[0]))
        circult.insert(pq.RY(n_qubits[0], x1[1]))
        circult.insert(pq.RZ(n_qubits[0], x1[2]))
        circult.insert(pq.RZ(n_qubits[0], param1[6]))
        circult.insert(pq.RY(n_qubits[0], param1[7]))
        circult.insert(pq.RZ(n_qubits[0], param1[8]))
        prog = pq.QProg()
        prog.insert(circult)

        return prog
```

### 优化器定义
本实例中使用的是sgd优化器，model.parameters()即需要更新的参数
```python
optimizer = sgd.SGD(model.parameters(),lr =1)
```

### 训练
    model contains quantum circuits or classic data layer 
    CategoricalCrossEntropy() is loss function
    backward() calculates model.parameters gradients 

```python
def train():
    model.train()
    x_train, y_train = circle(500)
    x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))  # 500*3
    x_train, y_train = QTensor(x_train),QTensor(y_train)
    epoch = 10
    print("start training...........")
    for i in range(epoch):
        accuracy = 0
        count = 0
        loss = 0
        for data, label in get_minibatch_data(x_train, y_train,batch_size):
            optimizer.zero_grad()
            data,label = QTensor(data), QTensor(label)
            output = model(data)
            Closs = CategoricalCrossEntropy()
            losss = Closs(label, output)

            losss.backward()
            optimizer._step()
            accuracy += get_score(output,label)
            
            loss += losss.item()
            print(f"epoch:{i}, train_accuracy:{accuracy}")
            print(f"epoch:{i}, train_loss:{losss.data.getdata()}")
            count += batch_size
            
        print(f"epoch:{i}, train_accuracy_for_each_batch:{accuracy/count}")
        print(f"epoch:{i}, train_loss_for_each_batch:{loss/count}")

```

### 测试
```python
def test():
    model.eval()
    print("start eval...................")
    x_test, y_test = circle(500)
    test_accuracy = 0
    count = 0
    x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))
    x_test, y_test = QTensor(x_test), QTensor(y_test)
    for test_data, test_label in get_minibatch_data(x_test,y_test, batch_size):

        test_data, test_label = QTensor(test_data),QTensor(test_label)
        output = model(test_data)
        test_accuracy += get_score(output, test_label)
        count += batch_size
    print(f"test_accuracy:{test_accuracy/count}")
```


### 结果

![qdrl_loss](qdrl_loss.png)
![qdrl_accuracy](qdrl_accuracy.png)
