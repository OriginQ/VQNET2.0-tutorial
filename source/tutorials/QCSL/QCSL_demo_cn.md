## 													量子线路结构学习实例


## 概述

在量子线路结构中，最经常使用的带参数的量子门就是RZ、RY、RX门，但是在什么情况下使用什么门是一个十分值得研究的问题，一种方法就是随机选择，但是这种情况很有可能达不到最好的效果，Quantum circuit structure learning任务的核心目标就是找到**最优的带参量子门组合**。那用什么标准来衡量这个最优，这里的做法是**这一组最优的量子逻辑门要使得目标函数（loss function）取得最小值**。运行示例可见[qcsl_test.py](../../pyVQNet/examples/qcsl_test.py)

大致的步骤如下：

    1. 初始化量子线路和参数
    
    2. 对于每一个参数，线路通过最小化目标函数从RX, RY, RZ中选择最佳量子逻辑门
    
    3. 通过Rotoselect操作去更新量子门参数
    
    4. 重复1-3直到所有的参数都更新完毕

## 步骤

### 数据准备

构建QCSL线路，共有两个参数即$\theta_1$和$\theta_2$, $$U$$的结构共有三种可以选择即$$RX,RY,RZ$$，在参数更新过程中，**我们在某一时刻只更新其中一个**假如是$$\theta_1$$，另一个固定。接着对于逻辑门进行选择，原则是在该时刻只对$$\theta_1$$对应的门进行三选一操作，将此时选定的门、$$\theta_2$$对应的门以及$$\theta_1,\theta_2$$输入到我们的网络中，分别可以得到$$\theta_1$$对应的三种逻辑门下的目标函数值（这里设为$$corss = 0.5*Y+0.8*Z-0.2*X$$），**取其中最小值所对应的逻辑门**。同理，更新第二个参数，此时$$\theta_1$$与其对应的逻辑门已经确定。

```python
params = QTensor(np.array([0.3,0.25]))
params = params.data.getdata()
generator = ["X","Y"]
generators = copy.deepcopy(generator)
epoch = 20
```

## QCSL 线路结构

这里U为待搜索的量子逻辑门。

### 量子线路如下 

### ![quantum_circuit](quantum_circuit.png)

```python
def gen(param:CoreTensor,generators,qbits,circuit):
    if generators == "X":
        circuit.insert(pq.RX(qbits,param))
    elif generators =="Y":
        circuit.insert(pq.RY(qbits,param))
    else:
        circuit.insert(pq.RZ(qbits,param))
def circuits(params,generators,circuit):
    gen(params[0], generators[0], nqbits[0], circuit)
    gen(params[1], generators[1], nqbits[1], circuit)
    circuit.insert(pq.CNOT(nqbits[0], nqbits[1]))
    prog = pq.QProg()
    prog.insert(circuit)
    return prog

def ansatz1(params:QTensor,generators):
    circuit = pq.QCircuit()
    params = params.data.getdata()
    prog = circuits(params,generators,circuit)
    return expval(machine,prog,{"Z0":1},nqbits), expval(machine,prog,{"Y1":1},nqbits)

def ansatz2(params:QTensor,generators):
    circuit = pq.QCircuit()
    params = params.data.getdata()
    prog = circuits(params, generators, circuit)
    return expval(machine,prog,{"X0":1},nqbits)
```

### 损失函数

$corss = 0.5*Y + 0.8*Z - 0.2*X$

```python
def loss(params,generators):
    Z, Y = ansatz1(params,generators)
    X = ansatz2(params,generators)
    return 0.5 * Y + 0.8 * Z - 0.2 * X
```

### 更新参数

在更新的过程中，我们在某一时刻只更新其中的一个，即当更新 $$\theta_1 $$时， $$\theta_2 $$是固定的。之后从RX, RY, RZ之中挑选最佳的量子逻辑门

```python
def rotosolve(d, params, generators, cost, M_0):  # M_0 only calculated once
    params[d] = np.pi / 2.0
    M_0_plus = cost(QTensor(params), generators)
    params[d] = -np.pi / 2.0
    M_0_minus = cost(QTensor(params), generators)
    a = np.arctan2(
        2.0 * M_0 - M_0_plus - M_0_minus, M_0_plus - M_0_minus
    )  # returns value in (-pi,pi]
    params[d] = -np.pi / 2.0 - a
    if params[d] <= -np.pi:
        params[d] += 2 * np.pi
    return cost(QTensor(params), generators)

def optimal_theta_and_gen_helper(index,params,generators):
    params[index] = 0.
    M_0 = loss(QTensor(params),generators)#init value
    for kind in ["X","Y","Z"]:
        generators[index] = kind
        params_cost = rotosolve(index, params, generators, loss, M_0)
        if kind == "X" or params_cost <= params_opt_cost:
            params_opt_d = params[index]
            params_opt_cost = params_cost
            generators_opt_d = kind
    return params_opt_d, generators_opt_d


def rotoselect_cycle(params:np,generators):
    for index in range(params.shape[0]):
        params[index], generators[index] = optimal_theta_and_gen_helper(index,params,generators)
    return params,generators
```

### 主函数
```python
for i in range(epoch):
    state_save.append(loss(QTensor(params), generators))
    params, generators = rotoselect_cycle(params,generators)
```

### 结果

**搜索结果，本次选取的是RY,RZ门**

![final_quantum_circuit](final_quantum_circuit.png)

**loss下降**

![loss](loss.png)

**以结果中得到的网络模型为准（逻辑门固定）可以发现目标函数是关于$$\theta_1,\theta_2$$的函数，可以在（4，-3）附近取得局部极值点**

![loss3d](loss3d.png)    