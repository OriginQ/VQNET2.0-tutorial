

## Quantum K-means

聚类算法是一种典型的无监督学习算法，主要用于将相似的样本自动归为一类。聚类算法中，根据样本之间的相似性，将样本划分为不同的类别。对于不同的相似度计算方法，会得到不同的聚类结果。常用的相似度计算方法是欧氏距离法。我们要展示的是量子 K-Means 算法。 K-Means算法是一种基于距离的聚类算法，它以距离作为相似度的评价指标，即两个对象之间的距离越近，相似度越大。该算法认为簇是由相距很近的对象组成的，因此紧凑且独立的簇是最终目标。

#### 算法原理

量子K-Means算法的实现主要使用swap test来比较输入数据点之间的距离。 从 N 个数据点中随机选择 K 个点作为质心，测量每个点到每个质心的距离，并将其分配到最近的质心类，重新计算已经得到的每个类的质心，迭代 2 到 3 步，直到新的 质心等于或小于指定的阈值，算法结束。 在我们的示例中，我们选择了100 个数据点、2 个质心，并使用CSWAP电路来计算距离。 最后，我们获得了两个数据点集群。$|0\rangle$ 是辅助比特, 通过 $H$  逻辑门量子比特位将变为 $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$. 在量比特 $|1\rangle$的控制下, 量子线路将会翻转 $|x\rangle$ 和 $|y\rangle$​ . 最终得到结果:

​                                                                             $$|0_{anc}\rangle |x\rangle |y\rangle \rightarrow \frac{1}{2}|0_{anc}\rangle(|xy\rangle + |yx\rangle) + \frac{1}{2}|1_{anc}\rangle(|xy\rangle - |yx\rangle)$$

如果我们单独测量辅助量子比特，那么基态最终状态的概率 ![$|1\rangle$](https://render.githubusercontent.com/render/math?math=%7C1%5Crangle&mode=inline) 是:

​                                                                             $$P(|1_{anc}\rangle) = \frac{1}{2} - \frac{1}{2}|\langle x | y \rangle|^2$$​​​

两个量子态之间的欧几里得距离如下：

​                                                                            $$Euclidean \ distance = \sqrt{(2 - 2|\langle x | y \rangle|)}$$

可见测量量子比特位 $|1\rangle$​与欧几里得距离有正相关. 本算法的量子线路如下所述：.

![image-0](qkmeans_cir.png)

#### VQNet implementation

##### 1 generate data

我们使用scipy.make_blobs来产生随机的高斯分布数据.

```
def get_data(n, k, std):
    data = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=std, random_state=100)
    points = data[0]
    centers = data[1]
    return points, centers

```

#### 2 Quantum Circuits

我们如下代码使用VQNet构建线路

```

def get_theta(d):
    x = d[0]
    y = d[1]
    theta = 2 * math.acos((x.item() + y.item()) / 2.0)
    return theta

def qkemas_circuits(x, y):
    
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
    prog << pq.Measure(qubits[0], cbits[0])
    prog.insert(pq.Reset(qubits[0]))
    prog.insert(pq.Reset(qubits[1]))
    prog.insert(pq.Reset(qubits[2]))

    result = machine.run_with_configuration(prog, cbits, 1024)

    data = result

    if len(data) == 1:
        return 0.0
    else:
        return data['001'] / 1024.0
```

#### 3 cluster iteration

迭代多个epoch去找到最近邻。

```
def draw_plot(points, centers, label=True):
    points = np.array(points)
    centers = np.array(centers)
    if label==False:
        plt.scatter(points[:,0], points[:,1])
    else:
        plt.scatter(points[:,0], points[:,1], c=centers, cmap='viridis')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def initialize_centers(points,k):
    return points[np.random.randint(points.shape[0],size=k),:]


def find_nearest_neighbour(points, centroids):
    n = points.shape[0]
    k = centroids.shape[0]
    

    centers = zeros([n])

    for i in range(n):
        min_dis = 10000
        ind = 0
        for j in range(k):

            temp_dis = qkemas_circuits(points[i, :], centroids[j, :])

            if temp_dis < min_dis:
                min_dis = temp_dis
                ind = j
        centers[i] = ind

    return centers

def find_centroids(points, centers):


    k = int(tensor.max(centers).item()) + 1

    centroids = tensor.zeros([k, 2])

    for i in range(k):

        cur_i = centers == i
    
        x = points[:,0]
        x = x[cur_i]
        y = points[:,1]
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


def qkeman_run():
    n = 100  # number of data points
    k = 3  # Number of centers
    std = 2  # std of datapoints

    points, o_centers = get_data(n, k, std)  # dataset

    points = preprocess(points)  # Normalize dataset

    centroids = initialize_centers(points, k)  # Intialize centroids

    epoch = 5
    points = QTensor(points)
    centroids = QTensor(centroids)
    # run k-means algorithm
    for i in range(epoch):
            centers = find_nearest_neighbour(points, centroids)  # find nearest centers
            centroids = find_centroids(points, centers)  # find centroids
            plt.figure()
            draw_plot(points.data, centers.data)
			
```

#### 4 result

聚类前的原始数据。

![image-1](ep_1.png)

聚类后的数据，各个颜色为不同分类结果。

![image-9](ep_9.png)





