# cuda-practice

Implement alexnet from 0 to 1 as an practice.

## ENV

| device           | cuda |
| ---------------- | ---- |
| GeForce 3080 12G | 11.7 |

cuda 11.7

## pytorch Implement

```
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```

## 问题和解决方案

动态库循环依赖: 其中一个使用外部声明

纯cpp编译依赖.cu ，没有找到优雅的解决方法，只是将所有cpp改成.cu用ncc编译了

undefine vtable 虚析构函数子类未实现

### lunch kerbnel失败

kelementwise_inplace 在 tensor的update lunch 会输入地址内容都为0 而且不执行 kernel代码直接跳转到kernel结束，在其他文件（test，network里面lunch都正常）

可以独立运行的一段代码，在tensor.cu里面调用kernel失败，其他文件里面都可以正常调用！！

同一个文件，类外部的函数可以正常调用， 类的静态方法可以正常调用, 类对象调用静态方法，正常

另外一个普通方法lunch正常

正常：

    // std::vector<Tensor*> vt;

    // Tensor* a = new Tensor({1});

    // vt.push_back(a);

    // vt[0]->update_weights(1.f, cudaStreamDefault);

异常：

    Tensor* a = new Tensor({1});

    ComputeGraph test_graph;

    test_graph._weight_tensors.push_back(a);

    test_graph._weight_tensors[0]->update_weights(1.f, cudaStreamDefault);

```
void Tensor::update_weights(float alpha, cudaStream_t cudastream) {
 
        float wtf[4] = {5, 5, 4, 2};
        float *www;
        checkCudaErrors(cudaMalloc(&www, 16));
        checkCudaErrors(cudaMemcpy(www, wtf, 16, cudaMemcpyHostToDevice));
        kelementwise_inplace<<<1, 32>>>(
            4,
            www,
            1.f,
            www,
            ELE_OP::MULTIPLY
        );
        checkCudaErrors(cudaDeviceSynchronize());
        float *back_www = new float[4];
        checkCudaErrors(cudaMemcpy(back_www, www, 16, cudaMemcpyDeviceToHost));
        int a = 1;
```

    Tensor* a = new Tensor({1});

    a->update_weights(1.f, cudaStreamDefault);

    ComputeGraph test_graph;  // 没有这句话，上面一句执行结果正常，加上这句话，上面一句执行结果异常。真见了鬼了

最终定位原因为编译连接问题，多处有同一kernel调用就会调用失败，workaround 全部文件一起编译了

### lunch kernnel 随机 Segmentation fault

## diary

[ RUN      ] general/test_relu_float_1d_input.check_output_vs_cpu/2
0 th, host_output: 0.340188, fetch_output 0
/home/dongwei/Workspace/cuda-practice/v1/tests/test_operators/test_relu.cu:101: Failure
Expected equality of these values:
  host_output[i]
    Which is: 0.34018773
  fetch_output[i]
    Which is: 0
0 th, host_output: 0.34018772840499878, fetch_output 0

1 th, host_output: 0, fetch_output 0
2 th, host_output: 0.283099, fetch_output 0.283099
3 th, host_output: 0.29844, fetch_output 0.29844
4 th, host_output: 0.411647, fetch_output 0.411647
5 th, host_output: 0, fetch_output 0
6 th, host_output: 0, fetch_output 0
7 th, host_output: 0.26823, fetch_output 0.26823
8 th, host_output: 0, fetch_output 0
9 th, host_output: 0.05397, fetch_output 0.05397
10 th, host_output: 0, fetch_output 0
11 th, host_output: 0.128871, fetch_output 0.128871

第 0 个元素值始终为 0

### v1

将所有算子实现为 __global\_\_ 从主机顺序调用

### Implement Completely alone, check Accuracy and performance

1. Implement relu, test and optimize it

* [ ] 新知识

![DynamicParallelism](cuda-playground/multifile/DynamicParallelism.png "DynamicParallelism")

```
# --relocatable-device-code {true|false}          (-rdc)  
#         Enable (disable) the generation of relocatable device code.  If disabled,
#         executable device code is generated.  Relocatable device code must be linked
#         before it can be executed.
#         Default value:  false.
```

# to do

合并 desc 和 T* 为 tensor 类，连续非连续 version，存储位置管理。

位置（host  cuda）枚举改为 位置类，单例，运行时 自动检测可用设备形成设备列表，新增设备就不用改所有相关位置 malloc就不用if cuda if host

网络 parse 功能

智能指针

glog gflag

managedmemery 导致 cuda gdb 卡住？

验证 lunch config thread num > 1024 的话，getlaterror 能捕捉到么

多线程式：重复使用tensor可以避免重复malloc free，对于推理，这其实就是几个推理线程的问题，多个推理线程可以共享统一份权重，申请多份流动tensor每个线程一个 stream即可

流水线式：不合理的，想要流水线，layer之间需要queue缓冲，queue里面还是tensor，否则整个网络推理被最慢的layer限制。而且还要为每个layer做线程池，暂时认为是很不好的实现。如果是为了节省内存，所有tensor（weight，data stream）仅申请一份，最慢的层没有得到结果之前，前面的层也无法保存结果。使用两份buffer解决这个问题，锁很复杂。实现也很难。

network类保存：

网络结构，即layer构成的有向无环图。

还负责遍历此图：多余所有输出layer，连接一个dummy outputlayer，以此为root节点向前（pre layers）宽度优先遍历，push进一个先进后出的可遍历的结构，称作launch queue，之后构造的推理器可按照此顺序执行forward来launch kernel（在相应的cudastream)

构建网络： 确定layer结构，给layer IO tensor* 赋值（空）

创建 推理器，确定shape，按照launch queue顺序 shape infer，layer IO tensor 得到shape信息

推理器为（数据流tensor）申请空间

推理器执行推理

samart ptr !!!!

# v3

多卡

多输入多数出网络

内部带有分支的网络

支持可变网络，且性能不下降

推理加速：1. 多cuda stream并行，每次forward制定stream，需要stream pool？ 2. 流水线？

训练加速： 支持多batch，分布式训练？

oprator 其实就是 compute gragh，把oprator 抽象为  compute gragh，可以实现 计算图的 连接 运算

kernel 模板貌似没什么意义

network operator tensor 的 loation（host device）需要理清一遍

析构函数清理

# VLOG

# schedule

8.15 ~ 8.16 matmul kernel test

8.16 ~ 8.18 matmul net forward backward
