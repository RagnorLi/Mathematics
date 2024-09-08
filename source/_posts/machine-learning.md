---
title: Hands On ML
date: 2024-08-08 21:51:44
label: 9月
background:
  bg-gradient-to-r from-pink-500 via-violet-400 to-blue-400 hover:from-pink-700 hover:via-violet-600 hover:to-blue-500
tags:
  - ai
plugins:
  - katex
  - copyCode
categories:
  - DataScience 
intro: |
    Hands On Machine Learning with Scikit-Learn Keras and TensorFlow
---



## Training Deep Neural Networks


### 训练DNN的挑战

#### 问题 1：训练深度神经网络有哪些主要挑战？

- **解答**：
  训练深度神经网络（DNN）相比于浅层网络具有更高的复杂性，主要挑战包括：
  - **梯度消失或爆炸**：当梯度在反向传播过程中变得越来越小或越来越大时，深层的网络难以更新权重，导致模型训练困难。
  - **数据不足**：大型网络需要大量的训练数据，然而在很多情况下数据不足，或者标注这些数据的成本很高。
  - **训练速度慢**：由于网络层数较多，参数量巨大，训练过程可能非常缓慢。
  - **过拟合风险高**：当模型参数量巨大时，如果训练数据不足或者数据质量不高，容易发生过拟合问题。

#### 问题 2：如何解决梯度消失或爆炸问题？

- **解答**：
  在后续章节，书中将介绍几种解决梯度消失或爆炸问题的技术：
  - 使用合适的激活函数，如ReLU（Rectified Linear Unit），来避免梯度消失问题。
  - 通过批量归一化（Batch Normalization）来加速训练并减轻梯度消失或爆炸的现象。
  - 改进权重初始化方法，确保初始梯度不会过小或过大。

#### 问题 3：如何处理训练数据不足的问题？

- **解答**：
  书中提到可以通过迁移学习（Transfer Learning）和无监督预训练（Unsupervised Pretraining）来应对数据不足的问题：
  - **迁移学习**：从预训练的模型中迁移知识，将在大规模数据上训练好的模型应用到新任务上，即使新任务的数据量较少，也能有效提高模型表现。
  - **无监督预训练**：通过无监督学习方法在未标注数据上训练模型，然后再在少量标注数据上进行微调。

#### 问题 4：如何加快深度神经网络的训练过程？

- **解答**：
  使用更高效的优化算法可以显著加快深度神经网络的训练。后续章节将讨论几种常用的优化器，包括：
  - **Adam** 和 **RMSProp** 等优化算法，这些优化器通过自适应学习率加速梯度下降过程。
  - 另外，通过并行化训练或使用专门的硬件（如GPU/TPU）可以进一步提升训练速度。

#### 问题 5：如何解决深度神经网络中的过拟合问题？

- **解答**：
  书中将介绍几种常用的正则化技术来防止模型过拟合：
  - **L2正则化**：通过在损失函数中加入权重的平方惩罚项，避免模型过拟合。
  - **Dropout**：在每次训练迭代时随机忽略一部分神经元，防止模型过度依赖某些特定的路径，从而提升模型的泛化能力。
  - **数据增强**：通过对训练数据进行旋转、缩放、裁剪等操作，生成新的训练样本，从而增强模型的泛化性能。

{.marker-round}

### 梯度消失跟梯度爆炸

#### 问题 1：什么是梯度消失和梯度爆炸问题？

- **解答**：
  梯度消失和梯度爆炸是深度神经网络中常见的两个问题，它们会导致网络难以有效训练。

  - **梯度消失（Vanishing Gradients）**：在反向传播过程中，随着梯度从输出层向输入层传播，梯度值会逐渐变小，导致较低层的权重几乎没有更新，模型无法有效学习。
  - **梯度爆炸（Exploding Gradients）**：在某些情况下，梯度值会随着传播逐层增大，导致网络权重更新时产生极大的值，最终使得训练过程发散。

#### 问题 2：梯度消失和爆炸问题的来源是什么？

- **解答**：
  梯度消失和爆炸的主要来源是：
  1. **激活函数**：例如Sigmoid或Tanh函数会导致梯度消失问题。当输入变得过大或过小时，激活函数的导数变得非常小，导致梯度几乎无法传递。
  2. **权重初始化不当**：如果网络权重初始化不合理，可能会导致梯度在传播过程中不断衰减或增大。

#### 问题 3：Sigmoid函数如何导致梯度消失问题？

- **解答**：
  在Sigmoid激活函数中，当输入值变得非常大（正或负）时，函数的输出会接近于0或1，而导数接近于0。这意味着在反向传播时，梯度几乎为0，导致下层网络几乎无法更新其权重。

  Sigmoid函数的表达式为：
  ```KaTeX
  \sigma(z) = \frac{1}{1 + e^{-z}}
  ```
  ![Figure11-1sigmoid激活函数饱和度.](../assets/attachment/hands_on_machine_learning/Figure11-1sigmoid激活函数饱和度.png)从图中（Figure 11-1）可以看出，当输入较大时（接近于-4或4），函数的输出趋于饱和（Saturating），而在接近于0的部分，函数呈现出线性区域。

  - 图示解释：
    - **Saturating 区域**：导数接近于0，梯度消失。
    - **Linear 区域**：导数值较大，梯度传递较好。

  （参考图名：**Figure 11-1: Sigmoid activation function saturation**）

#### 问题 4：如何缓解梯度消失和爆炸问题？

- **解答**：
  解决梯度消失和爆炸问题的方法包括：
  - **使用ReLU激活函数**：ReLU函数在大部分输入区间具有常数梯度，不会导致梯度消失问题。
  - **权重初始化技术**：可以使用更合理的权重初始化方法，如Xavier初始化或He初始化，确保在训练初期梯度值不会过大或过小。
  - **批量归一化（Batch Normalization）**：通过归一化每层的输入，使得每一层的输入保持在合理的范围内，防止梯度爆炸或消失。

{.marker-none}

###  Glorot&He初始化


#### 问题 1：什么是Glorot初始化？

- **解答**：
  Glorot初始化（也称为Xavier初始化）是Glorot和Bengio在2010年提出的一种初始化策略，旨在解决梯度消失和梯度爆炸问题。他们指出信号在前向传播和反向传播时，应该在每一层都能保持适当的大小，即输入和输出的方差应该相等，这样才能防止信号衰减或爆炸。

  Glorot初始化的公式如下：

  ```KaTeX
  \sigma^2 = \frac{1}{fan_{avg}}
  ```

  其中，`KaTeX:fan_{avg}` 是输入节点和输出节点的平均值，计算公式为：

  ```KaTeX
  fan_{avg} = \frac{fan_{in} + fan_{out}}{2}
  ```

  该方法使用均值为0的正态分布或均匀分布对权重进行初始化。

  - 如果使用均匀分布，权重取值范围为：`[-r, +r]`，其中 `r` 计算为：

  ```KaTeX
  r = 3 \sigma^2
  ```

  （参考 **Equation 11-1**: Glorot initialization）

#### 问题 2：什么是He初始化？

- **解答**：
  He初始化由Kaiming He等人提出，专为ReLU及其变体（如Leaky ReLU、ELU等）设计。He初始化确保信号在前向和反向传播中不会被放大或缩小。

  对于He初始化，权重的方差为：

  ```KaTeX
  \sigma^2 = \frac{2}{fan_{in}}
  ```

  He初始化通常使用正态分布，均值为0，方差为上述公式给定的值。

#### 问题 3：如何选择合适的权重初始化方法？

- **解答**：
  Glorot和He初始化策略的选择取决于你所使用的激活函数。下表给出了不同激活函数对应的初始化参数：

  **Table 11-1: Initialization parameters for each type of activation function**

| Initialization | Activation functions           | `KaTeX:\sigma^2` (Normal) |
|----------------|--------------------------------|---------------------------|
| Glorot         | None, tanh, sigmoid, softmax   | `KaTeX:1 / fan_{avg}`     |
| He             | ReLU, Leaky ReLU, ELU, GELU, Swish, Mish | `KaTeX:2 / fan_{in}`      |
| LeCun          | SELU                           | `KaTeX:1 / fan_{in} `     |

{.show-header .left-text}

如表所示，不同的初始化策略适用于不同的激活函数。Glorot初始化适合于Sigmoid和Tanh等函数，而He初始化适合于ReLU及其变体，LeCun初始化则主要用于SELU。

#### 问题 4：如何在Keras中使用Glorot和He初始化？

- **解答**：
  Keras默认使用Glorot初始化，若需要使用He初始化，可以通过以下代码设置 `kernel_initializer` 参数：

  ```python
  import tensorflow as tf
  dense = tf.keras.layers.Dense(50, activation="relu", kernel_initializer="he_normal")
  ```

  也可以使用VarianceScaling初始化器来自定义初始化策略。例如，使用基于 `fan_{avg}` 的He初始化，可以使用如下代码：

  ```python
  he_avg_init = tf.keras.initializers.VarianceScaling(scale=2., mode='fan_avg', distribution='uniform')
  dense = tf.keras.layers.Dense(50, activation="sigmoid", kernel_initializer=he_avg_init)
  ```

{.marker-none}

### 更好的激活函数

#### 问题 1：为什么ReLU是更好的激活函数选择？

- **解答**：
  Glorot和Bengio在2010年的论文中提出，不稳定梯度问题部分源于选择了不合适的激活函数。与Sigmoid相比，ReLU激活函数更适合深度神经网络，原因包括：
  - ReLU不会像Sigmoid那样在大输入值时饱和，从而避免梯度消失问题。
  - ReLU计算非常快，适合深层网络的训练。

  然而，ReLU也存在**dying ReLUs**问题：在训练过程中，一些神经元会“死亡”，即这些神经元输出恒为0，这种现象尤其在学习率过大时容易发生。

#### 问题 2：如何解决dying ReLUs问题？

- **解答**：
  Leaky ReLU是ReLU的一个变体，用于解决dying ReLUs问题。Leaky ReLU定义如下：

  ```KaTeX
  LeakyReLU_{\alpha}(z) = \max(\alpha z, z)
  ```

  其中，`KaTeX:\alpha` 是一个超参数，用来控制负值区域的斜率。通过让负值部分有一个小斜率（例如 `KaTeX:\alpha = 0.2`），避免神经元彻底“死亡”。

  - ![Figure11-2LeakyReLU的曲线](../assets/attachment/hands_on_machine_learning/Figure11-2LeakyReLU的曲线.png)**Figure 11-2** 显示了Leaky ReLU的曲线，其中可以看到负数部分仍有一个“漏出”值。

#### 问题 3：除了Leaky ReLU，还有哪些ReLU的变体？

- **解答**：
  除了Leaky ReLU外，还有其他的ReLU变体，包括：
  - **RReLU（Randomized Leaky ReLU）**：在训练过程中，`KaTeX:\alpha` 是随机选择的，但在测试时固定为某个均值。
  - **PReLU（Parametric Leaky ReLU）**：`KaTeX:\alpha` 是一个可学习的参数，而不是超参数，训练过程中通过反向传播优化。

#### 问题 4：ELU和SELU激活函数是什么？

- **解答**：
  ELU（Exponential Linear Unit）是另一个ReLU的变体，定义如下：

  ```KaTeX
  ELU_{\alpha}(z) = \alpha (e^{z} - 1) \text{ if } z < 0, \text{ else } z
  ```

  ELU的特性包括：
  - 对于负值输入，ELU接近于零，从而帮助解决梯度消失问题。
  - 当  `KaTeX:\alpha = 1`  时，ELU在零附近是光滑的，这有助于加速梯度下降。

  SELU（Scaled Exponential Linear Unit）是ELU的缩放版本，其定义类似于ELU，但带有一个额外的缩放因子，主要用于使得神经网络具有自归一化（self-normalizing）特性。![Figure11-3ELU和SELU的函数曲线](../assets/attachment/hands_on_machine_learning/Figure11-3ELU和SELU的函数曲线.png)**Figure 11-3** 显示了ELU和SELU的函数曲线。

#### 问题 5：如何在Keras中使用Leaky ReLU、PReLU、ELU和SELU？

- **解答**：
  在Keras中，可以直接使用这些激活函数。例如，使用Leaky ReLU的代码如下：

  ```python
  leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
  dense = tf.keras.layers.Dense(50, activation=leaky_relu, kernel_initializer="he_normal")
  ```

  对于ELU和SELU，可以如下使用：

  ```python
  dense_elu = tf.keras.layers.Dense(50, activation="elu")
  dense_selu = tf.keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal")
  ```

#### 问题 6：什么是GELU、Swish和Mish激活函数？

- **解答**：
  GELU（Gaussian Error Linear Unit）是一种平滑的ReLU变体，其定义为：

  ```KaTeX
  GELU(z) = z \Phi(z)
  ```

  其中 `KaTeX:\Phi(z)` 是标准高斯分布的累积分布函数（CDF）。

  - ![Figure11-4GELU和Swish和Mish的函数曲线](../assets/attachment/hands_on_machine_learning/Figure11-4GELU和Swish和Mish的函数曲线.png)**Figure 11-4** 显示了GELU、Swish和Mish的函数曲线。Swish函数被定义为：

  ```KaTeX
  Swish(z) = z \sigma(\beta z)
  ```

  Mish函数被定义为：

  ```KaTeX
  Mish(z) = z \tanh(\log(1 + e^z))
  ```

  GELU、Swish和Mish都比传统的ReLU激活函数表现更好，尤其在处理复杂任务时。

{.marker-none}

### 批量正则化BN

Batch Normalization（BN）由Sergey Ioffe和Christian Szegedy在2015年提出，目的是解决深度神经网络中的梯度消失和梯度爆炸问题。BN通过对每一层的输入进行归一化，并应用缩放和偏移，能够加速模型训练并提升性能。

#### 问题 1：为什么需要Batch Normalization？

- **解答**：
  即使使用了He初始化和ReLU激活函数，深度神经网络中仍可能出现梯度消失和梯度爆炸问题。随着网络深度的增加，层间输入的分布会发生变化，使得训练变得不稳定。Batch Normalization通过对每一层的输入进行标准化，确保每一层的输入均值为0，标准差为1，从而使训练更为稳定。此外，BN还允许使用更大的学习率，从而加速收敛。

#### 问题 2：Batch Normalization是如何工作的？

- **解答**：
  BN的核心思想是在每个mini-batch中计算输入的均值和标准差，对输入进行归一化后，再进行缩放和偏移操作。具体步骤如下：

  **1. 计算mini-batch均值**：
  ```KaTeX
  \mu_B = \frac{1}{m_B} \sum_{i=1}^{m_B} x^{(i)}
  ```
  其中，`KaTeX:\mu_B` 是mini-batch中每个特征的均值。

  **2. 计算mini-batch标准差**：
  ```KaTeX
  \sigma_B^2 = \frac{1}{m_B} \sum_{i=1}^{m_B} \left( x^{(i)} - \mu_B \right)^2
  ```

  **3. 对输入进行归一化**：
  ```KaTeX
  \hat{x}^{(i)} = \frac{x^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
  ```
  其中，`KaTeX:\epsilon` 是一个小常数，用于防止除以零。

  **4. 应用缩放和偏移**：
  ```KaTeX
  z^{(i)} = \gamma \hat{x}^{(i)} + \beta
  ```
  - `KaTeX:\gamma` 是缩放参数，用来恢复网络的表达能力。
  - `KaTeX:\beta` 是偏移参数，用来保证网络具有足够的灵活性。

#### 问题 3：Batch Normalization在训练和测试中的行为有何不同？

- **解答**：
  - **训练时**：BN会基于当前mini-batch的均值和标准差进行归一化操作。因此，在训练时，BN依赖于mini-batch的统计数据。
  - **测试时**：由于测试时通常输入是单个样本或批量较小，BN不能再依赖mini-batch的统计数据。此时，BN使用训练过程中累积的均值和标准差（通过移动平均的方式计算）来进行归一化，确保推理阶段的稳定性。

#### 问题 4：Batch Normalization有哪些优点？

- **解答**：
  - **加速训练**：BN减少了梯度消失和梯度爆炸的现象，使得网络训练更加稳定，从而可以使用更大的学习率，加快训练速度。
  - **减少对权重初始化的依赖**：由于BN会动态地调整输入，网络对权重初始化不再敏感。
  - **正则化效果**：BN有一定的正则化作用，因为每个mini-batch都引入了随机噪声，类似于Dropout的效果，从而减少了过拟合的风险。

#### 问题 5：Batch Normalization的不足之处是什么？

- **解答**：
  - **计算开销增加**：BN需要在每个mini-batch中计算均值和标准差，并引入额外的缩放和偏移参数，这会增加训练和推理阶段的计算开销。
  - **对mini-batch大小的敏感性**：BN的效果依赖于mini-batch的大小。如果mini-batch过小，统计数据的波动会增加，导致模型性能不稳定。

#### 问题 6：如何在Keras中使用Batch Normalization？

- **解答**：
  在Keras中，可以通过在模型中添加`BatchNormalization`层来使用BN。通常将BN层放在激活函数之前或之后使用。以下是一个示例代码：

  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=[28, 28]),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(10, activation="softmax")
  ])
  ```

#### 问题 7：Batch Normalization的算法和参数解析

- **解答**：
  在每个BN层中，会为每个输入特征增加4个参数：`KaTeX:\gamma`（缩放参数）、`KaTeX:\beta`（偏移参数）、`KaTeX:\mu`（移动平均的均值）和`KaTeX:\sigma`（移动平均的标准差）。

  例如，假设模型的输入层有784个输入特征：
  - 对于该层，BN会增加`KaTeX: 4 \times 784 = 3136`个参数。

  模型的具体参数量总结如下：

  ```plaintext
  Layer (type)                Output Shape              Param #
  =================================================================
  flatten (Flatten)           (None, 784)               0
  batch_normalization (BatchNo (None, 784)              3136
  dense (Dense)               (None, 300)               235500
  batch_normalization_1 (Batch (None, 300)              1200
  dense_1 (Dense)             (None, 100)               30100
  batch_normalization_2 (Batch (None, 100)              400
  dense_2 (Dense)             (None, 10)                1010
  =================================================================
  Total params: 271,346
  Trainable params: 268,978
  Non-trainable params: 2,368
  ```

  其中`KaTeX: 2,368`个非训练参数是均值和标准差的移动平均，不通过反向传播更新。

#### 问题 8：如何调整BN的超参数？

- **解答**：
  Batch Normalization的主要超参数包括：
  - **Momentum**：决定均值和标准差的移动平均的更新速率，通常接近1，例如0.9、0.99或0.999。
  - **Axis**：决定归一化的轴，默认情况下为-1，即最后一个轴。

  在训练过程中，BN层会随着每个batch更新均值和标准差。Keras会自动使用移动平均来保存这些统计数据，确保在推理阶段使用稳定的统计信息。

#### 问题 9：如何优化BN在推理时的性能？

- **解答**：
  在推理阶段，BN的额外计算会带来一些开销。这些开销可以通过在训练后将BN层与前一层的权重合并来优化。具体做法是将BN的缩放参数`KaTeX:\gamma`和偏移参数`KaTeX:\beta`直接与前一层的权重和偏差相结合，形成新的权重`KaTeX:W'`和偏差`KaTeX:b'`。

  优化公式如下：

  ```KaTeX
  W' = \frac{\gamma W}{\sigma}
  ```

  ```KaTeX
  b' = \frac{\gamma (b - \mu)}{\sigma} + \beta
  ```

  这种优化可以显著减少推理阶段的计算开销。

{.marker-none}

### 梯度裁剪Gradient Clipping

#### 问题 1：什么是梯度裁剪（Gradient Clipping）？

- **解答**：
  梯度裁剪（Gradient Clipping）是一种缓解梯度爆炸问题的技术。在反向传播过程中，若梯度值变得过大，可能会导致权重更新不稳定，甚至训练发散。梯度裁剪通过将梯度限制在一个预设的阈值之内，确保梯度不会超过该阈值，从而防止梯度爆炸。

#### 问题 2：梯度裁剪在什么情况下使用？

- **解答**：
  梯度裁剪通常在处理**递归神经网络（RNN）**时使用，因为RNN容易遭遇梯度爆炸问题。在某些情况下，Batch Normalization可能并不适合使用，而梯度裁剪是一种简单且有效的替代方案。

#### 问题 3：如何在Keras中实现梯度裁剪？

- **解答**：
  在Keras中，梯度裁剪的实现非常简单。只需在创建优化器时设置`clipvalue`或`clipnorm`参数。例如：

  - **按值裁剪**：将梯度的每个分量限制在`[-1.0, 1.0]`的范围内。
    ```python
    optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
    model.compile(optimizer=optimizer)
    ```

  - **按范数裁剪**：将整个梯度的`L2`范数限制在给定的阈值内，例如1.0。
    ```python
    optimizer = tf.keras.optimizers.SGD(clipnorm=1.0)
    model.compile(optimizer=optimizer)
    ```

#### 问题 4：梯度裁剪的效果是什么？

- **解答**：
  - **按值裁剪**：会将梯度向量的每个分量限制在某个范围内，例如`[-1.0, 1.0]`。这意味着所有梯度分量的绝对值都不会超过该阈值。但这种方式可能会改变梯度向量的方向。

    **举例**：
    如果原始梯度向量为`[0.9, 100.0]`，经过按值裁剪后，它将变为`[0.9, 1.0]`，这改变了梯度向量的方向。

  - **按范数裁剪**：这种方法会将整个梯度向量的`L2`范数限制在一个阈值内（例如1.0）。与按值裁剪不同，按范数裁剪不会改变梯度向量的方向，只是缩放它的长度。

    **举例**：
    如果梯度向量为`[0.9, 100.0]`，在`clipnorm=1.0`的情况下，经过裁剪后它会变为`[0.00899964, 0.9999595]`，其方向保持不变，但长度被缩小。

#### 问题 5：如何选择合适的梯度裁剪策略？

- **解答**：
  梯度裁剪的阈值是一个可调的超参数，选择合适的值需要根据具体任务进行实验调整。如果梯度爆炸问题非常严重，可以尝试更严格的裁剪策略（例如较小的阈值）。此外，按值裁剪和按范数裁剪的效果有所不同，应该分别进行测试，选择在验证集上表现最佳的方法。

#### 问题 6：如何监控梯度大小？

- **解答**：
  在使用梯度裁剪时，可以通过工具如**TensorBoard**监控梯度的大小。如果在训练过程中发现梯度爆炸，可以尝试启用梯度裁剪并调整裁剪的阈值。此外，通过可视化梯度的变化，可以帮助理解模型的训练动态，并优化梯度裁剪的使用。

{.marker-none}

### 重用预训练层


#### 问题 1：为什么要重用预训练层？

- **解答**：
  从零开始训练一个非常大的深度神经网络（DNN）往往是不太现实的，因为这需要大量的数据和计算资源。相反，如果能够找到一个已经解决了类似问题的预训练神经网络，可以大大减少训练时间，并降低所需的训练数据量。这种方法被称为**迁移学习（transfer learning）**。

  - **迁移学习的优势**：
    1. 大幅减少训练时间。
    2. 降低对大量标记数据的需求。

#### 问题 2：如何重用预训练的层？

- **解答**：
  设想你有一个DNN，它被训练来对100种类别（如动物、植物、车辆等）进行分类，现在你想要训练一个DNN来专门分类某些特定类型的车辆。这些任务非常相似，因此你可以尝试重用原网络的大部分层，只需替换掉一些较高层的神经元即可，![Figure11-5重用预训练层](../assets/attachment/hands_on_machine_learning/Figure11-5重用预训练层.png)如下图所示（**Figure 11-5**）：

  - **图解**：在图中，可以看到如何从一个已经训练好的任务A的神经网络中重用较低层的隐藏层（如Hidden 1和Hidden 2）到新任务B中。这些层的权重会被冻结，防止在训练过程中被修改。同时，较高层的权重（如Hidden 3和Hidden 4）保持可训练的状态，以便适应新任务。

#### 问题 3：如何确定哪些层可以重用？

- **解答**：
  - **输出层**：通常应替换，因为它与新任务的输出类别不一致（例如，原任务可能有100个类别，而新任务可能只有10个类别）。
  - **较高层隐藏层**：这些层的高层特征对原任务更有用，而对新任务可能不太相关。因此，最好重新训练这些层，或者删除它们。
  - **较低层隐藏层**：低层特征通常对各种任务都有用（例如边缘、纹理等低级视觉特征），因此更适合重用。

  **TIP**：任务越相似，重用的层数就越多。对于非常相似的任务，可以尝试保留所有隐藏层，只替换输出层。

#### 问题 4：如何冻结和解冻预训练层？

- **解答**：
  1. **初始冻结所有重用层**：首先，将所有重用的层设置为不可训练（即权重保持固定）。然后，训练模型并观察其表现。
  2. **逐渐解冻部分高层**：如果表现不佳，尝试解冻顶部隐藏层的1-2层，让反向传播算法对其进行微调，看看性能是否改善。重用的层越多，所需的训练数据越少。
  3. **调整学习率**：当你解冻重用的层时，降低学习率可以避免破坏这些已经调整得较好的权重。

  **TIP**：如果你有足够的训练数据，可以尝试替换掉顶部的隐藏层，甚至添加更多的新隐藏层。

#### 问题 5：重用预训练层的最佳策略是什么？

- **解答**：
  - 如果你有大量的训练数据，可以更多地解冻高层隐藏层，甚至替换掉它们。
  - 如果训练数据较少，可以冻结较低层，并且只解冻或重新训练较少的高层隐藏层。
  - 可以不断试验不同的冻结和解冻策略，直到找到最适合的层数。

{.marker-none}

### 使用Keras进行迁移学习

#### 问题 1：如何使用Keras进行迁移学习？

- **解答**：
  在迁移学习中，我们会重用一个预训练模型的部分层，尤其是低层的特征提取器。这些层通常对各种任务都有用，而高层的特征往往与特定任务紧密相关，因此通常需要替换高层或输出层。

  举例说明：假设你有一个Fashion MNIST数据集，它包含除凉鞋和衬衫外的所有类目。你训练了一个Keras模型`model_A`，它在分类任务上达到了90%以上的准确率。现在，你有一个任务，想要基于这个模型，训练一个二分类器，用来区分T恤和凉鞋。

  你可以直接重用`model_A`的所有层，除了输出层。以下代码展示了如何在Keras中实现这一操作：

  ```python
  model_A = tf.keras.models.load_model("my_model_A")
  model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])  # Reuse all layers except the last one
  model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # Add new output layer
  ```

  **注意**：`model_A`和`model_B_on_A`现在共享了一些层。这意味着训练`model_B_on_A`时，`model_A`的共享层也会被更新。如果不希望出现这种情况，需要先克隆`model_A`，再使用其权重。

#### 问题 2：如何克隆预训练模型并使用其权重？

- **解答**：
  如果希望在迁移学习中避免修改原始模型的权重，可以通过`clone_model()`函数克隆模型架构，并手动复制权重：

  ```python
  model_A_clone = tf.keras.models.clone_model(model_A)
  model_A_clone.set_weights(model_A.get_weights())
  ```

  **Warning**：`clone_model()`只克隆模型的架构，而不会复制权重，必须使用`set_weights()`手动设置权重。

#### 问题 3：迁移学习过程中如何处理随机初始化的输出层？

- **解答**：
  新添加的输出层由于权重是随机初始化的，可能会在训练初期产生较大的误差梯度，从而破坏共享的层。为避免这种情况，可以先冻结所有重用的层，训练几个epoch让新层学习合理的权重。可以通过以下方式冻结层并编译模型：

  ```python
  for layer in model_B_on_A.layers[:-1]:
      layer.trainable = False

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
  model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
  ```

  **注意**：每次在冻结或解冻层后，都需要重新编译模型。

#### 问题 4：如何解冻层并继续训练模型？

- **解答**：
  在训练了几个epoch后，可以解冻部分或所有重用的层，并通过重新编译模型后继续训练。这是为了微调模型的共享层，以适应新任务。为了防止损坏重用层的权重，建议降低学习率。代码如下：

  ```python
  for layer in model_B_on_A.layers[:-1]:
      layer.trainable = True

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
  model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

  history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))
  ```

#### 问题 5：迁移学习的最终效果如何？

- **解答**：
  在这个示例中，模型`B_on_A`通过迁移学习，最终测试准确率从91.85%提升到了93.85%，误差率减少了接近25%。

  ```plaintext
  >>> model_B_on_A.evaluate(X_test_B, y_test_B)
  [0.2546142041683197, 0.938499871253967]
  ```

#### 问题 6：迁移学习的局限性是什么？

- **解答**：
  虽然迁移学习在某些任务中效果显著，但它并不总是有效。特别是在小型稠密网络中，迁移学习的效果并不明显，因为这些网络通常学到的是特定任务的模式，难以在其他任务中复用。相反，迁移学习更适合深度卷积神经网络（CNN），因为这些网络倾向于学习更加通用的特征，尤其是在低层。

{.marker-none}

### 无监督预训练器


#### 问题 1：在缺少标注数据时，该如何训练复杂的深度神经网络？

- **解答**：
  当你面临复杂任务且标注数据有限时，可以使用无监督预训练（unsupervised pretraining）来解决这个问题。无监督学习通过未标注的数据进行训练，然后通过少量的标注数据进行微调。具体方法可以使用**自编码器（autoencoder）**或**生成对抗网络（GAN）**，它们是常用的无监督学习方法。

  **Figure 11-6**展示了无监督预训练的过程：![Figure11-6无监督预训练的过程示意图](../assets/attachment/hands_on_machine_learning/Figure11-6无监督预训练的过程示意图.png)
  - 左侧部分，模型通过未标注数据进行无监督训练。
  - 右侧部分，使用少量标注数据对模型进行微调，以适应特定任务。

#### 问题 2：如何利用无监督预训练进行迁移学习？

- **解答**：
  你可以通过自编码器或GAN的较低层（例如GAN的判别器部分），将其训练为无监督模型，并通过未标注数据来初始化这些层的权重。然后你可以在这些基础上构建新的网络，在高层添加输出层并使用少量的标注数据进行监督学习，来微调整个网络。这个流程如下：
  1. 使用未标注的数据，训练自编码器或GAN。
  2. 冻结低层权重，添加新的输出层。
  3. 使用有标注的数据，微调整个模型。

#### 问题 3：无监督预训练的历史和发展？

- **解答**：
  无监督预训练是深度学习复兴的关键因素之一。2006年，Geoffrey Hinton及其团队首次提出了这种技术，最初的预训练方法通常使用受限玻尔兹曼机（RBM）进行逐层无监督预训练（称为“greedy layer-wise pretraining”）。在那个时期，训练深度神经网络非常困难，特别是梯度消失问题。通过逐层无监督预训练，可以有效解决这些问题。

  - **早期方法**：训练单层的RBM，然后冻结该层，再在其上训练另一层，以此类推。
  - **现今方法**：使用自编码器或GAN替代RBM，并且通常在一轮训练中完成全部无监督模型的训练，而不再需要逐层预训练。

#### 问题 4：什么是贪心逐层预训练（greedy layer-wise pretraining）？

- **解答**：
  在深度学习早期，由于难以训练深度网络，人们会使用贪心逐层预训练的方法。该方法的流程如下：
  - 训练单层的无监督模型（例如RBM）。
  - 冻结该层，添加新的层并再次训练（只训练新层）。
  - 重复上述步骤，直到网络深度足够。

  这种方法当时非常有效，直到梯度消失问题逐步得到解决，如今，较少使用贪心逐层预训练，而是使用更先进的自编码器或GAN一次性训练整个模型。

#### 问题 5：无监督预训练的现状如何？

- **解答**：
  如今，无监督预训练依然是一种有效的技术，尤其是在标注数据稀缺的情况下。当没有类似任务的预训练模型可以重用时，且有大量未标注数据时，无监督预训练是一种良好的选择。相比于早期的RBM，现在更多使用自编码器或GAN进行无监督预训练。自编码器可以从未标注数据中提取有用的特征，而GAN的判别器部分也可以用来提取低层特征。

{.marker-none}

### 辅助任务上的预训练

#### 问题 1：什么是“辅助任务上的预训练”？

- **解答**：
  当标注数据不足时，辅助任务上的预训练（pretraining on an auxiliary task）是一种常见的解决方案。你可以先在一个辅助任务上训练第一个神经网络，这个任务应该可以方便地获取或生成标注数据。训练好第一个神经网络后，你可以重用其低层的特征检测器，将这些特征用于你的实际任务。

  **关键点**：
  - 第一个神经网络会学习低层的特征检测器，这些特征在第二个神经网络中仍然是有用的。
  - 这种方法有效减少了对实际任务的标注数据需求。

#### 问题 2：如何在图像识别中应用辅助任务上的预训练？

- **解答**：
  假设你想要构建一个面部识别系统，但你仅有少量的图片，无法直接训练出一个高质量的分类器。一个实际的做法是：
  - 你可以从网上收集大量随机人的图片，训练一个网络来检测图像中是否存在相同的两个人。这种网络会学到良好的面部特征检测器。
  - 然后，重用这些低层特征检测器，可以训练一个面部识别分类器，即使你只有少量的标注数据。

#### 问题 3：如何在自然语言处理（NLP）中应用辅助任务上的预训练？

- **解答**：
  在自然语言处理任务中，可以通过下载大量的文本语料库，并自动生成标注数据。一个经典的例子是**掩码语言模型（masked language model）**：
  - 你可以随机屏蔽掉一些词汇，训练一个模型来预测被屏蔽的词汇。例如，在句子“What ___ you saying?”中，模型应该预测出“are”或“were”。
  - 当模型在这种任务上取得良好表现后，它已经学习了很多关于语言的特征。你可以将这个模型用于你的实际任务，并使用少量标注数据进行微调。

  这种自动生成标注数据并进行训练的方法称为**自监督学习（self-supervised learning）**。

#### NOTE：
**自监督学习（Self-supervised learning）** 是指从数据本身自动生成标签，然后在生成的“标注”数据集上使用监督学习技术进行训练。

### 加速神经网络训练


#### 问题 1：如何加速深度神经网络的训练？

- **解答**：
  训练非常大的深度神经网络可能会非常缓慢。在之前的章节中，我们已经讨论了四种加速训练的方法：
  - 应用一个良好的**权重初始化策略**。
  - 使用一个**有效的激活函数**。
  - 使用**批量归一化**（Batch Normalization）。
  - 重用部分预训练网络的层，例如从辅助任务或无监督学习模型中提取的层。

  除了这些方法，另一个巨大的加速来源是使用比常规**梯度下降优化器**更快的优化器。在这一节中，我们将介绍一些最流行的优化算法，包括：
  - **动量优化器（Momentum Optimizer）**
  - **Nesterov 加速梯度（Nesterov Accelerated Gradient）**
  - **AdaGrad**
  - **RMSProp**
  - **Adam 及其变种**

这些优化器大多是通过对梯度下降进行改进来加速收敛，并且在处理复杂的神经网络时表现良好。在接下来的部分中，我们将详细讨论每种优化器的原理和实现方法。

{.marker-none}

### 动量优化Momentum

#### 问题 1：什么是动量优化（Momentum Optimization）？

- **解答**：
  动量优化的核心思想是通过使用“动量”概念来加速梯度下降。可以将动量优化想象为一个球在光滑的斜坡上滚动。起初，球滚动得很慢，但随着时间推移，它逐渐加速，直到达到终端速度。动量优化通过记住之前的梯度，使梯度下降不仅依赖当前的梯度更新，还考虑之前的梯度，这样可以加速收敛速度，尤其是在缓慢变化的情况下。

#### 问题 2：动量优化与普通梯度下降有何不同？

- **解答**：
  - 在普通的梯度下降中，每次迭代中参数的更新只取决于当前的梯度，它忽略了之前的更新历史。因此，当局部梯度较小时，更新的步长会非常小，导致训练非常缓慢。
  - 动量优化则通过引入一个“动量向量”来记住之前的更新历史，在每次迭代时，它会将当前的梯度与之前的动量向量相加，从而使更新过程更加平滑和快速。

#### 问题 3：动量优化的数学表达是什么？

- **解答**：
  动量优化的数学公式如下：

  ```KaTeX
  \begin{aligned}
  1. & \quad \mathbf{m} \leftarrow \beta \mathbf{m} - \eta \nabla_{\mathbf{\theta}} J(\mathbf{\theta}) \\
  2. & \quad \mathbf{\theta} \leftarrow \mathbf{\theta} + \mathbf{m}
  \end{aligned}
  ```

  其中：
  - `KaTeX:\mathbf{m}` 表示动量向量。
  - `KaTeX:\mathbf{\theta}` 是网络的权重。
  - `KaTeX:\nabla_{\mathbf{\theta}} J(\mathbf{\theta})` 是当前的梯度。
  - `KaTeX:\eta` 是学习率。
  - `KaTeX:\beta` 是动量超参数，常用值为0.9。

  第一步通过引入动量向量`KaTeX:\mathbf{m}`，累积之前的梯度，第二步使用累积的动量更新权重。

#### 问题 4：动量优化为什么更快？

- **解答**：
  动量优化可以让训练加速，尤其是当代价函数的形状像一个细长的碗（例如输入具有不同的尺度时）。普通梯度下降会在陡峭的方向上快速下降，但在较平缓的方向上更新缓慢，导致训练花费很长时间。而动量优化可以加速在平缓方向上的更新，使模型更快地到达最优点。

  **图解**：普通梯度下降像是滚动到山谷底部的过程中速度很慢，而动量优化则像是滚下山谷时速度越来越快。

#### 问题 5：动量优化的实现？

- **解答**：
  在Keras中实现动量优化非常简单，只需在SGD优化器中设置动量超参数，如下：

  ```python
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
  ```

  通过这个设置，动量优化可以显著加快训练过程，并且动量值为0.9通常是一个合理的选择。

#### 问题 6：动量优化的一个可能问题是什么？

- **解答**：
  动量优化的一个潜在问题是由于动量的存在，优化器可能会在接近最优点时出现**过冲现象**（overshooting），即超出目标值，然后返回，并可能反复振荡。尽管如此，动量优化依然通常比普通梯度下降要快很多。


{.marker-none}

### Nesterov动量优化

#### 问题 1：什么是 Nesterov Accelerated Gradient (NAG)？

- **解答**：
  Nesterov Accelerated Gradient (NAG)，也称为 **Nesterov 动量优化**，是动量优化的一种变体。由 **Yurii Nesterov** 在 1983 年提出，NAG 通常比常规动量优化更快。NAG 的核心思想是提前沿动量方向计算梯度，而不是在当前位置计算梯度。通过这种方法，可以获得更精确的梯度更新，从而加速收敛。

  NAG 的梯度更新公式如下：

  ```KaTeX
  \begin{aligned}
  1. & \quad \mathbf{m} \leftarrow \beta \mathbf{m} - \eta \nabla_{\mathbf{\theta}} J(\mathbf{\theta} + \beta \mathbf{m}) \\
  2. & \quad \mathbf{\theta} \leftarrow \mathbf{\theta} + \mathbf{m}
  \end{aligned}
  ```

  在这个算法中：
  - `KaTeX:\mathbf{m}` 是动量向量。
  - `KaTeX:\mathbf{\theta}` 是模型权重。
  - `KaTeX:\nabla_{\mathbf{\theta}} J(\mathbf{\theta} + \beta \mathbf{m})` 是在 `KaTeX:\mathbf{\theta} + \beta \mathbf{m}` 位置计算的梯度。
  - `KaTeX:\eta` 是学习率。
  - `KaTeX:\beta` 是动量系数，通常取值为 0.9。

#### 问题 2：NAG 与普通动量优化有何不同？

- **解答**：
  - **普通动量优化**：在当前位置 `KaTeX:\mathbf{\theta}` 计算梯度并更新权重。
  - **NAG**：在提前一步的位置 `KaTeX:\mathbf{\theta} + \beta \mathbf{m}` 计算梯度，从而更加准确地更新权重。

  通过在动量方向上提前一步，NAG 可以更精确地调整参数，避免局部震荡，加快收敛速度。

  **图解**：![Figure11-7NAG更新对比动量更新示意图](../assets/attachment/hands_on_machine_learning/Figure11-7NAG更新对比动量更新示意图.png)如 **Figure 11-7** 所示，NAG 的更新最终更接近最优点（optimum），且随着时间推移，这种提前计算带来的微小改进会累积，最终使得 NAG 比普通动量优化更快。

#### 问题 3：如何在 Keras 中实现 NAG？

- **解答**：
  在 Keras 中实现 NAG 非常简单，只需要在 SGD 优化器中设置 `nesterov=True`，如下所示：

  ```python
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
  ```

  通过这个设置，Keras 会自动使用 NAG 进行梯度优化。NAG 能有效减少优化过程中的振荡现象，从而加快模型的收敛速度。

{.marker-none}

### AdaGrad算法

#### 问题 1：什么是 AdaGrad 算法？如何运作？

- **解答**：
  AdaGrad 是一种自适应学习率优化算法。它通过缩放每个参数对应的梯度来调节学习率，尤其是对陡峭维度的梯度进行更大的缩减，以加快朝着全局最优点的收敛速度。其核心思想是对每个参数的梯度平方进行累积，从而使得学习率在较陡峭的维度上迅速下降，保持在较平缓的维度上稳定更新。

  AdaGrad 的更新公式如下：

  ```KaTeX
  \begin{aligned}
  1. & \quad \mathbf{s} \leftarrow \mathbf{s} + \nabla_{\mathbf{\theta}} J(\mathbf{\theta}) \circ \nabla_{\mathbf{\theta}} J(\mathbf{\theta}) \\
  2. & \quad \mathbf{\theta} \leftarrow \mathbf{\theta} - \frac{\eta}{\sqrt{\mathbf{s}} + \epsilon} \circ \nabla_{\mathbf{\theta}} J(\mathbf{\theta})
  \end{aligned}
  ```

  - `KaTeX:\mathbf{s}`：累积梯度平方的向量。
  - `KaTeX:\nabla_{\mathbf{\theta}} J(\mathbf{\theta})`：关于参数 `KaTeX:\mathbf{\theta}` 的损失函数 `KaTeX:J(\mathbf{\theta})` 的梯度。
  - `KaTeX:\eta`：学习率。
  - `KaTeX:\epsilon`：平滑项，防止分母为 0，通常设为 `KaTeX:10^{-10}`。
  - `KaTeX:\circ`：表示逐元素操作。

  该算法的第一步通过累积各个梯度的平方来形成向量 `KaTeX:\mathbf{s}`，第二步使用 `KaTeX:\mathbf{s}` 的平方根对学习率进行调整，避免了学习率过大或过小的现象。

#### 问题 2：AdaGrad 与梯度下降的区别是什么？

- **解答**：
  - **梯度下降**：每个维度使用相同的学习率 `KaTeX:\eta`，这可能导致在陡峭维度上进展缓慢，平坦维度上更新不稳定。
  - **AdaGrad**：通过对每个维度累积梯度平方，使得学习率在梯度较陡峭的维度上迅速衰减，而在平坦维度上保持相对较高的学习率，从而更容易指向全局最优点。

  ![Figure11-8AdaGrade对比GD的区别](../assets/attachment/hands_on_machine_learning/Figure11-8AdaGrade对比GD的区别.png)如 **Figure 11-8** 所示，AdaGrad 相较于传统梯度下降，可以更早地校正其方向，使其朝向全局最优点。

#### 问题 3：AdaGrad 的优缺点是什么？

- **优点**：
  - **自适应学习率**：AdaGrad 自动调节每个参数的学习率，使得其在不同的维度上表现出自适应性，尤其适合稀疏梯度问题。
  - **较少的学习率调优**：相较于普通梯度下降，AdaGrad 对学习率的敏感度较低，减少了调参的复杂性。

- **缺点**：
  - **学习率衰减过快**：AdaGrad 在处理深度神经网络时，学习率衰减得太快，导致优化器在到达全局最优点之前就停滞不前。

#### 问题 4：Keras 中如何实现 AdaGrad？

- **解答**：
  Keras 提供了现成的 AdaGrad 优化器，可以直接使用：

  ```python
  optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
  ```

{.marker-none}

### RMSProp

#### 问题 1：RMSProp 如何改进 AdaGrad？

- **解答**：
  AdaGrad 有一个主要的缺点，即它倾向于随着时间的推移，使学习率逐渐变得非常小，导致优化器在到达全局最优点之前停止更新。RMSProp 通过仅累积最近迭代的梯度来解决这一问题，而不是从训练开始累积所有梯度。它通过在第一步使用指数衰减来实现这一点，从而对梯度更新进行加权平均。这避免了学习率快速下降的问题。

  RMSProp 算法的更新步骤如下：

  ```KaTeX
  \begin{aligned}
  1. & \quad \mathbf{s} \leftarrow \rho \mathbf{s} + (1 - \rho) \nabla_{\mathbf{\theta}} J(\mathbf{\theta}) \circ \nabla_{\mathbf{\theta}} J(\mathbf{\theta}) \\
  2. & \quad \mathbf{\theta} \leftarrow \mathbf{\theta} - \frac{\eta}{\sqrt{\mathbf{s}} + \epsilon} \circ \nabla_{\mathbf{\theta}} J(\mathbf{\theta})
  \end{aligned}
  ```

  其中：
  - `KaTeX:\rho` 是衰减率，典型值为 0.9。
  - `KaTeX:\mathbf{s}` 是梯度平方的累积和。
  - `KaTeX:\nabla_{\mathbf{\theta}} J(\mathbf{\theta})` 是参数 `KaTeX:\mathbf{\theta}` 对于损失函数 `KaTeX:J(\mathbf{\theta})` 的梯度。
  - `KaTeX:\eta` 是学习率。
  - `KaTeX:\epsilon` 是平滑项，通常设为 `KaTeX:10^{-10}`，防止分母为 0。

#### 问题 2：RMSProp 如何选择衰减率 `KaTeX:\rho`？

- **解答**：
  典型的衰减率 `KaTeX:\rho` 值为 0.9。虽然 `KaTeX:\rho` 是一个超参数，但其默认值在大多数情况下都表现良好，因此不需要特别调整。`KaTeX:\rho` 控制了梯度的指数加权移动平均的更新速率，值越接近 1，历史梯度的影响越大；值越接近 0，最近的梯度影响越大。

#### 问题 3：RMSProp 与 Keras 中的实现

- **解答**：
  Keras 中的 RMSProp 优化器可以通过以下代码直接使用：

  ```python
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
  ```

#### 问题 4：RMSProp 的适用场景是什么？

- **解答**：
  RMSProp 在大多数深度学习任务中表现良好，尤其是在处理神经网络训练的复杂问题时。它解决了 AdaGrad 中学习率衰减过快的问题，因此几乎总是比 AdaGrad 表现更好。实际上，在 Adam 优化器出现之前，RMSProp 是许多研究人员的首选优化算法。

{.marker-none}

### Adam


#### 问题 1: Adam 是什么？它如何工作的？

Adam 是 `KaTeX:\text{自适应动量估计 (adaptive moment estimation)}` 的缩写，它结合了 `KaTeX:\text{动量优化 (Momentum Optimization)}` 和 `KaTeX:\text{RMSProp}` 的思想。像动量优化一样，Adam 追踪过去梯度的指数衰减平均值，同时像 RMSProp 一样，它还追踪梯度平方的指数衰减平均值。这些值用来估计梯度的均值和方差，均值通常称为 `KaTeX:\text{第一矩 (first moment)}`，方差称为 `KaTeX:\text{第二矩 (second moment)}`。

#### 问题 2: Adam 的核心公式是什么？

```KaTeX
\begin{aligned}
1. \ & \mathbf{m} \leftarrow \beta_1 \mathbf{m} + (1 - \beta_1) \nabla_\theta J(\theta) \\
2. \ & \mathbf{s} \leftarrow \beta_2 \mathbf{s} + (1 - \beta_2) \nabla_\theta J(\theta) \odot \nabla_\theta J(\theta) \\
3. \ & \hat{\mathbf{m}} \leftarrow \mathbf{m} / (1 - \beta_1^t) \\
4. \ & \hat{\mathbf{s}} \leftarrow \mathbf{s} / (1 - \beta_2^t) \\
5. \ & \theta \leftarrow \theta - \eta \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{s}}} + \epsilon}
\end{aligned}
```

- `KaTeX:\mathbf{m}`：梯度的一阶矩估计（梯度均值）。
- `KaTeX:\mathbf{s}`：梯度的二阶矩估计（梯度方差）。
- `KaTeX:\beta_1` 和 `KaTeX:\beta_2`：控制一阶矩和二阶矩衰减率的超参数，通常 `KaTeX:\beta_1 = 0.9`，`KaTeX:\beta_2 = 0.999`。
- `KaTeX:\epsilon`：防止除零的一个小常数，通常取 `KaTeX:10^{-7}`。

#### 问题 3: 为什么 Adam 比其他优化器更受欢迎？

Adam 综合了动量优化和 RMSProp 的优点：
- **自适应学习率**：可以为每个参数动态调整学习率，使得训练过程更加高效且稳定。
- **较少的超参数调整**：通常可以直接使用默认的超参数（如 `KaTeX:\eta = 0.001`, `KaTeX:\beta_1 = 0.9`, `KaTeX:\beta_2 = 0.999`），适用于大多数任务。
- **快速收敛**：相比标准梯度下降算法，Adam 可以更快地收敛到全局最优。

#### 问题 4: Adam 优化器的 Keras 实现如何？

在 Keras 中可以很方便地使用 Adam 优化器，下面是 Adam 的默认实现：

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
```

这样，Adam 的默认超参数配置就可以直接应用到你的模型中。

{.marker-round}































## Machine Learning Project CheckList 

### CheckList Table {.col-span-3}

| ID  | English Steps                                        | 中文                       |
|-----|------------------------------------------------------|--------------------------|
| 1   | Frame the problem and look at the big picture.       | 确定问题并从整体上看待问题。           |
| 2   | Get the data.                                        | 获取数据。                    |
| 3   | Explore the data to gain insights.                   | 探索数据以获得洞察。               |
| 4   | Prepare the data to better expose the underlying data patterns to machine learning algorithms. | 准备数据，以便更好地揭示数据模式给机器学习算法。 |
| 5   | Explore many different models and shortlist the best ones. | 探索多种不同的模型，并挑选出最好的几个。     |
| 6   | Fine-tune your models and combine them into a great solution. | 微调模型，并将它们结合成一个出色的解决方案。   |
| 7   | Present your solution.                               | 展示你的解决方案。                |
| 8   | Launch, monitor, and maintain your system.           | 部署、监控并维护你的系统。            |

{.show-header .left-text}

## Machine learning Real-world Data 

### Data Source Table {.col-span-3}

| ID  | 分类                                    | 数据源                                      | 中文说明                           |
|-----|-----------------------------------------|---------------------------------------------|------------------------------------|
| 1   | **Popular open data repositories**      | [OpenML.org](https://www.openml.org)        | 一个开放机器学习数据集的平台       |
| 2   |                                         | [Kaggle.com](https://www.kaggle.com)        | 数据科学竞赛和数据集的平台         |
| 3   |                                         | [PapersWithCode.com](https://www.paperswithcode.com) | 提供机器学习论文及相关代码和数据集的平台 |
| 4   |                                         | [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) | 加州大学尔湾分校的机器学习数据集库 |
| 5   |                                         | [Amazon’s AWS datasets](https://registry.opendata.aws) | 亚马逊AWS提供的开放数据集         |
| 6   |                                         | [TensorFlow datasets](https://www.tensorflow.org/datasets) | TensorFlow框架下的开放数据集      |
| 7   | **Meta portals**                        | [DataPortals.org](https://www.dataportals.org) | 列出开放数据集门户网站的汇总平台 |
| 8   |                                         | [OpenDataMonitor.eu](http://www.opendatamonitor.eu) | 监控和分析开放数据集的平台        |
| 9   | **Other pages listing data repositories** | [Wikipedia’s list of machine learning datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research) | 维基百科的机器学习数据集列表       |
| 10  |                                         | [Quora.com](https://www.quora.com)         | 知识问答平台，有时包含数据集资源   |
| 11  |                                         | [The datasets subreddit](https://www.reddit.com/r/datasets/) | Reddit上的数据集讨论和分享社区   |

{.show-header .left-text}

## Machine Learning Performance Measure

### 性能度量指标选择表 {.col-span-3}

| **序号** | **性能度量指标** | **适用场景** | **优点** | **缺点** | **数学定义** | **度量原理** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **均方根误差 (RMSE)** | 回归问题 | 对大误差敏感，适用于正态分布的数据 | 对异常值非常敏感 | ```KaTeX:RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left[h(x^{(i)}) - y^{(i)}\right]^2}``` | 衡量预测值与实际值之间的平方差的平均值的平方根，强调大误差 |
| 2 | **平均绝对误差 (MAE)** | 回归问题 | 对异常值不敏感，更适用于含有异常值的数据 | 不如 ```KaTeX:RMSE``` 强调大误差 | ```KaTeX:MAE = \frac{1}{m} \sum_{i=1}^{m} \mid h(x^{(i)}) - y^{(i)}\mid``` | 计算预测值与实际值之间的绝对差的平均值，强调所有误差的平均水平 |
| 3 | **R² 决定系数** | 回归问题 | 易于理解，表示模型的解释能力 | 在数据异常分布时可能误导 | ```KaTeX:R^2 = 1 - \frac{\sum_{i=1}^{m} \left[y^{(i)} - \hat{y}^{(i)}\right]^2}{\sum_{i=1}^{m} \left[y^{(i)} - \bar{y}\right]^2}``` | 衡量模型解释方差的比例，表示模型对实际数据的拟合程度 |
| 4 | **对数损失 (Log Loss)** | 二分类问题 | 对模型概率输出的评估有效 | 对分类错误非常敏感 | ```KaTeX:Log Loss = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]``` | 衡量分类模型输出的概率分布与实际标签之间的差异 |
| 5 | **准确率 (Accuracy)** | 分类问题 | 简单直观，适用于类均衡数据 | 对类不平衡数据敏感 | ```KaTeX:Accuracy = \frac{TP + TN}{TP + TN + FP + FN}``` | 衡量预测正确的样本占总样本的比例 |
| 6 | **精确率 (Precision)** | 分类问题（尤其关注正类） | 适用于需要减少误报的场景 | 对正负类不均衡数据不敏感 | ```KaTeX:Precision = \frac{TP}{TP + FP}``` | 衡量预测为正的样本中实际为正的比例 |
| 7 | **召回率 (Recall)** | 分类问题（尤其关注正类） | 适用于减少漏报的场景 | 对负类不敏感，可能导致更多误报 | ```KaTeX:Recall = \frac{TP}{TP + FN}``` | 衡量实际为正的样本中被正确预测为正的比例 |
| 8 | **F1 Score** | 分类问题 | 结合了精确率和召回率的优点 | 对类别不均衡数据有效 | ```KaTeX:F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}``` | 综合精确率和召回率的调和平均，平衡两者的重要性 |


{.show-header .left-text}