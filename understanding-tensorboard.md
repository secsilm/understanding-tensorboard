# 理解 TensorBoard

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 是用于可视化 TensorFlow 模型的训练过程的工具（the flow of tensors），在你安装 TensorFlow 的时候就已经安装了 TensorBoard。我在前面的 [【TensorFlow】TensorFlow 的卷积神经网络 CNN - TensorBoard版](http://blog.csdn.net/u010099080/article/details/62882006) 和 [【Python | TensorBoard】用 PCA 可视化 MNIST 手写数字识别数据集](http://blog.csdn.net/u010099080/article/details/53560426) 分别非常简单的介绍了一下这个工具，没有详细说明，这次来（尽可能详细的）整体说一下，而且这次也是对 [前者](http://blog.csdn.net/u010099080/article/details/62882006) 代码的一个升级，很大程度的改变了代码结构，将输入和训练分离开来，结构更清晰。小弟不才，如有错误，欢迎评论区指出。

全部代码和 TensorBoard 文件均在 [我的 GitHub](https://github.com/secsilm/understanding-tensorboard) 上。

**Tensorboard 使用的版本为 0.1.4，对应于 TensorFlow 1.3.0，但训练代码未在 TensorFlow 1.3.0 上测试，不过应该是可以运行的。Windows 下 TensorFlow 的安装可以看 [【TensorFlow】Windows10 64位下安装TensorFlow - 官方原生支持](http://blog.csdn.net/u010099080/article/details/53418159)** 。

## TensorBoard 是如何工作的？

简单来说，TensorBoard 是通过一些操作（summary operations）将数据记录到文件（event files）中，然后再读取文件来完成作图的。想要在浏览器上看到 TensorBoard 页面，大概需要这几步：

1. **summary**。在定义计算图的时候，在适当的位置加上一些 [summary 操作](https://www.tensorflow.org/api_guides/python/summary) 。
2. **merge**。你很可能加了很多 summary 操作，我们需要使用 `tf.summary.merge_all` 来将这些 summary 操作聚合成一个操作，由它来产生所有 summary 数据。
3. **run**。在没有运行的时候这些操作是不会执行任何东西的，仅仅是定义了一下而已。在运行（开始训练）的时候，我们需要通过 `tf.summary.FileWriter()` 指定一个目录来告诉程序把产生的文件放到哪。然后在运行的时候使用 `add_summary()` 来将某一步的 summary 数据记录到文件中。

当训练完成后，在命令行使用 `tensorboard --logdir=path/to/log-directory` 来启动 TensorBoard，按照提示在浏览器打开页面，注意把 `path/to/log-directory` 替换成你上面指定的目录。

## OVERVIEW

总体上，目前 TensorBoard 主要包括下面几个面板：

![tensorboard-overview](https://i.imgur.com/TK5BIH4.png)

其中 `TEXT` 是 最新版（应该是 1.3）才加进去的，实验性功能，官方都没怎么介绍。除了 `AUDIO`（没用过）、`EMBEDDINGS`（还不是很熟） 和 `TEXT`（没用过） 这几个，这篇博客主要说剩下的几个，其他的等回头熟练了再来说，尽量避免误人子弟。

TensorBoard 的工作原理是读取模型训练时产生的 TensorFlow events 文件，这个文件包括了一些 summary 数据（就是作图时用的数据）。

## SCALARS

![](http://i.imgur.com/s9drW9Z.png)
*TensorBoard 的默认打开样式*

TensorBoard 打开时默认直接进入 `SCALARS`，并且默认使用 `.*` 正则表达式显示所有图（其他面板同理，下面就不再赘述），你用到的面板会在顶部导航栏直接显示，而其他用不到的（你代码中没有相关代码）则会收起到 `INACTIVE` 中。

![](http://i.imgur.com/W8ewfgz.png)

`SCALARS` 主要用于记录诸如准确率、损失和学习率等单个值的变化趋势。在代码中用 [`tf.summary.scalar()`](https://www.tensorflow.org/api_docs/python/tf/summary/scalar) 来将其记录到文件中。对应于我的代码中，我是使用其记录了训练准确率和损失。

训练准确率：

```python
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
```

![](http://i.imgur.com/fpWU4rj.png)
*全部 run 的 acuracy*

> 可以看到这些曲线并不是那么平滑，这是因为我记录的步数比较少，也就是点比较少，如果每一步都记录或者间隔比较短，那么最后的文件会很大。下同

损失：

```python
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) + \
        BETA * tf.add_n([tf.nn.l2_loss(v)
                        for v in trainable_vars if not 'b' in v.name])

    tf.summary.scalar('loss', loss)
```

![](http://i.imgur.com/zLPpuJt.png)
*全部 run 的 loss*

每个图的右下角都有 3 个小图标，第一个是查看大图，第二个是是否对 y 轴对数化，第三个是如果你拖动或者缩放了坐标轴，再重新回到原始位置。

![](http://i.imgur.com/jPNysqZ.gif)
*Fit domain to data*

`tf.summary.scalar(name, tensor)` 有两个参数：

- `name`：可以理解为图的标题。在 `GRAPHS` 中则是该节点的名字
- `tensor`：包含单个值的 tensor，说白了就是作图的时候要用的数据

在上面的图中，可以看到除了 `accuracy` 和 `loss` 外，还有一个 `eval_accuracy`，这个是我用来记录验证准确率的，代码中相关的部分如下：

```python
eval_writer = tf.summary.FileWriter(LOGDIR + '/eval')

# Some other codes

eval_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='eval_accuracy', simple_value=np.mean(test_acc))]), i)
```

![](http://i.imgur.com/XxQ1oRW.png)
*全部 run 的 eval_accuracy*

这里我是手动添加了一个验证准确率到 `SCALARS` 中，其实想要记录验证准确率完全不必这么做，和训练准确率不同的只是 feed 的数据不一样而已。然而由于我的显存不够一次装下整个验证集，所以我就分了两部分计算然后求平均值来得到整个验证集的准确率。*如果谁有更好的方法，请给我发邮件或者在评论区评论，非常感谢 :-)*

当我们有很多的 tag （图）的时候，我们可以在左上角写正则表达式来选择其中一些 tag，比如我想选择包括 `accuracy` 的 tag，那么我直接写上 `accuracy` 就可以了，右侧就会多出一个 `accuracy` 的 tag，显示匹配出的结果。

页面左上是 `Show data download links` 和 `Ignore outliers in chart scaling`，这两个比较好理解，第一个就是显示数据下载链接，可以把 TensorBoard 作图用的数据下载下来，点击后可以在图的右下角可以看到下载链接以及选择下载哪一个 run 的，下载格式支持 CSV 和 JSON。第二个是排除异常点，默认选中。

当我们用鼠标在图上滑过的时候可以显示出每个 run 对应的点的值，这个显示顺序是由 `Tooltip sorting method` 来控制的，有 `default`、`descending`（降序）、`asceding` （升序）和 `nearest` 四个选项，大家可以试试点几下。

而下面的 `Smoothing` 指的是作图时曲线的平滑程度，使用的是**类似**指数平滑的处理方法。如果不平滑处理的话，有些曲线波动很大，难以看出趋势。0 就是不平滑处理，1 就是最平滑，默认是 0.6。

`Horizontal Axis` 顾名思义指的是横轴的设置：

- `STEP`：默认选项，指的是横轴显示的是训练迭代次数
- `RELATIVE`：这个相对指的是相对时间，相对于训练开始的时间，也就是说是*训练用时* ，单位是小时
- `WALL`：指训练的绝对时间

最下面的 `Runs` 列出了各个 run，你可以选择只显示某一个或某几个。

## IMAGES

如果你的模型输入是图像（的像素值），然后你想看看模型每次的输入图像是什么样的，以保证每次输入的图像没有问题（因为你可能在模型中对图像做了某种变换，而这种变换是很容易出问题的），`IMAGES` 面板就是干这个的，它可以显示出相应的输入图像，默认显示最新的输入图像，如下图：

![](http://i.imgur.com/nqVkbBo.png)
*第 45000 次迭代时输入的 3 个图像*

图的右下角的两个图标，第一个是查看大图，第二个是查看原图（真实大小，默认显示的是放大后的）。左侧和 `SCALARS` 差不多，我就不赘述了。

而在代码中，需要在合适的位置使用 [`tf.summary.image()`](https://www.tensorflow.org/api_docs/python/tf/summary/image) 来把图像记录到文件中，其参数和 `tf.summary.scalar()` 大致相同，多了一个 `max_outputs` ，指的是最多显示多少张图片，默认为 3。对应于我的代码，如下：

```python
x = tf.placeholder(tf.float32, shape=[None, N_FEATURES], name='x')
x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), perm=[0, 2, 3, 1])
tf.summary.image('input', x_image, max_outputs=3)
y = tf.placeholder(tf.float32, [None, N_CLASSES], name='labels')
```

## GRAPHS

这个应该是最常用的面板了。很多时候我们的模型很复杂，包含很多层，我们想要总体上看下构建的网络到底是什么样的，这时候就用到 `GRAPHS` 面板了，在这里可以展示出你所构建的网络整体结构，显示数据流的方向和大小，也可以显示训练时每个节点的用时、耗费的内存大小以及参数多少。默认显示的图分为两部分：主图（Main Graph）和辅助节点（Auxiliary Nodes）。其中主图显示的就是网络结构，辅助节点则显示的是初始化、训练、保存等节点。我们可以双击某个节点或者点击节点右上角的 `+` 来展开查看里面的情况，也可以对齐进行缩放，每个节点的命名都是我们在代码中使用 `tf.name_scope()` 定义好的。下面介绍下该面板左侧的功能。


![](http://i.imgur.com/r8mXplM.png)
*计算图*

左上是 `Fit to screen`，顾名思义就是将图缩放到适合屏幕。下面的 `Download PNG` 则是将图保存到本地。`Run` 和 `Session Run` 分别是不同的训练和迭代步数。比如我这里以不同的超参训练了 6 次，那么 就有 6 个 run，而你所记录的迭代次数（并不是每一步都会记录当前状态的，那样的话太多了，一般都是每隔多少次记录一次）则显示在 `Session Run` 里。再下面大家应该都能看懂，我就不详细说每个功能的意思了。

![](https://i.imgur.com/sJFNrmY.png)
*选择迭代步数*

TensorBoard 默认是不会记录每个节点的用时、耗费的内存大小等这些信息的，那么如何才能在图上显示这些信息呢？关键就是如下这些代码，主要就是在 `sess.run()` 中加入 `options` 和 `run_metadata` 参数。

```python
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
s, lss, acc , _ = sess.run([merged_summary, loss, accuracy, train_step], 
                           feed_dict={x: batch_x, y: batch_y, phase: 1},
                           options=run_options,
                           run_metadata=run_metadata)
summary_writer.add_run_metadata(run_metadata, 'step{}'.format(i))
summary_writer.add_summary(s, i)
```

然后我们就可以选择 `Compute time` 或者 `Memory` 来查看相应信息，颜色深浅代表耗时多少或者内存耗用多少。

![](https://i.imgur.com/TIMp4tq.png)
*计算耗时*

我们也可以将某个节点从主图移除，将其放到辅助节点中，以便于我们更清晰的观察整个网络。具体操作是 *右键该节点，选择 `Remove from main graph`* 。

## DISTRIBUTIONS

`DISTRIBUTIONS` 主要用来展示网络中各参数随训练步数的增加的变化情况，可以说是 *多分位数折线图* 的堆叠。下面我就下面这张图来解释下。

![conv1-weights](https://i.imgur.com/UIrCsl9.png)
*权重分布*

这张图表示的是第二个卷积层的权重变化。横轴表示训练步数，纵轴表示权重值。而从上到下的折现分别表示权重分布的不同分位数：`[maximum, 93%, 84%, 69%, 50%, 31%, 16%, 7%, minimum]`。对应于我的代码，部分如下：

```python
with tf.name_scope(name):
    W = tf.Variable(tf.truncated_normal(
        [k, k, channels_in, channels_out], stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='b')
    conv = tf.nn.conv2d(inpt, W, strides=[1, s, s, 1], padding='SAME')
    act = tf.nn.relu(conv)
    tf.summary.histogram('weights', W)
    tf.summary.histogram('biases', b)
    tf.summary.histogram('activations', act)
```

## HISTOGRAMS

**`HISTOGRAMS` 和 `DISTRIBUTIONS` 是对同一数据不同方式的展现**。与 `DISTRIBUTIONS` 不同的是，`HISTOGRAMS` 可以说是 *频数分布直方图* 的堆叠。

![conv1-weights-hist](https://i.imgur.com/nJdgFG4.png)
*权重分布*

横轴表示权重值，纵轴表示训练步数。颜色越深表示时间越早，越浅表示时间越晚（越接近训练结束）。除此之外，`HISTOGRAMS` 还有个 `Histogram mode`，有两个选项：`OVERLAY` 和 `OFFSET`。选择 `OVERLAY` 时横轴为权重值，纵轴为频数，每一条折线为训练步数。颜色深浅与上面同理。默认为 `OFFSET` 模式。

## 后记

这篇博文写了好久，从准备数据到开始动笔写，中间一直被各种事干扰。由于我水平有限，我只能尽最大程度的给出尽可能正确的解释，然而还有很多我目前还兼顾不到，很多话也不是很通顺。如有错误，欢迎在评论区或者给我私信或者给我邮件指出。

## REFERENCES

- [TensorBoard: Visualizing Learning](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
- [How does one interpret histograms given by TensorFlow in TensorBoard?](https://stats.stackexchange.com/questions/220491/how-does-one-interpret-histograms-given-by-tensorflow-in-tensorboard)
- [TensorBoard](https://github.com/tensorflow/tensorboard/blob/master/README.md)
- [TensorBoard Histogram Dashboard](https://www.tensorflow.org/get_started/tensorboard_histograms)
- [Understanding TensorBoard (weight) histograms](https://stackoverflow.com/questions/42315202/understanding-tensorboard-weight-histograms)
- [Hands-on TensorBoard (TensorFlow Dev Summit 2017)](https://www.youtube.com/watch?v=eBbEDRsCmv4&t=1105s)

## END
