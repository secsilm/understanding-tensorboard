# 理解 TensorBoard

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 是用于可视化 TensorFlow 模型的训练过程的工具（the flow of tensors），在你安装 TensorFlow 的时候就已经安装了 TensorBoard。我在前面的 [【TensorFlow】TensorFlow 的卷积神经网络 CNN - TensorBoard版](http://blog.csdn.net/u010099080/article/details/62882006) 和 [【Python | TensorBoard】用 PCA 可视化 MNIST 手写数字识别数据集](http://blog.csdn.net/u010099080/article/details/53560426) 分别非常简单的介绍了一下这个工具，没有详细说明，这次来（尽可能详细的）整体说一下，而且这次也是对 [前者](http://blog.csdn.net/u010099080/article/details/62882006) 代码的一个升级，很大程度的改变了代码结构，将输入和训练分离开来，结构更清晰。小弟不才，如有错误，欢迎评论区指出。

# OVERVIEW

总体上，目前 TensorBoard 主要包括下面几个面板：

![tensorboard-overview](https://i.imgur.com/TK5BIH4.png)

其中 `TEXT` 是 最新版（应该是 1.3）才加进去的，实验性功能，官方都没怎么介绍。除了 `AUDIO`（没用过）、`EMBEDDINGS`（还不是很熟） 和 `TEXT`（没用过） 这几个，这篇博客主要说剩下的几个，其他的等回头熟练了再来说，尽量避免误人子弟。

## SCALARS

`SCALARS` 主要用于记录诸如准确率、损失和学习率等单个值的变化趋势。在代码中用 [`tf.summary.scalar()`](https://www.tensorflow.org/api_docs/python/tf/summary/scalar) 来将其记录到文件中。对应于我的代码中，我是使用其记录了训练准确率和损失。

训练准确率：

```python
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
```

损失：

```python
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) + \
        BETA * tf.add_n([tf.nn.l2_loss(v)
                        for v in trainable_vars if not 'b' in v.name])

    tf.summary.scalar('loss', loss)
```

效果：

每个图的右下角都有两个小图标，第一个是查看大图，第二个是是否对 y 轴对数化。

`tf.summary.scalar(name, tensor)` 有两个参数：

- `name`：可以理解为图的标题。在 `GRAPHS` 中则是该节点的名字
- `tensor`：包含单个值的 tensor，说白了就是作图的时候要用的数据

在上面的图中，可以看到除了 `accuracy` 和 `loss` 外，还有一个 `eval_accuracy`，这个是我用来记录验证准确率的，代码中相关的部分如下：

```python
eval_writer = tf.summary.FileWriter(LOGDIR + '/eval')

# Some other codes

eval_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='eval_accuracy', simple_value=np.mean(test_acc))]), i)
```

这里我是手动添加了一个验证准确率到 `SCALARS` 中，其实想要记录验证准确率完全不必这么做，和训练准确率不同的只是 feed 的数据不一样而已。然而由于我的显存不够一次装下整个验证集，所以我就分了两部分计算然后求平均值来得到整个验证集的准确率。*如果谁有更好的方法，请给我发邮件或者在评论区评论，非常感谢 :-)*

当我们有很多的 tag （图）的时候，我们可以在左上角写正则表达式来选择其中一些 tag，比如我想选择包括 `accuracy` 的 tag，那么我直接写上 `accuracy` 就可以了，右侧就会多出一个 `accuracy` 的 tag，显示匹配出的结果。

在下面是 `Split on underscores` 和 `Data download links`，这两个比较好理解，第一个就是以下划线分割，第二个是显示数据下载链接，可以把 TensorBoard 作图用的数据下载下来，点击后可以在图的右下角可以看到下载链接以及选择下载哪一个 run 的，下载格式支持 CSV 和 JSON。

当我们用鼠标在图上滑过的时候可以显示出每个 run 对应的点的值，这个显示顺序是由 `Tooltip sorting method` 来控制的，有 `default`、`descending`（降序）、`asceding` （升序）和 `nearest` 四个选项，大家可以试试点几下。

而下面的 `Smoothing` 指的是作图时曲线的平滑程度，使用的是**类似**指数平滑的处理方法。如果不平滑处理的话，有些曲线波动很大，难以看出趋势。0 就是不平滑处理，1 就是最平滑，默认是 0.6。

`Horizontal Axis` 顾名思义指的是横轴的设置：

- `STEP`：默认选项，指的是横轴显示的是训练迭代次数
- `RELATIVE`：这个相对指的是相对时间，相对于训练开始的时间，也就是说是*训练用时* ，单位是小时
- `WALL`：指训练的绝对时间

最下面的 `Runs` 列出了各个 run，你可以选择只显示某一个或某几个。

## IMAGES

如果你的模型输入是图像（的像素值），然后你想看看模型每次的输入图像是什么样的，以保证每次输入的图像没有问题（因为你可能在模型中对图像做了某种变换，而这种变换是很容易出问题的），`IMAGES` 面板就是干这个的，它可以显示出相应的输入图像，默认显示最新的输入图像，如下图：

图的右下角的两个图标，第一个是查看大图，第二个是查看原图（真实大小，默认显示的是放大后的）。左侧和 `SCALARS` 差不多，我就不赘述了。

而在代码中，需要在合适的位置使用 [`tf.summary.image()`](https://www.tensorflow.org/api_docs/python/tf/summary/image) 来把图像记录到文件中，其参数和 `tf.summary.scalar()` 大致相同，多了一个 `max_outputs` ，指的是最多显示多少张图片，默认为 3。对应于我的代码，如下：

```python
x = tf.placeholder(tf.float32, shape=[None, N_FEATURES], name='x')
x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), perm=[0, 2, 3, 1])
tf.summary.image('input', x_image, max_outputs=3)
y = tf.placeholder(tf.float32, [None, N_CLASSES], name='labels')
```

## GRAPHS

这个应该是最常用的面板了。很多时候我们的模型很复杂，包含很多层，我们想要总体上看下构建的网络到底是什么样的，这时候就用到 `GRAPHS` 面板了。
## DISTRIBUTIONS

## HISTOGRAMS

## TODO
