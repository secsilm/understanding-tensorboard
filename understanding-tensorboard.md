# 理解 TensorBoard

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 是用于可视化 TensorFlow 模型的训练过程的工具（the flow of tensors），在你安装 TensorFlow 的时候就已经安装了 TensorBoard。我在前面的 [【TensorFlow】TensorFlow 的卷积神经网络 CNN - TensorBoard版](http://blog.csdn.net/u010099080/article/details/62882006) 和 [【Python | TensorBoard】用 PCA 可视化 MNIST 手写数字识别数据集](http://blog.csdn.net/u010099080/article/details/53560426) 分别非常简单的介绍了一下这个工具，没有详细说明，这次来（尽可能详细的）整体说一下，而且这次也是对 [前者](http://blog.csdn.net/u010099080/article/details/62882006) 代码的一个升级，很大程度的改变了代码结构，将输入和训练分离开来，结构更清晰。小弟不才，如有错误，欢迎评论区指出。

# OVERVIEW

总体上，目前 TensorBoard 主要包括下面几种：
![tensorboard-overview](http://i.imgur.com/Bn30VhS.png)
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

## IMAGES

## DISTRIBUTIONS

## HISTOGRAMS

## TODO
