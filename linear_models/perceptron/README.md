## 1. 模型
感知机是一个二分类的线性分类模型，属于判别模型。
数学公式表现为：
$$y=sign(w \cdot x+b) \\
sign = \left \{
\begin{aligned}
+1, & \ x \geq 0 \\
-1, & \ x < 0 \\
\end{aligned}

\right.$$
## 2. 策略
感知机的优化策略为让分错的点离分割超平面尽可能的小
已知点到超平面的距离为
$$\frac{|wx+b|}{||w||}$$
分类错误的点满足条件
$$y_i(wx+b) < 0$$
$$-y_i(wx+b) > 0$$
分类错误点到超平面的距离为
$$-y_i \frac{wx+b}{||w||}$$
由此可以定义其loss函数为
$$\min -y_i(wx+b)$$
整体的loss函数为
$$\min -\sum_{i=1}^N y_i(wx+b)$$
## 3. 算法
使用梯度下降算法来最小化loss，对$$w$$和$$b$$的导数分别为
$$\frac{\partial L}{\partial w} = -y_ix \\
\frac{\partial L}{\partial b} = -y_i $$
## 4. 代码实现
这里用代码简单的实现了感知机迭代的过程，结果保存在`iter_result`目录下。