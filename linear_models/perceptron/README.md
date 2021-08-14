## 1. 模型
感知机是一个二分类的线性分类模型，属于判别模型。
数学公式表现为：
![](https://cdn.nlark.com/yuque/__latex/b7dd8ef85d37e8a4156b7c13579cb9de.svg#card=math&code=y%3Dsign%28w%20%5Ccdot%20x%2Bb%29%20%5C%5C%0Asign%20%3D%20%5Cleft%20%5C%7B%0A%5Cbegin%7Baligned%7D%0A%2B1%2C%20%26%20%5C%20x%20%5Cgeq%200%20%5C%5C%0A-1%2C%20%26%20%5C%20x%20%3C%200%20%5C%5C%0A%5Cend%7Baligned%7D%0A%0A%5Cright.&height=67&width=724)
## 2. 策略
感知机的优化策略为让分错的点离分割超平面尽可能的小
已知点到超平面的距离为
![](https://cdn.nlark.com/yuque/__latex/7c46b4a45f3b92e2823242eb432bea1f.svg#card=math&code=%5Cfrac%7B%7Cwx%2Bb%7C%7D%7B%7C%7Cw%7C%7C%7D&height=47&width=65)
分类错误的点满足条件
![](https://cdn.nlark.com/yuque/__latex/2cf34f9f348a3d9fc91b018f035d3418.svg#card=math&code=y_i%28wx%2Bb%29%20%3C%200&height=20&width=107)
![](https://cdn.nlark.com/yuque/__latex/d3c5e0b9bedb32c74c99733e21971d35.svg#card=math&code=-y_i%28wx%2Bb%29%20%3E%200&height=20&width=120)
分类错误点到超平面的距离为
![](https://cdn.nlark.com/yuque/__latex/27d4544d911f46bf73e6d80a05c54de4.svg#card=math&code=-y_i%20%5Cfrac%7Bwx%2Bb%7D%7B%7C%7Cw%7C%7C%7D&height=45&width=82)
由此可以定义其loss函数为
![](https://cdn.nlark.com/yuque/__latex/46c42538c27a522511d56c663c047134.svg#card=math&code=%5Cmin%20-y_i%28wx%2Bb%29&height=20&width=120)
整体的loss函数为
![](https://cdn.nlark.com/yuque/__latex/f8eac76f44d291716f23b2a9c68434c5.svg#card=math&code=%5Cmin%20-%5Csum_%7Bi%3D1%7D%5EN%20y_i%28wx%2Bb%29&height=53&width=150)
## 3. 算法
使用梯度下降算法来最小化loss，对![](https://cdn.nlark.com/yuque/__latex/f1290186a5d0b1ceab27f4e77c0c5d68.svg#card=math&code=w&height=12&width=12)和![](https://cdn.nlark.com/yuque/__latex/92eb5ffee6ae2fec3ad71c777531578f.svg#card=math&code=b&height=16&width=7)的导数分别为
![](https://cdn.nlark.com/yuque/__latex/27b0b84880ee0d5c535f1a0e60263697.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w%7D%20%3D%20-y_ix%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%7D%20%3D%20-y_i%20&height=83&width=724)
## 4. 代码实现
这里用代码简单的实现了感知机迭代的过程，结果保存在`iter_result`目录下。
