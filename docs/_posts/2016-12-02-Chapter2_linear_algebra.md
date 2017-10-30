---
title: 线性代数
layout: post
share: false
---

线性代数作为数学的一个分支，广泛应用于科学和工程中。
然而，因为线性代数主要是面向连续数学，而非离散数学，所以很多计算机科学家很少接触它。
掌握好线性代数对于理解和从事机器学习算法相关工作是很有必要的，尤其对于深度学习算法而言。
因此，在开始介绍深度学习之前，我们集中探讨一些必备的线性代数知识。


如果你已经很熟悉线性代数，那么可以轻松地跳过本章。
如果你已经了解这些概念，但是需要一份索引表来回顾一些重要公式，那么我们推荐\emph{The Matrix Cookbook} {cite?}。
如果你没有接触过线性代数，那么本章将告诉你本书所需的线性代数知识，不过我们仍然非常建议你参考其他专门讲解线性代数的文献，例如~{shilov1977linear}。
最后，本章略去了很多重要但是对于理解深度学习非必需的线性代数知识。





# 标量、向量、矩阵和张量


学习线性代数，会涉及以下几类数学概念：

+ 标量：一个标量就是一个单独的数，它不同于线性代数中研究的其他大部分对象（通常是多个数的数组）。
    我们用斜体表示标量。标量通常被赋予小写的变量名称。
    当我们介绍标量时，会明确它们是哪种类型的数。
    比如，在定义实数标量时，我们可能会说"令$\Ss \in \SetR$表示一条线的斜率"；在定义自然数标量时，我们可能会说"令$\Sn\in\SetN$表示元素的数目"。

<!-- % -- 29 -- -->

+ 向量：一个向量是一列数。
    这些数是有序排列的。
    通过次序中的索引，我们可以确定每个单独的数。
    通常我们赋予向量粗体的小写变量名称，比如$\Vx$。
    向量中的元素可以通过带脚标的斜体表示。
    向量$\Vx$的第一个元素是$\Sx_1$，第二个元素是$\Sx_2$，等等。
    我们也会注明存储在向量中的元素是什么类型的。
    如果每个元素都属于$\SetR$，并且该向量有$\Sn$个元素，那么该向量属于实数集$\SetR$的$\Sn$次笛卡尔乘积构成的集合，记为$\SetR^n$。
    当需要明确表示向量中的元素时，我们会将元素排列成一个方括号包围的纵列：
    \begin{equation}
        \Vx=\begin{bmatrix} \Sx_1   \\  
                            \Sx_2   \\ 
                            \vdots  \\ 
                            \Sx_n 
                \end{bmatrix}.
    \end{equation}
    我们可以把向量看作空间中的点，每个元素是不同坐标轴上的坐标。
    
    有时我们需要索引向量中的一些元素。
    在这种情况下，我们定义一个包含这些元素索引的集合，然后将该集合写在脚标处。
    比如，指定$\Sx_1$，$\Sx_3$和$\Sx_6$，我们定义集合$S=\{1,3,6\}$，然后写作$\Vx_S$。我
    们用符号－表示集合的补集中的索引。
    比如$\Vx_{-1}$表示$\Vx$中除$\Sx_1$外的所有元素，$\Vx_{-S}$表示$\Vx$中除$\Sx_1$，$\Sx_3$，$\Sx_6$外所有元素构成的向量。

+ 矩阵：矩阵是一个二维数组，其中的每一个元素被两个索引（而非一个）所确定。
    我们通常会赋予矩阵粗体的大写变量名称，比如$\MA$。
    如果一个实数矩阵高度为$m$，宽度为$n$，那么我们说$\MA\in \SetR^{m\times n}$。
    我们在表示矩阵中的元素时，通常以不加粗的斜体形式使用其名称，索引用逗号间隔。
    比如，$\SA_{1,1}$表示$\MA$左上的元素，$\SA_{m,n}$表示$\MA$右下的元素。
    我们通过用":"表示水平坐标，以表示垂直坐标$\Si$中的所有元素。
    比如，$\MA_{i,:}$表示$\MA$中垂直坐标$i$上的一横排元素。
    这也被称为$\MA$的第$i$~行。
    同样地，$\MA_{:,i}$表示$\MA$的第$i$~列。
    当我们需要明确表示矩阵中的元素时，我们将它们写在用方括号括起来的数组中：
    \begin{equation}
        \begin{bmatrix}
            A_{1,1} & A_{1,2} \\
            A_{2,1} & A_{2,2} \\
        \end{bmatrix}.
    \end{equation}
    有时我们需要索引矩阵值表达式，而这些表达式不是单个字母。
    在这种情况下，我们在表达式后面接下标，但不必将矩阵的变量名称小写化。
    比如，$f(\MA)_{i,j}$表示函数$f$作用在$\MA$上输出的矩阵的第$i$行第$j$列元素。

<!-- % -- 30 -- -->

+ 张量：在某些情况下，我们会讨论坐标超过两维的数组。
    一般地，一个数组中的元素分布在若干维坐标的规则网格中，我们称之为张量。
    我们使用字体$\TSA$来表示张量"A"。
    张量$\TSA$中坐标为$(i,j,k)$的元素记作$\TEA_{i,j,k}$。



转置是矩阵的重要操作之一。
矩阵的转置是以对角线为轴的镜像，这条从左上角到右下角的对角线被称为主对角线。
\fig?显示了这个操作。
我们将矩阵$\MA$的转置表示为$\MA^\top$，定义如下
\begin{equation}
(\MA^\top)_{i,j}= \SA_{j,i}.
\end{equation}

向量可以看作只有一列的矩阵。
对应地，向量的转置可以看作是只有一行的矩阵。
有时，我们通过将向量元素作为行矩阵写在文本行中，然后使用转置操作将其变为标准的列向量，来定义一个向量，比如$\Vx=[\Sx_1, \Sx_2, \Sx_3]^\top$.


标量可以看作是只有一个元素的矩阵。
因此，标量的转置等于它本身，$\Sa=\Sa^\top$。

\begin{figure}[!hbt]
\ifOpenSource
\centerline{\includegraphics{figure.pdf}}
\else
\centerline{\includegraphics{Chapter2/figures/transpose}}
\fi
\caption{矩阵的转置可以看成以主对角线为轴的一个镜像。}
\end{figure}

<!-- % -- 31 -- -->

只要矩阵的形状一样，我们可以把两个矩阵相加。
两个矩阵相加是指对应位置的元素相加，比如$\MC=\MA+\MB$，其中$\SC_{i,j}= \SA_{i,j}+\SB_{i,j}$。


标量和矩阵相乘，或是和矩阵相加时，我们只需将其与矩阵的每个元素相乘或相加，比如$\MD = \Sa \cdot \MB + \Sc$，其中$\SD_{i,j} = \Sa\cdot  \SB_{i,j} + \Sc$。


在深度学习中，我们也使用一些不那么常规的符号。
我们允许矩阵和向量相加，产生另一个矩阵：$\MC=\MA + \Vb$，其中$\SC_{i,j}= \SA_{i,j} + \Sb_{j}$。
换言之，向量$\Vb$和矩阵$\MA$的每一行相加。
这个简写方法使我们无需在加法操作前定义一个将向量$\Vb$复制到每一行而生成的矩阵。
这种隐式地复制向量$\Vb$到很多位置的方式，被称为广播。





# 矩阵和向量相乘


矩阵乘法是矩阵运算中最重要的操作之一。
两个矩阵$\MA$和$\MB$的矩阵乘积是第三个矩阵$\MC$。
为了使乘法定义良好，矩阵$\MA$的列数必须和矩阵$\MB$的行数相等。
如果矩阵$\MA$的形状是$\Sm \times \Sn$，矩阵$\MB$的形状是$\Sn\times \Sp$，那么矩阵$\MC$的形状是$\Sm\times \Sp$。
我们可以通过将两个或多个矩阵并列放置以书写矩阵乘法，例如
\begin{equation}
    \MC=\MA\MB.
\end{equation}


具体地，该乘法操作定义为
\begin{equation}
    \SC_{i,j}=\sum_k \SA_{i,k} \SB_{k,j}.
\end{equation}


需要注意的是，两个矩阵的标准乘积\emph{不是}指两个矩阵中对应元素的乘积。
不过，那样的矩阵操作确实是存在的，被称为元素对应乘积或者Hadamard乘积，记为$\MA\odot\MB$。


两个相同维数的向量$\Vx$和$\Vy$的点积可看作是矩阵乘积$\Vx^\top\Vy$。
我们可以把矩阵乘积$\MC=\MA\MB$中计算$\SC_{i,j}$的步骤看作是$\MA$的第$\Si$行和$\MB$的第$\Sj$列之间的点积。


矩阵乘积运算有许多有用的性质，从而使矩阵的数学分析更加方便。
比如，矩阵乘积服从分配律：
\begin{equation}
    \MA(\MB+\MC)=\MA\MB +\MA\MC.
\end{equation}
矩阵乘积也服从结合律：
\begin{equation}
\MA(\MB\MC)=(\MA\MB)\MC.
\end{equation}


<!-- % -- 32 -- -->


不同于标量乘积，矩阵乘积\emph{并不}满足交换律（$\MA\MB=\MB\MA$的情况并非总是满足）。
然而，两个向量的点积满足交换律：
\begin{equation}
\Vx^\top\Vy=\Vy^\top\Vx.
\end{equation}


矩阵乘积的转置有着简单的形式：
\begin{equation}
(\MA\MB)^\top=\MB^\top\MA^\top.
\end{equation}
利用两个向量点积的结果是标量，标量转置是自身的事实，我们可以证明\eqn?：
\begin{equation}
    \Vx^\top \Vy = \left(\Vx^\top \Vy \right)^\top = \Vy^\top \Vx.
\end{equation}


由于本书的重点不是线性代数，我们并不试图展示矩阵乘积的所有重要性质，但读者应该知道矩阵乘积还有很多有用的性质。


现在我们已经知道了足够多的线性代数符号，可以表达下列线性方程组：
\begin{equation}
\MA\Vx=\Vb
\end{equation}
其中$\MA\in \SetR^{m\times n}$是一个已知矩阵，$\Vb\in\SetR^m$是一个已知向量，$\Vx\in\SetR^n$是一个我们要求解的未知向量。
向量$\Vx$的每一个元素$\Sx_i$都是未知的。
矩阵$\MA$的每一行和$\Vb$中对应的元素构成一个约束。
我们可以把\eqn?重写为
\begin{gather}
\MA_{1,:}\Vx=b_1\\
\MA_{2,:}\Vx=b_2 \\
\cdots \\
\MA_{m,:}\Vx=b_m
\end{gather}
或者，更明确地，写作
\begin{gather}
    \MA_{1,1}x_1+\MA_{1,2}x_2+\cdots \MA_{1,n}x_n = b_1\\
    \MA_{2,1}x_1+\MA_{2,2}x_2+\cdots \MA_{2,n}x_n = b_2\\
    \cdots\\
    \MA_{m,1}x_1+\MA_{m,2}x_2+\cdots \MA_{m,n}x_n = b_m.
\end{gather}


矩阵向量乘积符号为这种形式的方程提供了更紧凑的表示。



<!-- % -- 33 -- -->



# 单位矩阵和逆矩阵



线性代数提供了被称为矩阵逆的强大工具。
对于大多数矩阵$\MA$，我们都能通过矩阵逆解析地求解\eqn?。


为了描述矩阵逆，我们首先需要定义单位矩阵的概念。
任意向量和单位矩阵相乘，都不会改变。
我们将保持$\Sn$维向量不变的单位矩阵记作$\MI_{\Sn}$。
形式上，$\MI_{\Sn}\in \SetR^{\Sn\times \Sn}$，
\begin{equation}
    \forall \Vx \in \SetR^{\Sn}, \MI_{\Sn} \Vx = \Vx.
\end{equation}
单位矩阵的结构很简单：所有沿主对角线的元素都是$1$，而所有其他位置的元素都是$0$。
如\fig?所示。
\begin{figure}[!htb]
\ifOpenSource
\centerline{\includegraphics{figure.pdf}}
\else
\centering
\begin{equation*}
\begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
\end{equation*}
\fi
\caption{单位矩阵的一个样例：这是$\MI_3$。}
\end{figure}


矩阵$\MA$的矩阵逆记作$\MA^{-1}$，其定义的矩阵满足如下条件
\begin{equation} \MA^{-1}\MA = \MI_{\Sn}. \end{equation}

现在我们可以通过以下步骤求解\eqn?：
\begin{gather}
\MA\Vx=\Vb \\
\MA^{-1}\MA\Vx = \MA^{-1}\Vb \\
\MI_{\Sn} \Vx=\MA^{-1}\Vb \\
\Vx=\MA^{-1}\Vb. 
\end{gather}


当然，这取决于我们能否找到一个逆矩阵$\MA^{-1}$。
在接下来的章节中，我们会讨论逆矩阵$\MA^{-1}$存在的条件。


当逆矩阵$\MA^{-1}$存在时，有几种不同的算法都能找到它的闭解形式。
理论上，相同的逆矩阵可用于多次求解不同向量$\Vb$的方程。
然而，逆矩阵$\MA^{-1}$主要是作为理论工具使用的，并不会在大多数软件应用程序中实际使用。
这是因为逆矩阵$\MA^{-1}$在数字计算机上只能表现出有限的精度，有效使用向量$\Vb$的算法通常可以得到更精确的$\Vx$。



<!-- % -- 34 -- -->



# 线性相关和生成子空间


如果逆矩阵$\MA^{-1}$存在，那么\eqn?肯定对于每一个向量$\Vb$恰好存在一个解。
但是，对于方程组而言，对于向量$\Vb$的某些值，有可能不存在解，或者存在无限多个解。
存在多于一个解但是少于无限多个解的情况是不可能发生的；因为如果$\Vx$和$\Vy$都是某方程组的解，则
\begin{equation}
\Vz=\alpha \Vx + (1-\alpha) \Vy
\end{equation}
（其中$\alpha$取任意实数）也是该方程组的解。


为了分析方程有多少个解，我们可以将$\MA$的列向量看作从原点（元素都是零的向量）出发的不同方向，确定有多少种方法可以到达向量$\Vb$。
在这个观点下，向量$\Vx$中的每个元素表示我们应该沿着这些方向走多远，即$\Sx_{\Si}$表示我们需要沿着第$\Si$个向量的方向走多远：
\begin{equation}
\MA \Vx = \sum_i x_i \MA_{:,i}.
\end{equation}
一般而言，这种操作被称为线性组合。
形式上，一组向量的线性组合，是指每个向量乘以对应标量系数之后的和，即：
\begin{equation}
    \sum_i \Sc_i \Vv^{(i)}.
\end{equation}
一组向量的生成子空间是原始向量线性组合后所能抵达的点的集合。


确定$\MA\Vx=\Vb$是否有解相当于确定向量$\Vb$是否在$\MA$列向量的生成子空间中。
这个特殊的生成子空间被称为$\MA$的列空间或者$\MA$的值域。


为了使方程$\MA \Vx=\Vb$对于任意向量$\Vb \in \SetR^m$都存在解，我们要求$\MA$的列空间构成整个$\SetR^{\Sm}$。
如果$\SetR^m$中的某个点不在$\MA$的列空间中，那么该点对应的$\Vb$会使得该方程没有解。
矩阵$\MA$的列空间是整个$\SetR^m$的要求，意味着$\MA$至少有$m$列，即$n\geq m$。
否则，$\MA$列空间的维数会小于$m$。
例如，假设$\MA$是一个$3\times 2$的矩阵。
目标$\Vb$是$3$维的，但是$\Vx$只有$2$维。
所以无论如何修改$\Vx$的值，也只能描绘出$\SetR^3$空间中的二维平面。
当且仅当向量$\Vb$在该二维平面中时，该方程有解。



<!-- % -- 35 -- -->


不等式$n\geq m$仅是方程对每一点都有解的必要条件。
这不是一个充分条件，因为有些列向量可能是冗余的。
假设有一个$\SetR^{2\times 2}$中的矩阵，它的两个列向量是相同的。
那么它的列空间和它的一个列向量作为矩阵的列空间是一样的。
换言之，虽然该矩阵有$2$列，但是它的列空间仍然只是一条线，不能涵盖整个$\SetR^2$空间。


正式地说，这种冗余被称为线性相关。
如果一组向量中的任意一个向量都不能表示成其他向量的线性组合，那么这组向量称为线性无关。
如果某个向量是一组向量中某些向量的线性组合，那么我们将这个向量加入这组向量后不会增加这组向量的生成子空间。
这意味着，如果一个矩阵的列空间涵盖整个$\SetR^m$，那么该矩阵必须包含至少一组$m$个线性无关的向量。
这是\eqn?对于每一个向量$\Vb$的取值都有解的充分必要条件。
值得注意的是，这个条件是说该向量集恰好有$m$个线性无关的列向量，而不是至少$m$个。
不存在一个$m$维向量的集合具有多于$m$个彼此线性不相关的列向量，但是一个有多于$m$个列向量的矩阵有可能拥有不止一个大小为$m$的线性无关向量集。


要想使矩阵可逆，我们还需要保证\eqn?对于每一个$\Vb$值至多有一个解。
为此，我们需要确保该矩阵至多有$m$个列向量。
否则，该方程会有不止一个解。


综上所述，这意味着该矩阵必须是一个方阵，即$m=n$，并且所有列向量都是线性无关的。一个列向量线性相关的方阵被称为奇异的。


如果矩阵$\MA$不是一个方阵或者是一个奇异的方阵，该方程仍然可能有解。
但是我们不能使用矩阵逆去求解。


目前为止，我们已经讨论了逆矩阵左乘。我们也可以定义逆矩阵右乘：
\begin{equation}
\MA\MA^{-1}=\MI.
\end{equation}
对于方阵而言，它的左逆和右逆是相等的。





# 范数


有时我们需要衡量一个向量的大小。
在机器学习中，我们经常使用被称为范数的函数衡量向量大小。
形式上，$L^p$范数定义如下
\begin{equation}
    \norm{\Vx}_p = \left( \sum_i |x_i|^p \right)^{\frac{1}{p}}
\end{equation}
其中$p\in \SetR$，$p\geq 1$。



<!-- % -- 36 -- -->


范数（包括$L^p$范数）是将向量映射到非负值的函数。
直观上来说，向量$\Vx$的范数衡量从原点到点$\Vx$的距离。
更严格地说，范数是满足下列性质的任意函数：

+ $f(\Vx) = 0 \Rightarrow \Vx = \mathbf{0}$ 
+ $f(\Vx + \Vy) \leq f(\Vx) + f(\Vy)$ （三角不等式）
+ $\forall \alpha \in \SetR$, $f(\alpha \Vx) = |\alpha| f(\Vx)$



当$p=2$时，$L^2$范数被称为欧几里得范数。
它表示从原点出发到向量$\Vx$确定的点的欧几里得距离。
$L^2$范数在机器学习中出现地十分频繁，经常简化表示为$\norm{x}$，略去了下标$2$。
平方$L^2$范数也经常用来衡量向量的大小，可以简单地通过点积 $\Vx^\top\Vx$计算。


平方$L^2$范数在数学和计算上都比$L^2$范数本身更方便。
例如，平方$L^2$范数对$\Vx$中每个元素的导数只取决于对应的元素，而$L^2$范数对每个元素的导数却和整个向量相关。
但是在很多情况下，平方$L^2$范数也可能不受欢迎，因为它在原点附近增长得十分缓慢。
在某些机器学习应用中，区分恰好是零的元素和非零但值很小的元素是很重要的。
在这些情况下，我们转而使用在各个位置斜率相同，同时保持简单的数学形式的函数：$L^1$范数。
$L^1$范数可以简化如下：
\begin{equation}
    \norm{\Vx}_1 = \sum_i  |x_i|.
\end{equation}
当机器学习问题中零和非零元素之间的差异非常重要时，通常会使用$L^1$范数。
每当$\Vx$中某个元素从$0$增加$\epsilon$，对应的$L^1$范数也会增加$\epsilon$。


有时候我们会统计向量中非零元素的个数来衡量向量的大小。
有些作者将这种函数称为"$L^0$范数"，但是这个术语在数学意义上是不对的。
向量的非零元素的数目不是范数，因为对向量缩放$\alpha$倍不会改变该向量非零元素的数目。
$L^1$范数经常作为表示非零元素数目的替代函数。



<!-- % -- 37 -- -->


另外一个经常在机器学习中出现的范数是$L^\infty$范数，也被称为\,最大范数。
这个范数表示向量中具有最大幅值的元素的绝对值：
\begin{equation}
    \norm{\Vx}_\infty = \max_i |x_i|.
\end{equation}


有时候我们可能也希望衡量矩阵的大小。
在深度学习中，最常见的做法是使用Frobenius 范数，
\begin{equation}
    \norm{\MA}_F = \sqrt{\sum_{i,j} A_{i,j}^2}, 
<!-- %%lyj 原文是\norm{A}_F ... -->
\end{equation}
其类似于向量的$L^2$范数。


两个向量的点积可以用范数来表示。
具体地，
\begin{equation}
    \Vx^\top\Vy = \norm{\Vx}_2\norm{\Vy}_2 \cos \theta
\end{equation}
其中$\theta$表示$\Vx$和$\Vy$之间的夹角。





# 特殊类型的矩阵和向量


有些特殊类型的矩阵和向量是特别有用的。


对角矩阵只在主对角线上含有非零元素，其他位置都是零。
形式上，矩阵$\MD$是对角矩阵，当且仅当对于所有的$i\neq j$，$\SD_{i,j}=0$。
我们已经看到过一个对角矩阵：单位矩阵，对角元素全部是$1$。
我们用$\text{diag}(\Vv)$表示一个对角元素由向量$\Vv$中元素给定的对角方阵。
对角矩阵受到关注的部分原因是对角矩阵的乘法计算很高效。
计算乘法$\text{diag}(\Vv)\Vx$，我们只需要将$\Vx$中的每个元素$x_i$放大$v_i$倍。
换言之，$\text{diag}(\Vv)\Vx=\Vv \odot \Vx$。
计算对角方阵的逆矩阵也很高效。
对角方阵的逆矩阵存在，当且仅当对角元素都是非零值，在这种情况下，$\text{diag}(\Vv)^{-1}=\text{diag}([1/v_1,\dots,1/v_n]^\top)$。
在很多情况下，我们可以根据任意矩阵导出一些通用的机器学习算法；但通过将一些矩阵限制为对角矩阵，我们可以得到计算代价较低的（并且简明扼要的）算法。


不是所有的对角矩阵都是方阵。
长方形的矩阵也有可能是对角矩阵。
非方阵的对角矩阵没有逆矩阵，但我们仍然可以高效地计算它们的乘法。
对于一个长方形对角矩阵$\MD$而言，乘法$\MD\Vx$会涉及到$\Vx$中每个元素的缩放，如果$\MD$是瘦长型矩阵，那么在缩放后的末尾添加一些零；如果$\MD$是胖宽型矩阵，那么在缩放后去掉最后一些元素。


<!-- % -- 38 -- -->

对称矩阵是转置和自己相等的矩阵：
\begin{equation}
    \MA=\MA^\top.
\end{equation}
当某些不依赖参数顺序的双参数函数生成元素时，对称矩阵经常会出现。
例如，如果$\MA$是一个距离度量矩阵，$\MA_{i,j}$表示点$i$到点$j$的距离，那么$\MA_{i,j}=\MA_{j,i}$，因为距离函数是对称的。


单位向量是具有单位范数的向量：
\begin{equation}
\norm{\Vx}_2=1.
\end{equation}


如果$\Vx^\top \Vy = 0$，那么向量$\Vx$和向量$\Vy$互相正交。
如果两个向量都有非零范数，那么这两个向量之间的夹角是$90$度。
在$\SetR^n$中，至多有$n$个范数非零向量互相正交。
如果这些向量不仅互相正交，并且范数都为$1$，那么我们称它们是标准正交。


正交矩阵是指行向量和列向量是分别标准正交的方阵：
\begin{equation}
    \MA^\top\MA=\MA\MA^\top=\MI.
\end{equation}
这意味着 
\begin{equation}
    \MA^{-1}=\MA^\top,
\end{equation}
所以正交矩阵受到关注是因为求逆计算代价小。
我们需要注意正交矩阵的定义。
违反直觉的是，正交矩阵的行向量不仅是正交的，还是标准正交的。
对于行向量或列向量互相正交但不是标准正交的矩阵，没有对应的专有术语。





# 特征分解


许多数学对象可以通过将它们分解成多个组成部分或者找到它们的一些属性而更好地理解，这些属性是通用的，而不是由我们选择表示它们的方式产生的。

<!-- % -- 39 -- -->

例如，整数可以分解为质因数。
我们可以用十进制或二进制等不同方式表示整数$12$，但是$12=2\times 2\times 3$永远是对的。
从这个表示中我们可以获得一些有用的信息，比如$12$不能被$5$整除，或者$12$的倍数可以被$3$整除。


正如我们可以通过分解质因数来发现整数的一些内在性质，我们也可以通过分解矩阵来发现矩阵表示成数组元素时不明显的函数性质。


特征分解是使用最广的矩阵分解之一，即我们将矩阵分解成一组特征向量和特征值。


方阵$\MA$的特征向量是指与$\MA$相乘后相当于对该向量进行缩放的非零向量$\Vv$：
\begin{equation}
    \MA\Vv=\lambda \Vv.
\end{equation}
标量$\lambda$被称为这个特征向量对应的特征值。
（类似地，我们也可以定义左特征向量 $\Vv^\top\MA=\lambda \Vv^\top$，但是通常我们更关注右特征向量）。


如果$\Vv$是$\MA$的特征向量，那么任何缩放后的向量$s\Vv$~($s\in \SetR$，$s\neq 0$)也是$\MA$的特征向量。
此外，$s\Vv$和$\Vv$有相同的特征值。
基于这个原因，通常我们只考虑单位特征向量。


假设矩阵$\MA$有$n$个线性无关的特征向量$\{\Vv^{(1)}, \dots, \Vv^{(n)}\}$，对应着特征值$\{\lambda_1, \dots , \lambda_n \}$。
我们将特征向量连接成一个矩阵，使得每一列是一个特征向量：$\MV=[\Vv^{(1)}, \dots, \Vv^{(n)}]$.
类似地，我们也可以将特征值连接成一个向量$\Vlambda = [\lambda_1, \dots , \lambda_n]^\top$。
因此$\MA$的特征分解可以记作
\begin{equation}
    \MA = \MV \text{diag}(\Vlambda) \MV^{-1}.
\end{equation}


我们已经看到了\emph{构建}具有特定特征值和特征向量的矩阵，能够使我们在目标方向上延伸空间。
我们还常常希望将矩阵分解成特征值和特征向量。
这样可以帮助我们分析矩阵的特定性质，就像质因数分解有助于我们理解整数。


不是每一个矩阵都可以分解成特征值和特征向量。
在某些情况下，特征分解存在，但是会涉及复数而非实数。
幸运的是，在本书中，我们通常只需要分解一类有简单分解的矩阵。
具体来讲，每个实对称矩阵都可以分解成实特征向量和实特征值：
\begin{equation}
    \MA = \MQ \VLambda \MQ^\top.
\end{equation}
其中$\MQ$是$\MA$的特征向量组成的正交矩阵，$\VLambda$是对角矩阵。
特征值$\Lambda_{i,i}$对应的特征向量是矩阵$\MQ$的第$i$列，记作$\MQ_{:,i}$。
因为$\MQ$是正交矩阵，我们可以将$\MA$看作沿方向$\Vv^{(i)}$延展$\lambda_i$倍的空间。
如\fig?所示的例子。

<!-- % -- 40 -- -->

虽然任意一个实对称矩阵$\MA$都有特征分解，但是特征分解可能并不唯一。
如果两个或多个特征向量拥有相同的特征值，那么在由这些特征向量产生的生成子空间中，任意一组正交向量都是该特征值对应的特征向量。
因此，我们可以等价地从这些特征向量中构成$\MQ$作为替代。
按照惯例，我们通常按降序排列$\VLambda$的元素。
在该约定下，特征分解唯一当且仅当所有的特征值都是唯一的。

\begin{figure}[!htb]
\ifOpenSource
\centerline{\includegraphics{figure.pdf}}
\else
\centerline{\includegraphics[width=0.8\textwidth]{Chapter2/figures/eigen_ellipse_color}}
\fi
\caption{特征向量和特征值的作用效果。
特征向量和特征值的作用效果的一个实例。
在这里，矩阵$\MA$有两个标准正交的特征向量，对应特征值为$\lambda_1$的$\Vv^{(1)}$以及对应特征值为$\lambda_2$的$\Vv^{(2)}$。
\emph{(左)}我们画出了所有的单位向量$\Vu\in\SetR^2$的集合，构成一个单位圆。
\emph{(右)}我们画出了所有的$\MA\Vu$点的集合。
通过观察$\MA$拉伸单位圆的方式，我们可以看到它将$\Vv^{(i)}$方向的空间拉伸了$\lambda_i$倍。	}
\end{figure}

<!-- % -- 41 -- -->


矩阵的特征分解给了我们很多关于矩阵的有用信息。
矩阵是奇异的当且仅当含有零特征值。
实对称矩阵的特征分解也可以用于优化二次方程$f(\Vx) = \Vx^\top \MA \Vx$，其中限制$\norm{\Vx}_2 = 1$。
当$\Vx$等于$\MA$的某个特征向量时，$f$将返回对应的特征值。
在限制条件下，函数$f$的最大值是最大特征值，最小值是最小特征值。


所有特征值都是正数的矩阵被称为正定；所有特征值都是非负数的矩阵被称为半正定。
同样地，所有特征值都是负数的矩阵被称为负定；所有特征值都是非正数的矩阵被称为半负定。
半正定矩阵受到关注是因为它们保证$\forall \Vx, \Vx^\top \MA \Vx \geq 0$。
此外，正定矩阵还保证$\Vx^\top \MA \Vx =0 \Rightarrow \Vx = \mathbf{0}$。






# 奇异值分解


在\sec?，我们探讨了如何将矩阵分解成特征向量和特征值。
还有另一种分解矩阵的方法，被称为奇异值分解，将矩阵分解为奇异向量和奇异值。
通过奇异值分解，我们会得到一些与特征分解相同类型的信息。
然而，奇异值分解有更广泛的应用。
每个实数矩阵都有一个奇异值分解，但不一定都有特征分解。
例如，非方阵的矩阵没有特征分解，这时我们只能使用奇异值分解。


回想一下， 我们使用特征分解去分析矩阵$\MA$时， 得到特征向量构成的矩阵$\MV$和特征值构成的向量$\Vlambda$，我们可以重新将$\MA$写作
\begin{equation}
    \MA=\MV\text{diag}(\Vlambda)\MV^{-1}.
\end{equation}

奇异值分解是类似的，只不过这回我们将矩阵$\MA$分解成三个矩阵的乘积：
\begin{equation}
    \MA=\MU\MD\MV^\top.
\end{equation}


假设$\MA$是一个$m\times n$的矩阵，那么$\MU$是一个$m\times m$的矩阵，$\MD$是一个$m\times n$的矩阵，$\MV$是一个$n\times n$矩阵。


<!-- % -- 42 -- -->

这些矩阵中的每一个经定义后都拥有特殊的结构。
矩阵$\MU$和$\MV$都定义为正交矩阵，而矩阵$\MD$定义为对角矩阵。
注意，矩阵$\MD$不一定是方阵。


对角矩阵$\MD$对角线上的元素被称为矩阵$\MA$的奇异值。
矩阵$\MU$的列向量被称为左奇异向量，矩阵$\MV$的列向量被称右奇异向量。


事实上，我们可以用与$\MA$相关的特征分解去解释$\MA$的奇异值分解。
$\MA$的左奇异向量是$\MA\MA^\top$的特征向量。
$\MA$的右奇异向量是$\MA^\top\MA$的特征向量。
$\MA$的非零奇异值是$\MA^\top\MA$特征值的平方根，同时也是$\MA\MA^\top$特征值的平方根。


SVD\,最有用的一个性质可能是拓展矩阵求逆到非方矩阵上。我们将在下一节中探讨。





# Moore-Penrose 伪逆



对于非方矩阵而言，其逆矩阵没有定义。
假设在下面的问题中，我们希望通过矩阵$\MA$的左逆$\MB$来求解线性方程，
\begin{equation}
    \MA\Vx=\Vy
\end{equation}
等式两边左乘左逆$\MB$后，我们得到
\begin{equation}
    \Vx=\MB\Vy.
\end{equation}
取决于问题的形式，我们可能无法设计一个唯一的映射将$\MA$映射到$\MB$。


如果矩阵$\MA$的行数大于列数，那么上述方程可能没有解。
如果矩阵$\MA$的行数小于列数，那么上述矩阵可能有多个解。


Moore-Penrose 伪逆使我们在这类问题上取得了一定的进展。
矩阵$\MA$的伪逆定义为：
\begin{equation}
    \MA^+ = \lim_{\alpha \searrow 0} (\MA^\top\MA + \alpha \MI)^{-1} \MA^\top.
\end{equation}
计算伪逆的实际算法没有基于这个定义，而是使用下面的公式：
\begin{equation}
    \MA^+ = \MV\MD^+\MU^\top.
\end{equation}
其中，矩阵$\MU$，$\MD$和$\MV$是矩阵$\MA$奇异值分解后得到的矩阵。
对角矩阵$\MD$的伪逆$\MD^+$是其非零元素取倒数之后再转置得到的。

<!-- % -- 43 -- -->

当矩阵$\MA$的列数多于行数时，使用伪逆求解线性方程是众多可能解法中的一种。
特别地，$\Vx=\MA^+\Vy$是方程所有可行解中欧几里得范数$\norm{\Vx}_2$最小的一个。


当矩阵$\MA$的行数多于列数时，可能没有解。
在这种情况下，通过伪逆得到的$\Vx$使得$\MA\Vx$和$\Vy$的欧几里得距离$\norm{\MA\Vx-\Vy}_2$最小。






# 迹运算


迹运算返回的是矩阵对角元素的和：
\begin{equation}
    \Tr(\MA)= \sum_i \MA_{i,i}.
\end{equation}
迹运算因为很多原因而有用。
若不使用求和符号，有些矩阵运算很难描述，而通过矩阵乘法和迹运算符号可以清楚地表示。
例如，迹运算提供了另一种描述矩阵Frobenius 范数的方式：
\begin{equation}
    \norm{A}_F = \sqrt{\text{Tr}(\MA \MA^\top)}.
\end{equation}


用迹运算表示表达式，我们可以使用很多有用的等式巧妙地处理表达式。
例如，迹运算在转置运算下是不变的：
\begin{equation}
    \Tr(\MA)=\Tr(\MA^\top).
\end{equation}


多个矩阵相乘得到的方阵的迹，和将这些矩阵中的最后一个挪到最前面之后相乘的迹是相同的。
当然，我们需要考虑挪动之后矩阵乘积依然定义良好：
\begin{equation}
\Tr(\MA\MB\MC)=\Tr(\MC\MA\MB)= \Tr(\MB\MC\MA).
\end{equation}
或者更一般地，
\begin{equation} 
\Tr(\prod_{i=1}^n \MF^{(i)})= \Tr(\MF^{(n)} \prod_{i=1}^{n-1} \MF^{(i)}).
\end{equation}
即使循环置换后矩阵乘积得到的矩阵形状变了，迹运算的结果依然不变。
例如，假设矩阵$\MA\in \SetR^{m\times n}$，矩阵$\MB\in \SetR^{n\times m}$，我们可以得到
\begin{equation} 
    \Tr(\MA\MB)= \Tr(\MB\MA)
\end{equation}
尽管$\MA\MB \in \SetR^{m\times m}$和$\MB\MA \in \SetR^{n\times n}$。


<!-- % -- 44 -- -->

另一个有用的事实是标量在迹运算后仍然是它自己：$a=\Tr(a)$。





# 行列式


行列式，记作$\text{det}(\MA)$，是一个将方阵$\MA$映射到实数的函数。
行列式等于矩阵特征值的乘积。
行列式的绝对值可以用来衡量矩阵参与矩阵乘法后空间扩大或者缩小了多少。
如果行列式是$0$，那么空间至少沿着某一维完全收缩了，使其失去了所有的体积。
如果行列式是$1$，那么这个转换保持空间体积不变。





# 实例：主成分分析


主成分分析是一个简单的机器学习算法，可以通过基础的线性代数知识推导。


假设在$\SetR^n$空间中我们有$m$个点$\{\Vx^{(1)}, \dots ,\Vx^{(m)}\}$，我们希望对这些点进行有损压缩。
有损压缩表示我们使用更少的内存，但损失一些精度去存储这些点。
我们希望损失的精度尽可能少。


一种编码这些点的方式是用低维表示。
对于每个点$\Vx^{(i)} \in \SetR^n$，会有一个对应的编码向量$\Vc^{(i)}\in \SetR^l$。
如果$l$比$n$小，那么我们便使用了更少的内存来存储原来的数据。
我们希望找到一个编码函数，根据输入返回编码，$f(\Vx)=\Vc$；我们也希望找到一个解码函数，给定编码重构输入，$\Vx\approx g(f(\Vx))$。


PCA~由我们选择的解码函数而定。
具体地，为了简化解码器，我们使用矩阵乘法将编码映射回$\SetR^n$，即$g(\Vc)=\MD\Vc$，其中$\MD\in \SetR^{n\times l}$是定义解码的矩阵。

<!-- % -- 45 -- -->

目前为止所描述的问题，可能会有多个解。
因为如果我们按比例地缩小所有点对应的编码向量$c_i$，那么我们只需按比例放大$\MD_{:,i}$，即可保持结果不变。
为了使问题有唯一解，我们限制$\MD$中所有列向量都有单位范数。


计算这个解码器的最优编码可能是一个困难的问题。
为了使编码问题简单一些，PCA\,限制$\MD$的列向量彼此正交（注意，除非$l=n$，否则严格意义上$\MD$不是一个正交矩阵）。


为了将这个基本想法变为我们能够实现的算法，首先我们需要明确如何根据每一个输入$\Vx$得到一个最优编码$\Vc^*$。
一种方法是最小化原始输入向量$\Vx$和重构向量$g(\Vc^*)$之间的距离。
我们使用范数来衡量它们之间的距离。
在\,PCA\,算法中，我们使用$L^2$范数：
\begin{equation}
 \Vc^* = \underset{\Vc}{\arg\min} \norm{\Vx-g(\Vc)}_2.
\end{equation}


我们可以用平方$L^2$范数替代$L^2$范数，因为两者在相同的值$\Vc$上取得最小值。
这是因为$L^2$范数是非负的，并且平方运算在非负值上是单调递增的。
\begin{equation}
\Vc^* = \argmin_{\Vc} \norm{\Vx - g(\Vc)}_2^2.
\end{equation}
该最小化函数可以简化成
\begin{equation}
(\Vx-g(\Vc))^\top(\Vx-g(\Vc))
\end{equation}
（\eqn?中$L^2$范数的定义）
\begin{equation}
    = \Vx^\top\Vx - \Vx^\top g(\Vc) - g(\Vc)^\top\Vx + g(\Vc)^\top g(\Vc)
\end{equation}
(分配律)
\begin{equation}
    = \Vx^\top \Vx - 2\Vx^\top g(\Vc) + g(\Vc)^\top g(\Vc)
\end{equation}
(因为标量$g(\Vc)^\top\Vx$的转置等于自己)


因为第一项$\Vx^\top\Vx$不依赖于$\Vc$，所以我们可以忽略它，得到如下的优化目标：
\begin{equation}
\Vc^* = \underset{\Vc}{\arg\min} - 2\Vx^\top g(\Vc) + g(\Vc)^\top g(\Vc).
\end{equation}

<!-- % -- 46 -- -->

更进一步，我们代入$g(\Vc)$的定义：
\begin{equation}
    \Vc^* = \underset{\Vc}{\arg\min} - 2\Vx^\top\MD\Vc + \Vc^\top\MD^\top\MD\Vc
\end{equation}
\begin{equation}
    = \underset{\Vc}{\arg\min} -2\Vx^\top\MD\Vc + \Vc^\top\MI_l\Vc
\end{equation}
(矩阵$\MD$的正交性和单位范数约束)
\begin{equation}
    = \underset{\Vc}{\arg\min} -2\Vx^\top\MD\Vc + \Vc^\top\Vc
\end{equation}


我们可以通过向量微积分来求解这个最优化问题（如果你不清楚怎么做，请参考\sec?）
\begin{gather}
    \nabla_{\Vc} (-2\Vx^\top \MD \Vc + \Vc^\top\Vc) = 0\\
    -2\MD^\top\Vx + 2\Vc = 0\\
    \Vc = \MD^\top \Vx.
\end{gather}


这使得算法很高效：最优编码$\Vx$只需要一个矩阵-向量乘法操作。
为了编码向量，我们使用编码函数：
\begin{equation}
    f(\Vx)=\MD^\top\Vx.
\end{equation}
进一步使用矩阵乘法，我们也可以定义\,PCA\,重构操作：
\begin{equation}
    r(\Vx)=g(f(\Vx)) = \MD\MD^\top \Vx.
\end{equation}


接下来，我们需要挑选编码矩阵$\MD$。
要做到这一点，我们回顾最小化输入和重构之间$L^2$距离的这个想法。
因为用相同的矩阵$\MD$对所有点进行解码，我们不能再孤立地看待每个点。
反之，我们必须最小化所有维数和所有点上的误差矩阵的\,Frobenius 范数：
\begin{equation}
    \MD^* =  \underset{\MD}{\arg\min} \sqrt{\sum_{i,j}\left( \Vx_j^{(i)} - r(\Vx^{(i)})_j\right)^2} \text{ subject to } \MD^\top\MD = \MI_l.
\end{equation}

为了推导用于寻求$\MD^*$的算法，我们首先考虑$l=1$的情况。
在这种情况下，$\MD$是一个单一向量$\Vd$。
将\eqn?代入\eqn?，简化$\MD$为$\Vd$，问题简化为
\begin{equation}
    \Vd^* = \underset{\Vd}{\arg\min} \sum_i \norm{\Vx^{(i)} - \Vd\Vd^\top \Vx^{(i)}}_2^2
    \text{ subject to } \norm{\Vd}_2 = 1.
\end{equation}

<!-- % -- 47 -- -->

上述公式是直接代入得到的，但不是文体表述最舒服的方式。
在上述公式中，我们将标量$\Vd^\top\Vx^{(i)}$放在向量$\Vd$的右边。
将该标量放在左边的写法更为传统。
于是我们通常写作
\begin{equation}
    \Vd^* = \underset{\Vd}{\arg\min} \sum_i \norm{\Vx^{(i)} - \Vd^\top \Vx^{(i)}\Vd}_2^2
        \text{ subject to } \norm{\Vd}_2 = 1,
\end{equation}
或者，考虑到标量的转置和自身相等，我们也可以写作
\begin{equation}
    \Vd^* = \underset{\Vd}{\arg\min} \sum_i \norm{\Vx^{(i)} - \Vx^{(i)\top}\Vd\Vd}_2^2
        \text{ subject to } \norm{\Vd}_2 = 1.
\end{equation}
读者应该对这些重排写法慢慢熟悉起来。


此时，使用单一矩阵来重述问题，比将问题写成求和形式更有帮助。
这有助于我们使用更紧凑的符号。
将表示各点的向量堆叠成一个矩阵，记为$\MX\in\SetR^{m\times n}$，其中$\MX_{i,:}=\Vx^{(i)^\top}$。
原问题可以重新表述为：
\begin{equation}
    \Vd^* = \underset{\Vd}{\arg\min} \norm{\MX - \MX\Vd\Vd^\top}_F^2
        \text{ subject to } \Vd^\top \Vd = 1.
\end{equation}
暂时不考虑约束，我们可以将\,Frobenius 范数简化成下面的形式：
\begin{equation}
     \underset{\Vd}{\arg\min} \norm{\MX - \MX \Vd\Vd^\top}_F^2
\end{equation}
\begin{equation}
    = \underset{\Vd}{\arg\min} \, \Tr \left( \left( \MX - \MX \Vd\Vd^\top  \right)^\top \left( \MX - \MX \Vd\Vd^\top  \right) \right)
\end{equation}
（\eqn?）
\begin{equation}
    = \underset{\Vd}{\arg\min} \, \Tr \left( \MX^\top\MX - \MX^\top\MX \Vd\Vd^\top - \Vd\Vd^\top \MX^\top\MX + \Vd\Vd^\top \MX^\top\MX\Vd\Vd^\top  \right)
\end{equation}
\begin{equation}
    = \underset{\Vd}{\arg\min} \, \Tr( \MX^\top\MX)  - \Tr(\MX^\top\MX \Vd\Vd^\top)  - \Tr(\Vd\Vd^\top \MX^\top\MX) + \Tr(\Vd\Vd^\top \MX^\top\MX\Vd\Vd^\top)
\end{equation}
\begin{equation}
    = \underset{\Vd}{\arg\min} \, - \Tr(\MX^\top\MX \Vd\Vd^\top)  - \Tr(\Vd\Vd^\top \MX^\top\MX) + \Tr(\Vd\Vd^\top \MX^\top\MX\Vd\Vd^\top)
\end{equation}
（因为与$\Vd$无关的项不影响$\arg\min$）
\begin{equation}
    = \underset{\Vd}{\arg\min} \, - 2\Tr(\MX^\top\MX \Vd\Vd^\top) + \Tr(\Vd\Vd^\top \MX^\top\MX\Vd\Vd^\top)
\end{equation}
（因为循环改变迹运算中相乘矩阵的顺序不影响结果，如\eqn?所示）
\begin{equation}
    = \underset{\Vd}{\arg\min} \, - 2\Tr(\MX^\top\MX \Vd\Vd^\top) + \Tr(\MX^\top\MX\Vd\Vd^\top\Vd\Vd^\top )
\end{equation}
（再次使用上述性质）

<!-- % -- 48 -- -->

此时，我们再来考虑约束条件:
\begin{equation}
    \underset{\Vd}{\arg\min} \, - 2\Tr(\MX^\top\MX \Vd\Vd^\top) + \Tr(\MX^\top\MX\Vd\Vd^\top\Vd\Vd^\top )
    \text{ subject to } \Vd^\top \Vd = 1
\end{equation}
\begin{equation}
    = \underset{\Vd}{\arg\min} \, - 2\Tr(\MX^\top\MX \Vd \Vd^\top) + \Tr(\MX^\top\MX\Vd\Vd^\top )
    \text{ subject to } \Vd^\top \Vd = 1
\end{equation}
 (因为约束条件)
 \begin{equation}
     = \underset{\Vd}{\arg\min} \, - \Tr(\MX^\top\MX \Vd\Vd^\top)
     \text{ subject to } \Vd^\top \Vd = 1
 \end{equation}
 \begin{equation}
     = \underset{\Vd}{\arg\max} \, \Tr(\MX^\top\MX \Vd\Vd^\top)
     \text{ subject to } \Vd^\top \Vd = 1
 \end{equation}
 \begin{equation}
     = \underset{\Vd}{\arg\max} \, \Tr(\Vd^\top\MX^\top\MX \Vd)
     \text{ subject to } \Vd^\top \Vd = 1 .
 \end{equation}


这个优化问题可以通过特征分解来求解。
具体来讲，最优的$\Vd$是$\MX^\top\MX$最大特征值对应的特征向量。


以上推导特定于$l=1$的情况， 仅得到了第一个主成分。
更一般地，当我们希望得到主成分的基时，矩阵$\MD$由前$l$个最大的特征值对应的特征向量组成。
这个结论可以通过归纳法证明，我们建议将此证明作为练习。


线性代数是理解深度学习所必须掌握的基础数学学科之一。
另一门在机器学习中无处不在的重要数学学科是概率论，我们将在下一章探讨。



<!-- % -- 49 -- -->

