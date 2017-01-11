# Deep Learning 中文翻译

公开1个多月，在众多网友的帮助下，草稿中的草稿慢慢变成了草稿。原本打算我们翻译人员先相互校对一遍再让网友校对，但由于时间不足，于是更改计划，打算一起校对。

[直译版](https://github.com/exacity/deeplearningbook-chinese/tree/literal)逐渐向意译版过渡，我们希望尽可能地保留原书[Deep Learning](http://www.deeplearningbook.org/)中的意思并保留原书的语句。
然而我们水平有限，哈姆雷特成千上万，我们无法消除众多读者的方差。我们需要大家的建议和帮助，一起减小翻译的偏差。

对应的翻译者：
  - 第1、4、7、10、14、20章及第12.4、12.5节由 @swordyork 负责
  - 第2、5、8、11、15、18章由 @liber145 负责
  - 第3、6、9章由 @KevinLee1110 负责
  - 第13、16、17、19章及第12.1至12.3节由 @futianfan 负责



面向的读者
--------------------

请直接下载[PDF](https://github.com/exacity/deeplearningbook-chinese/releases/download/v0.4-alpha/dlbook_cn_v0.4-alpha.pdf)阅读。
虽然这一版准确性有所提高，但我们仍然建议英文好的同学或研究者直接阅读[原版](http://www.deeplearningbook.org/)。



校对认领
--------------------

我们划分4个类别的校对人员。每个类别需要很多人。
 - 负责人也就是对应的翻译者。
 - 我们需要有人简单地阅读，最好是刚入门或者想入门的同学。有什么翻得不明白的地方可以直接指出，不用给出意见。或者可以对语句不通顺的地方提出修改意见。每章至少3人。
 - 我们也需要有人进行中英对比，最好能排除少翻错翻的情况。最好是时间充足、能中英对应阅读、细心的同学。每章至少2人。
 - 专业人士则需要纠正译者的错误理解。大家不要谦虚，阅读过相关论文的同学可以作为专业人士。每章至少1人。

我们会在纸质版正式出版的时候，在书中致谢，正式感谢各位作出贡献的同学！

我们采用网上批注的形式。但转成markdown后，图片和某些公式会错误，排版也会出问题。希望大家谅解。

| 章节 | 负责人 | 简单阅读 | 中英对比 | 专业人士 |
| ------------ | ------------- | ---------- |  ------------ | --------- |
| [第一章 前言](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter1_introduction/) | @swordyork | liu chang | @linzhp |  |
| [第二章 线性代数](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter2_linear_algebra/) | @liber145 | |  |  |
| [第三章 概率与信息论](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter3_probability_and_information_theory/) | @KevinLee1110 | |  |  |
| [第四章 数值计算](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter4_numerical_computation/) | @swordyork | |  |  |
| [第五章 机器学习基础](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter5_machine_learning_basics/) | @liber145 | |  |  |
| [第六章 深度前馈网络](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter6_deep_feedforward_networks/) | @KevinLee1110 |  |  |  |
| [第七章 深度学习的正则化](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter7_regularization/) | @swordyork | |  |  |
| [第八章 深度模型中的优化](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter8_optimization_for_training_deep_models/) | @liber145 | |  |  |
| [第九章 卷积神经网络](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter9_convolutional_networks/) | @KevinLee1110 |  | @zhiding |  |
| [第十章 序列建模：循环和递归网络](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter10_sequence_modeling_rnn/) | @swordyork | liu chang |  |  |
| [第十一章 实用方法](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter11_practical_methodology/) | @liber145 | |  |  |
| [第十二章 应用](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter12_applications/) | @swordyork , @futianfan | |  |  |
| [第十三章 线性因子模型](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter13_linear_factor_models/) | @futianfan | |  |  |
| [第十四章 自动编码器](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter14_autoencoders/) | @swordyork | |  |  |
| [第十五章 表示学习](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter15_representation_learning/) | @liber145 |  |  |  |
| [第十六章 深度学习中的结构化概率模型](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter16_structured_probabilistic_modelling/) | @futianfan | |  |  |
| [第十七章 蒙特卡罗方法](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter17_monte_carlo_methods/) | @futianfan | |  |  |
| [第十八章 面对配分函数](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter18_confronting_the_partition_function/) | @liber145 | |  |  |
| [第十九章 近似推断](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter19_approximate_inference/) | @futianfan | |  |  |
| [第二十章 深度生成模型](https://via.hypothes.is/https://exacity.github.io/deeplearningbook-chinese/Chapter20_deep_generative_models/) | @swordyork | |  |  |



TODO
---------

 1. 语句通顺
 2. 排版，见issue [#35](https://github.com/exacity/deeplearningbook-chinese/issues/35)

实在有问题，请发邮件至`echo c3dvcmQueW9ya0BnbWFpbC5jb20K | base64 -d`。



致谢
---------

如果我们采用了大家的建议，我们会列在这。具体见[acknowledgments_github.md](https://github.com/exacity/deeplearningbook-chinese/blob/master/acknowledgments_github.md)。

@tttwwy @tankeco @fairmiracle @GageGao @huangpingchun @MaHongP @acgtyrant @yanhuibin315 @Buttonwood @titicacafz 
@weijy026a @RuiZhang1993 @zymiboxpay @xingkongliang @oisc @tielei @yuduowu @Qingmu @HC-2016 @xiaomingabc 
@bengordai @Bojian @JoyFYan @minoriwww @khty2000 @gump88 @zdx3578 @PassStory @imwebson @wlbksy @roachsinai @Elvinczp 
@endymecy name:YUE-DaJiong @9578577 @linzhp


注意
-----------

 - 各种问题或者建议可以提issue，建议使用中文。 
 - 由于版权问题，我们不能将图片和bib上传，请见谅。
 - Due to copyright issues, we would not upload figures and the bib file.
 - 可用于学习研究目的，不得用于任何商业行为。谢谢！
 - 大约每周release一个版本，[PDF](https://github.com/exacity/deeplearningbook-chinese/releases/download/v0.4-alpha/dlbook_cn_v0.4-alpha.pdf)文件每天更新。
 - 大家不要watch啊，邮箱可能会炸。
 - **先不要打印，这一版不值得打印，浪费钱** 打印版仅供学习参考和找茬纠错，正式出版后，希望大家多多支持纸质正版书籍。



笔记
---------
为帮助小白学得轻松一点，希望大家多多贡献笔记，单靠我们估计要大半年才能写出不错的笔记，而且时间不允许。
笔记见各章文件夹内的`README.md`。



Markdown格式
------------
这种格式确实比较重要，方便查阅，也方便索引。初步转换后，生成网页，具体见[deeplearningbook-chinese](https://exacity.github.io/deeplearningbook-chinese)。
注意，这种转换没有把图放进去，也不会放图。目前使用单个[脚本](scripts/convert2md.sh)，基于latex文件转换，以后可能会更改但原则是不直接修改[md文件](docs/_posts)。
需要的同学可以自行修改[脚本](scripts/convert2md.sh)。



Updating.....
