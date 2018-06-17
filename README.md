# Deep Learning 中文翻译

在众多网友的帮助和校对下，中文版终于出版了。尽管还有很多问题，但至少90%的内容是可读的，并且是准确的。
我们尽可能地保留了原书[Deep Learning](http://www.deeplearningbook.org/)中的意思并保留原书的语句。

然而我们水平有限，我们无法消除众多读者的方差。我们仍需要大家的建议和帮助，一起减小翻译的偏差。

大家所要做的就是阅读，然后汇总你的建议，提issue（最好不要一个一个地提）。如果你确定你的建议不需要商量，可以直接发起PR。

对应的翻译者：
  - 第1、4、7、10、14、20章及第12.4、12.5节由 @swordyork 负责
  - 第2、5、8、11、15、18章由 @liber145 负责
  - 第3、6、9章由 @KevinLee1110 负责
  - 第13、16、17、19章及第12.1至12.3节由 @futianfan 负责



面向的读者
--------------------

请直接下载[PDF](https://github.com/exacity/deeplearningbook-chinese/releases/download/v0.5-beta/dlbook_cn_v0.5-beta.pdf)阅读。
不打算提供EPUB等格式，如有需要请自行修改。

这一版准确性已经有所提高，读者可以以中文版为主、英文版为辅来阅读学习，但我们仍建议研究者阅读[原版](http://www.deeplearningbook.org/)。



出版及开源原因
--------------------

本书由人民邮电出版社出版，如果你觉得中文版PDF对你有所帮助，希望你能支持下纸质正版书籍。
如果你觉得中文版不行，希望你能多提建议。非常感谢各位！
纸质版也会进一步更新，需要大家更多的建议和意见，一起完善中文版。

纸质版目前在人民邮电出版社的异步社区出售，见[地址](http://www.epubit.com.cn/book/details/4278)。
价格不低，但看了样本之后，我们认为物有所值。
注意，我们不会通过媒体进行宣传，希望大家先看电子版内容，再判断是否购买纸质版。


以下是开源的具体原因：

 1. 我们不是文学工作者，不专职翻译。单靠我们，无法给出今天的翻译，众多网友都给我们提出了宝贵的建议，因此开源帮了很大的忙。出版社会给我们稿费（我们也不知道多少，可能2万左右），我们也不好意思自己用，商量之后觉得捐出是最合适的，以所有贡献过的网友的名义（我们把稿费捐给了杉树公益，用于4名贵州高中生三年的生活费，见[捐赠情况](https://github.com/exacity/deeplearningbook-chinese/blob/master/donation.pdf)）。
 2. PDF电子版对于技术类书籍来说是很重要的，随时需要查询，拿着纸质版到处走显然不合适。国外很多技术书籍都有对应的电子版（虽然不一定是正版），而国内的几乎没有。个人认为这是出版社或者作者认为国民素质还没有高到主动为知识付费的境界，所以不愿意"泄露"电子版。时代在进步，我们也需要改变。特别是翻译作品普遍质量不高的情况下，要敢为天下先。
 3. 深度学习发展太快，日新月异，所以我们希望大家更早地学到相关的知识。我觉得原作者开放PDF电子版也有类似的考虑，也就是先阅读后付费。我们认为中国人口素质已经足够高，懂得为知识付费。当然这不是付给我们的，是付给出版社的，出版社再付给原作者。我们不希望中文版的销量因PDF电子版的存在而下滑。出版社只有值回了版权才能在以后引进更多的优秀书籍。我们这个开源翻译先例也不会成为一个反面案例，以后才会有更多的PDF电子版。
 4. 开源也涉及版权问题，出于版权原因，我们不再更新此初版PDF文件，请大家以最终的纸质版为准。（但源码会一直更新）



致谢
--------------------

我们有3个类别的校对人员。
 - 负责人也就是对应的翻译者。
 - 简单阅读，对语句不通顺或难以理解的地方提出修改意见。
 - 中英对比，进行中英对应阅读，排除少翻错翻的情况。

所有校对建议都保存在各章的`annotations.txt`文件中。

| 章节 | 负责人 | 简单阅读 | 中英对比 |
| ------------ | ------------ | ------------ | ------------ |
| [第一章 前言](https://exacity.github.io/deeplearningbook-chinese/Chapter1_introduction/) | @swordyork | lc, @SiriusXDJ, @corenel, @NeutronT | @linzhp |
| [第二章 线性代数](https://exacity.github.io/deeplearningbook-chinese/Chapter2_linear_algebra/) | @liber145 | @SiriusXDJ, @angrymidiao | @badpoem |
| [第三章 概率与信息论](https://exacity.github.io/deeplearningbook-chinese/Chapter3_probability_and_information_theory/) | @KevinLee1110 | @SiriusXDJ | @kkpoker, @Peiyan |
| [第四章 数值计算](https://exacity.github.io/deeplearningbook-chinese/Chapter4_numerical_computation/) | @swordyork | @zhangyafeikimi | @hengqujushi |
| [第五章 机器学习基础](https://exacity.github.io/deeplearningbook-chinese/Chapter5_machine_learning_basics/) | @liber145 | @wheaio, @huangpingchun | @fairmiracle, @linzhp |
| [第六章 深度前馈网络](https://exacity.github.io/deeplearningbook-chinese/Chapter6_deep_feedforward_networks/) | @KevinLee1110 | David_Chow, @linzhp, @sailordiary |  |
| [第七章 深度学习中的正则化](https://exacity.github.io/deeplearningbook-chinese/Chapter7_regularization/) | @swordyork | | @NBZCC |
| [第八章 深度模型中的优化](https://exacity.github.io/deeplearningbook-chinese/Chapter8_optimization_for_training_deep_models/) | @liber145 | @happynoom, @codeVerySlow |  @huangpingchun |
| [第九章 卷积网络](https://exacity.github.io/deeplearningbook-chinese/Chapter9_convolutional_networks/) | @KevinLee1110 | @zhaoyu611, @corenel | @zhiding |
| [第十章 序列建模：循环和递归网络](https://exacity.github.io/deeplearningbook-chinese/Chapter10_sequence_modeling_rnn/) | @swordyork | lc | @zhaoyu611, @yinruiqing |
| [第十一章 实践方法论](https://exacity.github.io/deeplearningbook-chinese/Chapter11_practical_methodology/) | @liber145 |  |  |
| [第十二章 应用](https://exacity.github.io/deeplearningbook-chinese/Chapter12_applications/) | @swordyork, @futianfan |  | @corenel |
| [第十三章 线性因子模型](https://exacity.github.io/deeplearningbook-chinese/Chapter13_linear_factor_models/) | @futianfan | @cloudygoose | @ZhiweiYang |
| [第十四章 自编码器](https://exacity.github.io/deeplearningbook-chinese/Chapter14_autoencoders/) | @swordyork |  | @Seaball, @huangpingchun |
| [第十五章 表示学习](https://exacity.github.io/deeplearningbook-chinese/Chapter15_representation_learning/) | @liber145 | @cnscottzheng | |
| [第十六章 深度学习中的结构化概率模型](https://exacity.github.io/deeplearningbook-chinese/Chapter16_structured_probabilistic_modelling/) | @futianfan | |
| [第十七章 蒙特卡罗方法](https://exacity.github.io/deeplearningbook-chinese/Chapter17_monte_carlo_methods/) | @futianfan |  | @sailordiary  |
| [第十八章 面对配分函数](https://exacity.github.io/deeplearningbook-chinese/Chapter18_confronting_the_partition_function/) | @liber145 | | @tankeco |
| [第十九章 近似推断](https://exacity.github.io/deeplearningbook-chinese/Chapter19_approximate_inference/) | @futianfan | | @sailordiary, @hengqujushi, huanghaojun |
| [第二十章 深度生成模型](https://exacity.github.io/deeplearningbook-chinese/Chapter20_deep_generative_models/) | @swordyork | | |
| 参考文献 | | | @pkuwwt |

我们会在纸质版正式出版的时候，在书中致谢，正式感谢各位作出贡献的同学！

还有很多同学提出了不少建议，我们都列在此处。

@tttwwy @tankeco @fairmiracle @GageGao @huangpingchun @MaHongP @acgtyrant @yanhuibin315 @Buttonwood @titicacafz 
@weijy026a @RuiZhang1993 @zymiboxpay @xingkongliang @oisc @tielei @yuduowu @Qingmu @HC-2016 @xiaomingabc 
@bengordai @Bojian @JoyFYan @minoriwww @khty2000 @gump88 @zdx3578 @PassStory @imwebson @wlbksy @roachsinai @Elvinczp 
@endymecy name:YUE-DaJiong @9578577 @linzhp @cnscottzheng @germany-zhu  @zhangyafeikimi @showgood163 @gump88
@kangqf @NeutronT @badpoem @kkpoker @Seaball @wheaio @angrymidiao @ZhiweiYang @corenel @zhaoyu611 @SiriusXDJ @dfcv24 EmisXXY
FlyingFire vsooda @friskit-china @poerin @ninesunqian @JiaqiYao @Sofring @wenlei @wizyoung @imageslr @@indam @XuLYC
@zhouqingping @freedomRen @runPenguin @pkuwwt @wuqi @tjliupeng @neo0801 @jt827859032 @demolpc @fishInAPool
@xiaolangyuxin @jzj1993 @whatbeg LongXiaJun jzd

如有遗漏，请务必通知我们，可以发邮件至`echo c3dvcmQueW9ya0BnbWFpbC5jb20K | base64 --decode`。
这是我们必须要感谢的，所以不要不好意思。


TODO
---------

 1. 排版



注意
-----------

 - 各种问题或者建议可以提issue，建议使用中文。 
 - 由于版权问题，我们不能将图片和bib上传，请见谅。
 - Due to copyright issues, we would not upload figures and the bib file.
 - 可用于学习研究目的，不得用于任何商业行为。谢谢！



Markdown格式
------------
这种格式确实比较重要，方便查阅，也方便索引。初步转换后，生成网页，具体见[deeplearningbook-chinese](https://exacity.github.io/deeplearningbook-chinese)。
注意，这种转换没有把图放进去，也不会放图。目前使用单个[脚本](scripts/convert2md.sh)，基于latex文件转换，以后可能会更改但原则是不直接修改[md文件](docs/_posts)。
需要的同学可以自行修改[脚本](scripts/convert2md.sh)。



HTML格式
------------
读者可以使用[pdf2htmlEX](https://github.com/coolwanglu/pdf2htmlEX)进行转换，直接将PDF转换为HTML。



Updating.....
