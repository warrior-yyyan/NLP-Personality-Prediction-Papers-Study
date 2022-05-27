[TOC]

# 1. A sentiment-aware deep learning approach for personality detection from text

[paper](./finished/01.pdf)	[translation](./finished/01zh.pdf)

- 采用1999 essay Big Five dataset 和 kaggle MBTI dataset
- BERT + CNN / GRU / LSTM结构，加入了Sentic情感词典信息
- 交叉熵损失
- 对MBTI倾斜数据集做了处理
- 采用sentence-level编码，把每一句输入BERT, 然后取倒数第二层的CLS作为表示，将一个人的所有句子拿CLS拼接起来做预测
- 对Sentic的词从-1 ~ 1映射为了0 ~ 2 且每0.1是一个level  映射为20个level，然后变成一个20 dimension的 one-hot vector 拼接进BERT中的768维中变成 788维
- 引入了超参数K，表示为连续K个句子输入进模型的效果，后续实验给出了最佳的K，和在最佳的K上做的实验结果
- 引入皮威尔逊系数说明了各种人格之间的联系不大
- 对比了单标签分类和多标签分类



# 2.Who Am I? Personality Detection Based on Deep Learning for Texts

[paper](./finished/02.pdf)	[translation](./finished/02zh.pdf)

- 把word的vector拼接在双向LSTM向量中间，形成了 LL-word-LR的形式
- 输入长度为500个word的序列用maxpool降为25长度，25代表25个句子，划分为句子级别
- 接下来沿用textCNN的思想，3个不同的kernel做句子级别的特征提取，然后maxpool，全连接降维分类

# 3. Predicting Myers-Briggs Type Indicator with Text Classification

[paper](./finished/03.pdf)

- kaggle中的数据比例失衡，在测试时人为的调整了各种性格的比例，达到正常比例
- 用NLTK处理数据，去掉网站链接，去掉停用词，去掉正文中含有INTJ INFP等预测名词，以防止网络学到记住这些知识
- 规定每句话长度为40词
- Embedding用的Glove，50 dimension
- 用的RNN，LSTM效果最好，加了0.1的dropout
- 训练4个二分类器
- 对用户人格预测的两种方式
  - 第一，把这个用户的所有句子都做预测，去类别最多的那个类
  - 第二，取平均（取平均的效果好很多）
- 16分类可能会在绝对准确度上比较高，但是这并没有考虑性格之间的关联，4个二分类可以得到较高的近似正确率



# 4. Myers-Briggs Personality Classiﬁcation and Personality-Speciﬁc Language Generation Using Pre-trained Language Models

[paper](./finished/04.pdf)

- 自己爬的数据集
- 清洗数据的时候把MBTI的类型换成了 '<type>'
- 用了BertForSequenceClassiﬁcation做分类，用了bert-base-uncased, （说用large效果会更好）
- 用了一种新的评价指标，at least match
- 16分类 acc做到0.479
- 4个二分类能做到最高0.758



# 5. Deep Learning-Based Document Modeling for Personality Detection from Text

[paper](./finished/05.pdf)

- 





# 6. Personality Trait Detection Using Bagged SVM over BERT Word Embedding Ensembles

[paper](./finished/06.pdf)

- Bert最长输入为512， 文章平均长度为650词，本文采用了sub-doc的方法，把文章切分为200词的小文章的组合，然后给这些小文章和大文章一样的标签
- 把文章按照句号 问号处切分为句子组合，并删除所有ASCII 字母、数字、引号和感叹号以外的字符
- 把缩写 you're 变成you are等，把sub-doc的长度从200变为250
- 把bert后四层向量和Mairesse特征拼接起来，形成3156维向量
- 使用10个SVM做分类，类似于bagging classifier
- BB-SVM模型平均预测精度 59.03
- 用同样的分类器，不同的Word Embedding 做了对比，取后四层的Bert向量 好过 w2v 将近2个点
- 取BERT最后四层表示 好过 取任何一层表示（单层对比来看9,10,11,12最后四层好过其它层）





# 7. Bottom-Up and Top-Down: Predicting Personality with Psycholinguistic and Language Model Features

[paper](./finished/07.pdf)	[translation](./finished/07zh.pdf)

**知识点：**

- **皮尔逊相关系数（Pearson correlation coefficient）：**在[统计学](https://baike.baidu.com/item/统计学/1175)中，**皮尔逊相关系数**( Pearson correlation coefficient），又称**皮尔逊积矩相关系数**（Pearson product-moment correlation coefficient，简称 **PPMCC**或**PCCs**），是用于度量两个变量X和Y之间的[相关](https://baike.baidu.com/item/相关/9882881)（线性相关），其值介于-1与1之间。在自然科学领域中，该系数广泛用于度量两个变量之间的线性相关程度。
- 点二列相关（point biserial correlation）是2014年公布的心理学名词。
- VAD和PAD的区别，[link](https://www.zhihu.com/question/339294737)  根据link中的回答来讲，语言理解上没有区别，可以当做一种东西。
- **SHAP（SHapley Additive exPlanations）**是一种博弈论方法，用于解释任何机器学习模型的输出。它使用博弈论中的经典 Shapley 值及其相关扩展将最优信用分配与局部解释联系起来（有关详细信息和引文，请参阅 [论文](https://github.com/slundberg/shap#citations)）。
- 心里语言学特征
  - Mairesse
  - SenticNet
  - NRC Emotion Lexicon
  - VAD Lexicon 
  - Readability




**论文贡献点：**

1. 提供了代码，可以作为baseline
2. 引入很多种用心理学语言做特征提取
3. 用点二列相关这个指标 衡量了 senticnet和大五人格的相关性，同时分析了每种人格最重要的三种特征
4. 用SHAP做了可视化解释
5. 做实验说明了 初始化的种子 对实验的影响很大



**改进点：**

1. 替换特征提取模型(deberta-v3)，能获得更好的结果，目前 63.5



# 8. SIMPA: Statement-to-Item Matching Personality Assessment from text

[paper](./finished/08.pdf)	[translation](./finished/08zh.pdf)

**知识点：**

- 



**论文贡献点：**

1. 



**改进点：**

1. 



# 9. Personality prediction model for social media using machine 
learning Technique

[paper](./finished/09.pdf)	[translation](./finished/09zh.pdf)

**知识点：**

- 



**论文贡献点：**

1. 



**改进点：**

1. 





