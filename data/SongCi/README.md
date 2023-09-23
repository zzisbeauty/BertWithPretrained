## 数据集集仓库
https://github.com/chinese-poetry/chinese-poetry


## 数据集划分
这里掌柜已经将所需要用到的原始数据都放在了当前目录中，只需要运行`read.py`这个脚本便可以将原始数据集划分为训练集、验证集和测试集。同时，由于是模型预训练，所以也没必要再有一个测试集，因此这里保持了验证集和测试集一样。

## 数据形式
划分完成的数据形式如下所示：
```python
鼎湖龙远，九祭毕嘉觞。遥望白云乡。箫笳凄咽离天阙，千仗俨成行。圣神昭穆盛重光。宝室万年藏。皇心追慕思无极，孝飨奉尝。
凤箫声断，缥缈溯丹邱。犹是忆河洲。荧煌宝册来天上，何处访仙游。葱葱郁郁瑞光浮。嘉酌侑芳羞。雕舆绣归新庙，百世与千秋。
中兴复古，孝治日昭鸿。原庙饰瑰宫。金壁千门万，楹桷竟穹崇。亭童芝盖拥旌龙。列圣俨相从。共锡神孙千万寿，龟鼎亘衡嵩。
```
其中每一行为一首词，句与句之间通过句号进行分割。

在实例化类`LoadBertPretrainingDataset`时，只需要将`data_name`参数指定为`songci`即可将本数据作为模型的训练语料。