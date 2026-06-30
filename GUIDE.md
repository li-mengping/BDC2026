# 打包docker
在训练与测试完成之后，需要首先将项目整体打包成一个docker镜像（打包），再将该镜像导出为一个.tar文件（导出），最终提交该tar文件即可，里面需要包含运行时的所有环境及依赖，具体可以参考或修改Dockerfile

## docker镜像创建

如果出现网络请求错误，请尝试使用代理或关闭代理。或者参考[本文](https://blog.csdn.net/m0_70878103/article/details/144130047)

镜像创建指令：
`docker buildx build  --platform linux/amd64  --build-arg IMAGE_NAME=nvidia/cuda  -t bdc2026 .`

成功创建镜像

<img src="./asset/docker_build.png" alt="docker_build" width="80%">

## 镜像导出
`docker save -o 队伍名称.tar bdc2026:latest`

成功导出镜像为.tar文件

<img src="./asset/export.png" alt="export" width="80%">

# 可运行性验证
选手根据需要可以进行三步运行验证，包括：
1. 本机环境直接运行验证

需要选手保证根目录的train.sh, test.sh成功运行，并且在output目录下生成一个result.csv

2. 对打包后的docker进行完整运行验证

在这一步，选手需要按照打包docker的流程先将项目打包为一个镜像（对应上面的docker镜像创建），暂时不用导出为.tar文件

然后在根目录下运行`docker compose up`。这一步会对打包成的docker是否可运行进行验证，如果运行后在test/output中得到result.csv，验证即成功。

3. (可选)模拟赛事方最终以批处理的方式进行打分验证

这一步需要选手将镜像导出为tar文件（最终需要选手提交的文件），然后将tar文件放到test/tars目录下，并将.tar文件名写入test/tar_files_list.txt文件中。（比如tar文件叫1.tar，则在tar_files_list.txt的第一行写入1.tar即可）

然后在根目录下运行
linux
`python test/test.py`
windows
`python test/test_windows.py`

成功运行后，如果在test/result.csv中看到，类似下面的结果，该步验证成功。
```
Team Name,Final Score
1,0.018867553640330992
```

## 代码审核要求
选手提交的代码应当符合一定的代码规范，主办方将对选手提交的代码进行可复现性以及合规审查。成绩有效性需同时满足以下所有要求：

1) 在固定随机种子点的前提下，从训练过程开始复现，选手提交项目代码运行生成的结果和提交的结果完全一致；
2) 确保代码可以在指定机器（i7-13650H，16GB内存，4060独显8GB显存，50GB存储）上运行并生成结果。预测时间：选手的模型预测时间不得超过5分钟。训练时间：模型训练时间不得超过8小时；
3) 选手提交的全部模型代码文件（docker文件，包含所有环境、库包、代码、数据和模型等）的大小总和不超过10G，不能压缩；
4) 允许使用开源的词典、embedding和预训练模型，以上数据和模型需在4月1 日前开源，且需通过邮件的形式报备开源链接地址和md5，报备邮箱为data@tsinghua.edu.cn；
5) 复现训练和预测时不得联网；
6) 主要贡献为机器学习方法，包括模型的训练和预测。对于不可复现的代码，主办方有权取消选手的获奖资格。

复现流程

1. 参赛选手在7月18日前将用到的开源模型及数据报备到报备邮箱
data@tsinghua.edu.cn，之后不得更改。邮件主题格式为“团队名称 + 模
型数据报备”，邮件内容需包含开源链接地址和md5，不得直接上传文件。

2. 选手按要求整理调试代码，并在规定时间内提交项目文件：
o docker文件：选手需提供完整的代码（python）、数据和训练模
型。
代码规范： 选手提交的文件夹结构如下，选手需按照提供的格式组织文
件，标注为必选的表示必须存在的文件或文件夹：
|--app
    |--code（储存运行代码，docker内）
    |--data（储存数据，在docker-compose.yml 中进行挂载）
    |--model（储存训练模型和其它数据，docker内）
    |--output（储存计算结果，在docker-compose.yml 中进行挂载）
        |--result.csv
    |--temp（储存中间结果，在docker-compose.yml 中进行挂载）
    |--init.sh（docker 内，必选）
    |--train.sh（docker 内，必选）
    |--test.sh（docker 内，必选）
    |--readme.md（必选）

选手需要按照上面所示将文件进行存储，在docker中储存code、model、init.sh、train.sh、test.sh 等文件，将docker 镜像命名为bdc2026，并导出为“队伍名称.tar”进行提交，我们会使用docker load-i 队伍名称.tar命令对docker镜像进行加载，并使用下发的docker-compose.yml 文件运行。

3. 选手需要在 readme.md 中对使用的算法（方法细节、针对性的问题解决方案、创新点等）、辅助数据、使用的预训练模型等进行说明，并描述模型训练、测试流程。其他相关注意事项也应说明。参考格式：

```
# 代码说明
## 环境配置
注明python、pytorch 等依赖的版本
## 数据
使用了XX公开数据，数据获取链接为XX，在训练YY模型时使用
## 预训练模型
使用了XX预训练模型，可以通过XX方式获得，对code/model.py中的YY网络
进行初始化
## 算法
### 整体思路介绍
### 方法的创新点（如果有）
### 网络结构
### 损失函数
### 数据扩增
### 模型集成
### 算法的其他细节
## 训练流程
对train.py 每一步进行描述，或者在train.py中对每一步添加注释
## 推理流程
对test.py 每一步进行描述，或者在test.py中对每一步添加注释
## 其他注意事项
例如验证数据的划分等
```

4. 项目提交截止后，复现人员将查验开源模型与数据的md5值，与报备信息进行比对。

5. 复现人员从训练过程开始完整复现本次参赛项目，并查验是否满足上述复
现要求