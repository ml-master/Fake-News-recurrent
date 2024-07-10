# Fake-News-recurrent

学号：2310274037

机器学习期末复现实验作业，www 24'  基于竞争智慧防御的大语言模型可解释假新闻检测算法的实现与分析 

论文全称：**Explainable Fake News Detection With Large Language Model via Defense Among Competing Wisdom**

论文地址：[[2405.03371] Explainable Fake News Detection With Large Language Model via Defense Among Competing Wisdom](https://arxiv.org/abs/2405.03371)

原文公开代码地址：[GitHub - wangbo9719/L-Defense_EFND: Explainable Fake News Detection With Large Language Model via Defense Among Competing Wisdom](https://github.com/wangbo9719/L-Defense_EFND)

## 1. 环境配置

```
conda create -n efnd python=3.8
source activate efnd
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers pandas tqdm nltk sklearn tensorboardX openai
```

## 2. 项目简介

论文《Explainable Fake News Detection With Large Language Model via Defense Among Competing Wisdom》提出了一种新的基于防御的可解释的假新闻检测框架（EFND），旨在解决现有方法的局限性。具体来说，该框架包含以下三个主要模块：
1.	证据提取模块：首先，该模块将群众智慧分成两个竞争方，分别检测显著证据。这一过程类似于在法庭上搜集控辩双方的证据，通过对比两种不同观点的证据，提供更多的信息来源。
2.	基于提示的大语言模型模块：接下来，利用大语言模型进行推理，生成支持两种可能真实性的证明。大语言模型能够结合上下文信息，通过提示生成具有逻辑性和说服力的文本。
3.	防御推理模块：最后，通过建模不同证明之间的防御来确定新闻的真实性。该模块模拟了法庭辩论中的防御机制，通过对比和分析不同证明之间的矛盾和共性，从而得出最终的真实性结论。

## 3. 数据集

论文使用的数据集：RAWFC、LIAR-RAW
 
课程提供的数据集：Style-based-Fake


## 4. 模型训练

### 4.1 训练特征提取器并提取top-k证据

这里需要自行部署roberta-large或连接hugging-face获取roberta-large。

数据集请根据dataset下的对应readme文件自行下载

第一步，运行以下代码，训练特征提取器，在--dataset_name处可以指定课程数据集style_based_fake为训练数据集

```
CUDA_VISIBLE_DEVICES=0 python source/extractor_train.py \
--model_name_or_path ./roberta-large \
--dataset_name RAWFC \
--output_dir ./runnings/step1_extraction_model_RAWFC \
--do_train \
--train_batch_size 1 \
--num_train_epochs 5 \
--eval_batch_size 64 \
--eval_steps 500 \
--max_seq_len 32 \
--gradient_accumulation_steps 1 \
--correlation_method mlp \
--learning_rate 1e-5 \
--cls_loss_weight 0.9
```

第二步，运行以下代码，获取提取的top-k证据，在--dataset_name处可以指定课程数据集style_based_fake为训练数据集

```
CUDA_VISIBLE_DEVICES=0 python source/extractor_train.py \
--model_name_or_path ./runnings/step1_extraction_model_RAWFC \
--dataset_name RAWFC \
--output_dir ./dataset/RAWFC_step2 \
--get_evidences \
--train_batch_size 1 \
--num_train_epochs 5 \
--eval_batch_size 32 \
--eval_steps 500 \
--max_seq_len 32 \
--gradient_accumulation_steps 1 \
--correlation_method mlp \
--learning_rate 1e-5 \
--cls_loss_weight 0.9
```

## 4.2 获取 LLM 生成的解释

```
python ./source/step2_explanation_generation.py \
--dataset_name RAWFC_step2 \
--dataset_type all \
--generate_explanation_llama2
```

## 4.3 最终的新闻真实性预测

```
CUDA_VISIBLE_DEVICES=0 python source/step3_final.py \
--dataset_name RAWFC_step2 \
--explanation_type gpt \
--model_name_or_path roberta-large \
--output_dir ./runnings/step3_RAWFC \
--do_train \
--do_prediction \
--train_batch_size 8 \
--num_train_epochs 5 \
--eval_batch_size 32 \
--eval_steps 500 \
--learning_rate 5e-6 \
```

## 5. 实验结果

**RAWFC数据集上进行的训练过程和实验结果如下：**

07/09 07:17:32  PM: Ep[1] Loss: 0.9904 KL-Loss: 0.3405 CLS_loss: 1.0627

07/09 07:19:28  PM: Ep[2] Loss: 1.0850 KL-Loss: 0.4030 CLS_loss: 1.1608

07/09 07:21:09  PM: Ep[3] Loss: 0.9821 KL-Loss: 0.0011 CLS_loss: 1.0911

07/09 07:22:53  PM: Ep[4] Loss: 1.0814 KL-Loss: 0.3725 CLS_loss: 1.1602

07/09 07:24:36  PM:  Ep[5] Loss: 1.0741 KL-Loss: 0.3801 CLS_loss: 1.1512

07/09 07:36:44  PM: Ep[6] Loss: 1.0500 KL-Loss: 0.3850 CLS_loss: 1.1300

07/09 07:38:28  PM: Ep[7] Loss: 1.0305 KL-Loss: 0.3730 CLS_loss: 1.1125

07/09 07:40:12  PM: Ep[8] Loss: 1.0221 KL-Loss: 0.3624 CLS_loss: 1.0980

07/09 07:41:56  PM: Ep[9] Loss: 1.0137 KL-Loss: 0.3518 CLS_loss: 1.0845

07/09 07:43:40  PM: Ep[10] Loss: 1.0074 KL-Loss: 0.3450 CLS_loss: 1.075

| Model         | Precision | Recall | Macro F1 | Precision | Recall | Macro F1 |
| ------------- | --------- | ------ | -------- | --------- | ------ | -------- |
| Dataset       | RAWFC     |        |          | LIAR-RAW  |        |          |
| LLaMA2:7b     | 37.40     | 38.03  | 36.77    | 17.2      | 17.7   | 15.14    |
| Command-r:35b | 42.12     | 45.01  | 37.21    | 29.64     | 23.57  | 21.99    |

**Styel-based-fake数据集上进行的训练过程和实验结果如下：**

07/08 07:45:24  PM: Ep[1] Loss: 1.1978 KL-Loss: 0.4325 CLS_loss: 1.3640

07/08 07:47:08  PM: Ep[2] Loss: 0.9893 KL-Loss: 0.3241 CLS_loss: 1.0526

07/08 07:48:52  PM:  Ep[3] Loss: 0.9810 KL-Loss: 0.3175 CLS_loss: 1.0437

07/08 07:50:36  PM: Ep[4] Loss: 0.9735 KL-Loss: 0.3108 CLS_loss: 1.0350

07/08 07:52:20  PM: Ep[5] Loss: 0.9660 KL-Loss: 0.3045 CLS_loss: 1.0263

07/08 07:54:04  PM: Ep[6] Loss: 0.9585 KL-Loss: 0.2981 CLS_loss: 1.0176

07/08 07:55:48  PM: Ep[7] Loss: 0.9518 KL-Loss: 0.2927 CLS_loss: 1.0098

07/08 07:57:32  PM: Ep[8] Loss: 0.9450 KL-Loss: 0.2875 CLS_loss: 1.0020

07/08 07:59:16  PM: Ep[9] Loss: 0.9383 KL-Loss: 0.2823 CLS_loss: 0.9943

07/08 08:09:40  PM:  Ep[10] Loss:0.8998 KL-Loss: 0.2548 CLS_loss: 0.9510

| Model         | Precision        | Recall | Macro F1 |
| ------------- | ---------------- | ------ | -------- |
| Dataset       | style-based-fake |        |          |
| LLaMA2:7b     | 60.95            | 60.03  | 61.77    |
| Command-r:35b | 62.52            | 66.18  | 65.43    |
