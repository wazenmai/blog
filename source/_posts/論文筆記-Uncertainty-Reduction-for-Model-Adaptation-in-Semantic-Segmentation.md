---
title: '[論文筆記] Uncertainty Reduction for Model Adaptation in Semantic Segmentation'
date: 2021-08-17 23:54:02
cover: /assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/model.png
description: 這篇是第一個把 "model adaptation" 這樣的方法用在 UDA semantic segmentation 領域的論文，也就是在沒有 source data 和 source label，但是有 target data 的情況下去改善 model 在 target domain 上的表現。
tags:
    - Semantic Segmentation
    - UDA
categories:
    - Machine Learning
katex: true
---
這篇是第一個把 *model adaptation* 這樣的方法用在 *UDA semantic segmentation* 領域的論文，也就是在**沒有 source data 和 source label，但是有 target data** 的情況下去改善 model 在 target domain 上的表現。

如果是想快速知道這篇論文大概在做什麼的讀者，可以直接看 [Overview](#overview) 就好；
如果是想看詳細方法介紹的讀者，筆者會在後面說明此論文要解決的問題是什麼，接著介紹作者提出的方法，最後再帶大家看實驗結果。

# Domain Knowledge

本篇會出現的一些常見的專有名詞會在這裡介紹，筆者一開始在讀論文時常常被這些專有名詞搞到很挫折，希望能透過淺白的講解讓大家快速進入狀況，如果是熟悉這領域的大大可以直接跳過這段～

{% note info no-icon %}
### Semantic Segmentation
Semantic Segmentation 可以看成是一種 pixel classification。傳統給 model 一張圖，叫它分辨那是人還是車子，現在是叫 model 區分一張圖的每個 pixel 分別是屬於什麼類別，常用於街景分析（區分哪些是行人哪些車子哪裡是路）。常見的 semantic segmentation datasets 為 Cityscapes, PASCAL VOC 和 ADE20K。
{% endnote %}

{% note info no-icon %}
### Unsupervised Domain Adaptation (UDA)
UDA 就是在有 **source domain data + source domain label + target domain data** 的情況下，想辦法利用在 source domain 學到的資訊應用到 target domain 上，讓 model 在 target domain 上也有不錯的表現。

會出現這樣的領域主要就是因為 target domain 的 label 需要花很久的時間才能標好，拿這篇的 semantic segmentation 的 task 來說好了，標好一張現實世界的 Cityscapes 的圖平均需要 90 分鐘。相反的 source domain 會相對容易標記，例如本篇用的 GTA5 是遊戲生成的圖片，平均 7 秒就能標好一張圖，而這兩個 datasets 會出現的類別有 19 個是重疊的，例如天空、汽車、行人等等。

專家們就想，如果 model 能學會分辨 source domain 的物體類別的話，那麼 target domain 的物體應該也能學得好，於是 UDA 領域就誕生啦！
{% endnote %}

# Overview

{% note default no-icon %}
標題：Uncertainty Reduction for Model Adaptation in Semantic Segmentation
作者：Prabhu Teja S, François Fleuret (Idiap Research Institute)
年份：2021
榮譽：CVPR 2021
{% endnote %}

{% note default no-icon %}
Paper  link: http://publications.idiap.ch/downloads/papers/2021/Sivaprasad_CVPR_2021.pdf
Github link: https://github.com/idiap/model-uncertainty-for-adaptation
{% endnote %}

此篇論文主要的貢獻有三點：
1. 是第一個把 *model adaptation* 應用在 *UDA semantic segmentation* 上的論文。
2. 由於沒有 source data 可以重複訓練，作者提出了一些方法來降低 model 預測的不確定性，包含運用 uncertainty loss, entropy regularizer 和 pseudo-labeling 的技巧。
3. 應用此方法，可以達到跟有用 source data 的 model 們差不多的表現。（註：此篇論文挑選的 baseline 均是 2019 前的方法）

模型表現：
| Task                            | Class Number | City   | mIoU |
|:--------------------------------|:-------------|:-------|:------|
|GTA $\rarr$ Cityscapes           | 19 classes   | x      | 45.1 |
|Synthia $\rarr$ Cityscapes       | 16 classes   | x      | 39.6 |
|                                 | 13 classes   | x      | 45.0 |
|Cityscapes $\rarr$ NTHU Crosscity| 13 classes   | Rome   | 53.8 |
|                                 | 13 classes   | Rio    | 53.5 |
|                                 | 13 classes   | Tokyo  | 49.8 |
|                                 | 13 classes   | Taipei | 50.1 |

# Methods
在進入正式方法前，先來區分一下 **model adaptation** 跟 **domain adaptation**。

我們知道 domain adaptation 是指把 source domain 學習到的東西遷移到 target domain，而這時我們擁有的資源有來自 source domain 的 data 跟 label，以及 target domain 的 data，希望藉由這些來讓 model 自己學習到 target domain 的 label。

model adaptation 要做的目標跟 domain adaptation 完全一樣，只差在現在我們**沒有 source data，也沒有 source label，但是有一個事先在 source domain 訓練完成的 pre-trained model 以及 target data**。問題來了，為什麼好端端的 source data 不用，要故意創造一個這麼艱難的問題呢？原因就在於處理真實世界的 UDA 問題時，source data 不一定會開放給大家使用，例如醫學影像的訓練，那些資料都包含了病人隱私問題，有些時候沒辦法開放大家使用，但 model 就沒有所謂隱私權問題，只是一堆參數罷了，這樣只拿 model 做更動的研究，被作者稱作 model adaptation（之後以 MA 代稱）。

這兩種類別都屬於遷移學習（transfer learning）的一種。

## Formal Problem

現在把 MA 的問題用正式的符號來表示，$\mathcal{X}$ 代表輸入、$\mathcal{Y}$ 代表輸出、$\mathcal{S}$ 代表 source domain、$\mathcal{T}$ 代表 target doamin。所以 $X_S$ 就是在 source domain 的輸入，包含 data 跟 label，而 $X_T$ 則是指 target domain 上的 data。

這裡以 $f$ 來代表整個模型架構，而且 $f(x, \theta_S)\equiv f_S(x)$，也就是說此處的模型就是那個先在 source domain 上的 pre-trained model，他們的架構是完全一樣的。這個 $f$ 是由一個 feature extractor network (代號 $g$，這裏採用 ResNet-101) 和一個 ASPP decoder (代號 $h$) 組成的。

{% note default no-icon %}
**Atrous Spatial Pyramid Pooling (ASPP)**
ASPP decoder 是 DeepLab-v2 採用的架構，由於此論文的重點在於作者設計的 loss funciton，因此筆者不會在此解釋當中的數學，有興趣的讀者可以參考這篇文章：[Review: DeepLabv1 & DeepLabv2 — Atrous Convolution (Semantic Segmentation)](https://towardsdatascience.com/review-deeplabv1-deeplabv2-atrous-convolution-semantic-segmentation-b51c5fbde92d) 或是 [Deeplab-v2 論文](https://arxiv.org/abs/1606.00915)。
{% endnote %}

因此我們的問題可以這樣表示：在擁有 $f_S$ 和 $X_T$ 的情況下，我們的目標是透過改善 $f_S$ 來在 $\mathcal{T}$ 的 data 上拿到更好的表現。

![](/assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/model.png)

**1. Adding noise to the feature representation using dropout**
首先，我們希望 model 的 output 越穩定越好，也就是 predict 出來的 class 不會被 noise 干擾，這有相當多的做法可以實現，這裏作者選擇一個簡單又強大的方法就是把 feature representation 分別通過 droupout，這樣我們拿到的就是多個有「殘缺」的 feature，再經過同樣的 ASPP decoder 架構後，我們希望這些 decoder 的 output 都可以是一樣的，這樣表示我們的 model 可以不受 noise 干擾的進行分類，也能讓 feature 變得更加 robust。

上方的圖片為作者提出的 model 架構，原本的 pre-trained model $f_S$ 就只包含黃色的 shared backbone 和 綠色的 main decoder，作者另外把 ResNet-101 產生的 feature representation 分別過了另外四個 dropout 並經過 auxiliary decoder (Aux decoder) 產生 semantic segmentation map 後，拿這些圖來跟 main decoder 做比對當作 **uncertainty loss $L_{un}$**。

$$ L_{un} = \frac 1 N \sum^N_{i=1}(\hat{y}^i-y)^2$$

其中 $y$ 為 main decoder 的 output，$\hat{y}$ 為 Aux decoder 的 output，總共有 $N$ 個 Aux decoder ($N = 4$)，這樣如果 $y$ 和 $\hat{y}$ 的結果很不一樣，$L_{un}$ 就會很大，就能迫使 model 讓他們的 prediction 盡量像。


**2. Entropy regularizer**
第二個方法同樣也是為了可以增加 model 的穩定性，這裏的 entropy regularizer 就是把 entropy 加進 loss 裡面一起算，會用 regularizer 只是因為他有正則化的作用，可以避免 class-overlap 的情況。

$$L_{ent}=H\{f(x;\theta)\}$$

這裏的 $H\{\}$ 為 input $x$ 對所有的 class 的機率分佈的 entropy。

{% note default no-icon %}
**Entropy**
熵，接收的所有訊息中所包含的資訊的平均量，也可以說是拿來看資料的亂度、不確定性。
$$H(x)=\sum_i-p_ilog_2(p_i)$$
當一件事情的發生機率越接近 0.5 時，entropy 會越高，也代表這個事件很不確定、訊息量越大。反之當一件事情的發生機率越接近 1 或 0 時，entropy 會越小，代表此事件已經相當確定了、不太可能有太大的變動，這就是為什麼我們希望 model prediction 的 entropy 盡量小的緣故。
{% endnote %}

**3. Pseudo-labeling**
雖然上述的兩個 loss 可以相當大程度上降低 uncertainty，但是卻無法應付 *interchanged labeling* （把 class-0 預測成 class-1，class-1 預測成 class-0），因此作者使用了 pseudo-label 的方式來避免這種情況，而這裡的 pseudo-label 跟一般方法不一樣的地方在於，只會採用大於某個 threshold 的 pseudo-label，而且這個 threshold 會根據 class 的不同有所變化，這樣可以保證只有比較確定的區域會被當成 pseudo-label，避免出現嚴重的 bias 問題。
$$y_{PL}=\begin{cases}
   \argmax f(x,\theta) &\text{if } \max (f(x,\theta))\ge \tau \\
   \text{IGNORE} &\text{otherwise }
\end{cases}$$
$$L_{PL}=-\bold{1}^T_{y_{PL}}log(y)$$


$-\bold{1}^T_{y_{PL}}$ 代表 target domain 上 pseudo-label 的 one-hot vector，$L_{PL}$ 其實就是 model 的 output 和 pseudo-label 的 cross-entropy loss。

到這裡，我們已經把作者提出的方法看完啦，可喜可賀！所以最終此論文的 objective function 就會是上面那三個 loss 相加，其中 $\lambda$ 是用來決定該 loss 佔整體 loss 的比重，在這篇論文中是使用 $\lambda_{ent}= 1.0, \lambda_{un} = 0.1$。
$$L=L_{PL}+\lambda_{ent}L_{ent}+\lambda_{un}L_{un}$$

# Experiments
## Toy Experiment
![](/assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/toy_example.png)

作者先用了一個較簡單的實驗來證明上述的 loss 真的可以幫助降低 uncertainty，這裏用的也是很簡單的 model 架構。可以看到 (a) (b) 顯示了 source data 跟 target data 的分佈（這裡的 target data 是由旋轉 source data 後獲得，此處只分兩類），這裏的實驗條件跟我們想處理的問題一樣，model 看不到 target data 的 label，且訓練過程中也不會用到 source data，所以一開始 target data feature 完全沒有分開（c），再加了 entropy regularizer 後可以看到 feature 大致上被分成兩邊 (d)，且在加了 uncertainty loss 後可以把 feature 拉離 decision boundary (e)，證明此方法是真的有用的。

## Datasets Information
{% tabs Unique name, [index] %} 
<!-- tab Cityscapes -->
- 圖片均為 2048 * 1024 的街景圖片
- 2975 annotated images for training set
- 500 images for validation set (benchmark)
- 19 classes
- https://www.cityscapes-dataset.com
<!-- endtab -->
<!-- tab GTA5 -->
- 圖片均為來自遊戲 Grand Theft Auto 的街景圖片
- 24966 frames, 1914 * 1052 
- Share 19 classes with Cityscapes
- https://download.visinf.tu-darmstadt.de/data/from_games
<!-- endtab -->
<!-- tab Synthia -->
- 9400 images, 1280 * 760
- share 16 class with Cityscapes
- https://synthia-dataset.net
<!-- endtab -->
<!-- tab NTHU Crosscity -->
- 共包含了四個城，2048 * 1024 的街景圖片 （Rome, Rio, Taipei, Tokyo）
- 3200 unlabeled image as training data
- 100 labeld image as target data
- share 13 classes with Cityscaoes
- https://yihsinchen.github.io/segmentation_adaptation
<!-- endtab -->
{% endtabs %}


## Main Results
實驗部分，雖然它在三個 task 中都沒有拿到很高的成績，但是就沒有 source data 的情況下分數已經相當不錯，另外作者有提到他們的 variance 較高，光是更改 random seed 就能讓分數有兩分左右的誤差。
### GTA5 - Cityscapes
![](/assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/GTA5_CC.png)
### Synthia - Cityscapes
![](/assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/Syn_CC.png)
### Cityscapes - NTHU Crosscity
![](/assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/CC_NTHU.png)

# Conclusion
筆者在這篇文章中帶大家看了在沒有 source data 的情況下要如何做 UDA segmentation，也把此論文提出的三個主要的方法講解過一遍，希望可以讓大家更了解這個領域的作品。這篇論文離 SOTA 的分數還有一段距離，筆者相信可以透過應用現有的 UDA 方法到 model adaptation 的問題中來得到更好的 performance，例如應用 consistency training 或是加入 data augmentation 的方法等等，就等大家來實現啦！

Reference:
- [Uncertainty Reduction for Model Adaptation in Semantic Segmentation, CVPR 2021](http://publications.idiap.ch/downloads/papers/2021/Sivaprasad_CVPR_2021.pdf)

{% note blue 'far fa-snowflake' modern %}
有任何問題都歡迎在下面提出，喜歡這篇文章的話可以幫我點一個讚，
祝福各位能在機器學習領域走出屬於自己的一條路。
{% endnote %}

