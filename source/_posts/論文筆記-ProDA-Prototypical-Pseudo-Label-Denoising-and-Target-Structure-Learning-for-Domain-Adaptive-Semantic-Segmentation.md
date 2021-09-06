---
title: >-
  [論文筆記] Prototypical Pseudo Label Denoising and Target Structure Learning
  for Domain Adaptive Semantic Segmentation
katex: true
date: 2021-08-26 22:35:42
cover: /assets/posts/ProDA/ProDA.png
description: >-
  2021 UDA segmentation SOTA. 
  GTA - Cityscapes 57.5 mIoU. SYNTHIA - Cityscapes 55.5 / 62.0 mIoU.
tags:
  - Semantic Segmentation
  - UDA
categories:
  - Machine Learning
---
這篇是筆者目前看到在 UDA, semantic segmentation 領域表現最好的一篇，分數在加了知識蒸餾（knowledge distillation）後整整甩其他人一大截，用的方法也不難懂，主要就是找各個類別在 feature space 的中心點（prototype），使同類別的 data point 接近這個中心點，讓模型更好分類。

如果只是想快速知道這篇論文大概在做什麼，可以直接看 [Overview](#overview) 就好；
如果是想看詳細方法介紹，筆者會在後面說明此論文要解決的問題是什麼，接著介紹作者提出的方法，最後再帶大家看實驗結果。

# Domain Knowledge

本篇會出現的一些常見的專有名詞會在這裡介紹，筆者一開始在讀論文時常常被這些專有名詞搞到很挫折，希望能透過淺白的講解讓大家快速進入狀況，如果是熟悉這領域的大大可以直接跳過這段～

如果是對 UDA 和 semantic segmentation 不了解的讀者，可以看看[這篇](/%E8%AB%96%E6%96%87%E7%AD%86%E8%A8%98-Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/#domain-knowledge)。

{% note info no-icon %}
### Pseudo Label
這是近年來在半監督學習(semi-supervised learning)非常常用的技術，主要用在缺少標記資料的情況，把模型預測的高可信度標籤當作一種「偽標籤」，並再度使用這些偽標籤讓模型學習。Pseudo label 也常應用在 UDA 的模型，因為具有高可信度的預測通常是準確的，這樣模型就能更好的學習到 target domain 的 feature。
現在應用 pseudo label 的方式有非常多種，這裏筆者舉一個最常見的步驟來介紹：
1. 使用**有標記**的資料訓練模型
2. 讓模型對**無標記**的資料做預測，並選擇預測機率高過某個值的樣本當作 pseudo label
3. 使用有標記的資料和 pseudo label 訓練新的模型
4. 重複 2 跟 3 直到模型表現不再提升
{% endnote %}

{% note info no-icon %}
### Knowledge Distillation

Konwledge Distillation（知識蒸餾）在論文 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) 被正式提出和應用，其概念就是透過一個已訓練好的網路（teacher network）來教導一個較小的網路來學習（student network），而學習的方法常常是讓學生去學習老師的 soft predictions，會在最後一層使用加上參數 temperature $T$ 的 softmax function，當 $T$ 越高，會使概率分佈越平均，選擇適當的 $T$，就能讓學生學習到除了 hard label 以外的重要資訊。
$$q_i=\frac{exp(z_i/T)}{\sum_j exp(z_j/T)}$$
這樣把大網路的知識傳給小網路的做法，能讓小網路達到比 train from scratch 還要好的表現，而且小網路本身也更容易應用、不需要太多的硬體資源跟運算資源，在 computer vision, object detection, natural language processing 都能看到知識蒸餾被納入訓練方法之中。
{% endnote %}

# Overview

{% note default no-icon %}
標題：Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation
作者：Pan Zhang, Bo Zhang, Ting Zhang, Dong Chen, Yong Wang1, Fang Wen (University of Science and Technology of China, Microsoft Research Asia)
年份：2021
榮譽：CVPR 2021
{% endnote %}

{% note default no-icon %}
Paper  link: https://arxiv.org/pdf/2101.10979.pdf
Github link: https://github.com/microsoft/ProDA/tree/9ba80c7dbbd23ba1a126e3f4003a72f27d121a1f
{% endnote %}

此篇論文主要的貢獻有四點：
1. 利用 prototype，即時更新 soft pseudo labels，解決 pseudo label noisy 的問題。
2. 提出一種 structure learning 的方法讓 model 能學習到緊湊的 target feature space，解決 model 通常都學習到分散的 target feature 的問題。
3. 把已經 train 好的 UDA model distilled 到 student model 上可以讓分數表現得更好。
4. 目前是 GTA5 $\rarr$ Cityscapes 和 SYNTHIA $\rarr$ Cityscapes 的 SOTA（state-of-the-art）。(2021/08/31)

模型表現：
| Task                            | Class Number | mIoU |
|:--------------------------------|:-------------|:-----|
|GTA $\rarr$ Cityscapes           | 19 classes   | 57.5 |
|Synthia $\rarr$ Cityscapes       | 16 classes   | 55.5 |
|                                 | 13 classes   | 62.0 |

# Methods
在了解方法以前，我們先來弄清楚他想解決什麼問題。
有在關注 UDA 領域都知道，早期很流行使用一些 adversarial learning 的方法讓 source domain feature space 和 target domain feature space 能盡量接近，其中最有名的就是 [AdaptSegNet](https://arxiv.org/abs/1802.10349)。近年則是 self-training 盛行，讓 model 自己產生 pseudo labels 來自我訓練，但這方法目前存在兩個問題：
1. 只選擇 confidence 高於某個嚴格閾值的預測來當作 pseudo label，結果不一定正確，導致 model 在 target domain 訓練失敗。
  <font color="gray">可以看到下圖 (a) 的左邊那張圖就呈現了這問題，正確來講藍色虛線圈裡的應該都要是被標記成 $+$ 的 pseudo label，卻因為 decisoin boundary 的錯誤而被標記成 $-$</font>
2. 由於 source / target domain 差距很大，model 學習到的 target feature 很常是分散的。
  <font color="gray">下圖的 (b) 可以看到即使 target feature 已經是分開的了，但因為彼此太過分散，以至於有近 1/4 的 data 是分類錯誤的。</font>

![](/assets/posts/ProDA/Issue.png)<font color="gray"><center>Figure 1. The existing issues of self-training by visualizing the feature space.</center></font>


針對第一個問題，作者改用每個 pixel 跟 prototype 的相對特徵距離作為依據，並且在訓練過程中及時調整 prototype 跟 pseudo label，想法上更為直觀，pseudo label 也不會有過時的問題，但這樣相對計算量就增加許多，後面會介紹作者用了什麼方法來減輕計算量。
  <font color="gray">圖（a）的右邊那張圖就把原本比較接近類別 A 的標記為 $+$，decision boundary 改變，隨後再根據新一輪的 feature 來更新 prototype。</font>


至於第二個問題，作者把一張圖經過兩種不同的轉換方式分別餵進兩個同架構的 model，並要求 model 計算出相似的 prototype 位置，在 [structure-learning-by-enforcing-consistency](#structure-learning-by-enforcing-consistency) 會更詳細解釋他的做法。

{% hideToggle 公式符號表 %}
- $\mathcal{X}_s = \{x_s\}^{n_s}_{j=1}$ : source dataset
- $\mathcal{Y}_s = \{y_s\}^{n_s}_{j=1}$ : source segmentation labels
- $\mathcal{X}_t = \{x_t\}^{n_t}_{j=1}$ : target dataset
- $K$ 個類別
- $h = f \circ g$ : 整體網路架構$h$ = 特徵提取網路$f$ ＋ 分類器$g$
- $f(x_t)^{(i)}$: 第 i 個 target data feature
- $\eta^{(k)}$: prototype, class k 的特徵中心點
- $\xi(\cdot)$: 把 soft prediction 轉成 hard label 的函數表示
{% endhideToggle %}

## Prototypical pseudo label denoising
如果我們讓 pseudo labels 一個訓練階段更新一次，那麼 model 可能早就 overfit 在 noisy labels 上，但同時更新網路參數跟 pseudo labels 又會造成 trivial solution，因此 ProDA 採用**固定 soft predictions（$p_t$），根據與 prototype 的距離在訓練過程中生成給每個類別的權重（$w_t$），並更新 hard pseudo label（$\hat{y}_t$）**。
$$\hat{y}_t^{(i,k)}=\xi(\omega_t^{(i,k)}p_{t,0}^{(i,k)})$$

{% note default no-icon %}
**Soft Predictions & Hard Labels**
Soft prediction 為模型計算出該資料屬於某個類別的機率，hard label 代表模型計算出該資料所屬的類別。
打個比方來說，現在要預測一張圖片屬於貓還是狗，如果模型輸出的是 soft prediction 那就會長這樣： [0.9, 0.1]，hard label 就會是 "0" 或是 "貓"。
{% endnote %}

{% hideToggle 權重公式如下： %}
$$\omega_t^{(i,k)}=\frac{exp(-||\tilde{f}(x_t)^{(i)}-\eta^{(k)}||/\tau}{\sum_{k'}exp(-||\tilde{f}(x_t)^{(i)}-\eta^{(k')}||/\tau)}$$
- $f(x_t)^{(i)}$: 第 i 個 target data feature
- $\tilde{f}$: momentum encoder，$f$ 的緩慢更新版本
- $\eta^{(k)}$: prototype，類別 k 的特徵中心點
- $\tau$: softmax temperature，這裡設為 1

大家看到公式先不要緊張，我們先從概念理解起。
假設現在某個 feature 離 prototype $\eta^{(k)}$ 很遠，表示它較不可能是屬於類別 k，它的權重就會比較小。相反的，如果距離 $\eta^{(k)}$較近，權重就會比較大。

而權重 $\omega_t^{(i,k)}$ 其實就是 feature $\tilde{f}(x_t)^{(i)}$ 跟 prototype $\eta^{(k)}$ 的距離過 softmax function 的結果，分母為所有類別跟 feature 的距離，分子則只算自己的類別。
{% endhideToggle %}

{% hideToggle prototype 公式如下： %}
![](/assets/posts/ProDA/prototype.png)

- indicator function: 符合條件的輸出 1，不符合的為 0。在這條式子的條件就是 pseudo label $\hat{y}_t^{(i,k)} == 1$。 

從公式中可以看到 prototype 其實就是該類別的 feature 平均，也就是中心點（centroid）的概念。但這樣每次計算都要跑過全部的 feature point，計算量龐大，因此 ProDA 是用 mini-batches 的中心點的 moving average（移動平均）取代計算。
$$\eta^{(k)}\leftarrow \lambda\eta^{(k)}+(1-\lambda)\eta'^{(k)}$$
- $\eta'^{(k)}$ : 當下訓練批次類別 k 的中心點，來自 momentum encoder。
- $\lambda$ = 0.9999
{% endhideToggle %}

最後用 pseudo label 跟 model predictions 算出的 symmetric cross-entropy 作為 target domain 的 loss。

$$l^t_{sce}=\alpha l_{ce}(p_t, \hat{y}_t)+\beta l_{ce}(\hat{y}_t, p_t)$$

$\alpha$ 和 $\beta$ 為平衡係數，這裏 $\alpha = 0.1, \beta= 1$。

{% note default no-icon %}
**Symmetric cross-entropy (SCE)**
出自 ICCV 2019 的 [Symmetric Cross Entropy for Robust Learning with Noisy Labels](https://arxiv.org/abs/1908.06112)，作者結合了傳統的 cross-entropy 和 reverse cross entropy (RCE) 得到 SCE，並透過實驗證明使用這樣的損失函數能讓模型對 noisy label 更 robust，也能收斂的更快更好，更詳細的說明請參考[論文](https://arxiv.org/abs/1908.06112)。
{% endnote %}

## Structure learning by enforcing consistency

為了使 target feature 能夠更緊湊，作者對 target data $x_t$ 分別做了弱增強 $\mathcal{T}(x_t)$ 和強增強 $\mathcal{T'}(x_t)$，所謂增強其實是對圖片做一些轉化，而實際上 ProDA 的弱增強就是直接餵原圖給 model，強增強就可能會對圖片做一些旋轉、明暗度調整、彩度調整等等，詳情可以看他們的 [github](https://github.com/microsoft/ProDA/blob/9ba80c7dbbd23ba1a126e3f4003a72f27d121a1f/data/randaugment.py)。

現在有了兩張圖片 $\mathcal{T}(x_t)$ 和 $\mathcal{T'}(x_t)$ 後，把弱增強的輸入 momentum encoder $\tilde{f}$，強增加的輸入原始的 encoder $f$，讓他們分別計算 prototype 位置 $z_\mathcal{T}$ 和 $z_\mathcal{T'}$（論文中稱他們為 soft prototypical assignment），並迫使 model 去降低這這兩個的 KL-divergence。

$$z_{\mathcal{T}}^{(i,k)}=\frac{exp(-||\tilde{f}(\mathcal{T}(x_t))^{(i)}-\eta^{(k)}||/\tau}{\sum_{k'}exp(-||\tilde{f}(\mathcal{T}(x_t))^{(i)}-\eta^{(k')}||/\tau)}$$
<font color="gray"><center>Soft prototypical assignment formula. (for $z_\mathcal{T'}$, just change $\tilde{f}(\mathcal{T}(x_t)))$ to $f(\mathcal{T'}(x_t))$.)</center></font>

$$l_{kl}^t=KL(z_{\mathcal{T}}||z_{\mathcal{T}'})$$
<font color="gray"><center>KL divergence between the prototypical assignments under two views.</center></font>

由於 $z_\mathcal{T}$ 是由弱增強的圖片計算得來，受到的干擾較小，計算出的 prototype 會較正確，於是我們就用這個 $z_\mathcal{T}$ 去教導原本的 encoder 在吃到強增強的圖片後也能得出一樣的 prototype assignment，就表示説他學習到更穩定、緊湊的 target feature，而這種讓模型學習如何得到跟另一個模型一樣結果的訓練方式，就叫做 **consistent learning**。 

![](/assets/posts/ProDA/model1.png)<font color="gray"><center>Figure 2. Model overview for structure learning by enforcing consistency. (由於論文裡沒有給他們的模型架構圖，這裏筆者自己畫了一張，希望能讓大家更理解他們使用的方法。）</center></font>

{% note default no-icon %}
**Momentum Encoder**
這詞出自於 2020 Facebook AI 發表的 [MoCo](https://arxiv.org/abs/1911.05722)，可以看作是更新比較緩慢的 encoder，每次參數都只會靠近原本的 encoder 一點點，可以參考下列 momentum encoder 更新公式（ProDA momentum encoder 也採用同樣的更新方式）：
$$\theta_{me} = m\theta_{me} + (1-m)\theta_{e}$$
其中 $\theta_{me}$ 為 momentum encoder 的參數，\theta_{e} 為 encoder 的參數，m 是可調整的超參數，通常會設為近似 1 的值。 
{% endnote %}

為了防止在學習 target feature 時出現 degeneration issue （有個類別的 cluster 是空的），需要再加上一個 regularization term，目的是鼓勵模型輸出類別能盡量平均，不要有一個機率總是特別高，或是特別低。
$$l_{reg}^t=-\sum_{i=1}^{H\times W}\sum_{j=1}^Klogp_t^{(i,k)}$$

講到這裡，我們終於把 ProDA 第一個訓練階段的 loss 都講解完啦！以下是最終的 loss function: 

$$l_{total}=l^s_{ce}+l^t_{sce}+\gamma_1l^t_{kl}+\gamma_2l^t_{reg}$$

其中 $\gamma_1 = 10, \gamma_2 = 0.1$。

## Distillation to self-supervised model

在 $l_{total}$ 收斂後，一般模型可能就到此收手，但 ProDA 更結合了知識蒸餾的概念，讓學生模型 (student model) 向老師模型（teacher network, 同時也是第一階段使用的 model）學習。雖然在這裡學生模型是跟老師模型有**完全相同的架構**（一般知識蒸餾中學生模型會比老師模型還要小），唯一的差別在學生模型有先使用 [Sim-CLRv2](https://arxiv.org/abs/2006.10029) pretrained weights 初始化。

看到這裡可能會有點矇，所以 ProDA 到底有幾個模型？訓練階段是什麼？
這裡簡單畫個模型圖跟大家做解說。

![](/assets/posts/ProDA/model2.png)<font color="gray"><center>Figure 3. ProDA model overview - stage 1 （註：此圖也是 $l^t_{sce}$ 是如何被計算出的模型架構圖，想更了解的讀者可以對照著 [Prototypical pseudo label denoising](#prototypical-pseudo-label-denoising) 看。</center></font>

ProDA 整個訓練總共有三個階段，階段一的目的在使 $l_{total}$ 收斂，模型架構圖會長得像上方 Figure 3 的樣子（這裡只畫了一個 loss 做為代表），而下方被灰色虛線匡著的 encoder $f$ 和 classifier $g$ 就是我們主要的 segmentation network $h$，這個 $h$ 也是在接下來的知識蒸餾過程中被視為 teacher network 的網路。

![](/assets/posts/ProDA/model3.png)<font color="gray"><center>Figure 4. ProDA model overview - stage 2 + 3. 咖啡色的箭頭代表跟 source domain 有關，黑色的箭頭則跟 target domain 有關。</center></font>

階段二和三則是知識蒸餾的過程，我們會拿階段一訓練好的模型當作 teacher model，把知識透過**降低 knowledge distillation loss $l_{KD}$** 來傳給 student model $h^\dag$。作法除了降低 source domain 的 cross entropy 外，還包含 student model predictions 和 teacher network hard labels 的 cross entropy 跟這兩個 model predictions 的 KL-divergence。 

Konwledge distillation loss
$$l_{KD}=l^s_{ce}(p_s,y_s)+l^t_{ce}(p^\dag_t, \xi(p_t))+\beta KL(p_t||p_t^\dag)$$

# Experiments
## Main Results
ProDA 不論是在 GTA5 $\rarr$ Cityscapes 還是 SYNTHIA $\rarr$ Cityscapes 的任務上都是表現最好的。他們使用的是 ResNet-101 + Deeplab-v2 的架構，需要四片 Tesla V100 GPU 來訓練，雖然作者並未公布他們訓練的時間，不過就筆者自己訓練的經驗來看大概需要兩天多的時間，有興趣的讀者可以去 [Github](https://github.com/microsoft/ProDA/tree/9ba80c7dbbd23ba1a126e3f4003a72f27d121a1f) 把模型載下來跑跑看。對這些 datasets 不熟悉的可以看[這裡](https://blog.wazenmai.com/%E8%AB%96%E6%96%87%E7%AD%86%E8%A8%98-Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/#datasets-information)

### GTA5 - Cityscapes
![](/assets/posts/ProDA/GTA5.png)
### STNTHIA - Cityscapes
![](/assets/posts/ProDA/Synthia.png)

## Ablation Study
### Effectiveness of the Methods
![](/assets/posts/ProDA/ablation.png)<font color="gray"><center>Figure 5. Ablation study of each proposed component on GTA5-Cityscapes. ST stands for self-training, PD for prototypical denoising, and SL for structure learning.</center></font>

{% tabs Unique name, [index] %} 
<!-- tab Pseudo label denoising -->
- 加上 warm-up : 41.6 mIoU **（+5）**
- 加上 offline pseudo labels : 45.2 mIoU **（+8.6）**
- 加上 symmetric cross-entropy : 45.6 mIoU **（+9.0）**
- 加上 prototypical denoising : 52.3 mIoU **（+15.7）**
<!-- endtab -->
<!-- tab Target structure learning -->
Target structure learning 藉由學習緊湊的 target fature cluster 來協助偽標籤能不受雜訊干擾，並提升 1.4 mIoU。
<!-- endtab -->
<!-- tab Distilling to self-supervised model-->
- self-supervised **>** supervised initialization
- 使用此方法初始化可以避免模型最後收斂在 local optima
- stage 2 + 3 證明了知識蒸餾的有效性
<!-- endtab -->
{% endtabs %}

### The UMAP visualization of target features

![](/assets/posts/ProDA/umap.png)

為了證明他們的 target feature 真的有學得比較好，ProDA 還提供了視覺化的 target features。
（a）單純把訓練在 source domain 的模型拿到 target domain 上做訓練，target features 的分佈。
（b）傳統自訓練方法可以讓 target feature 分開一些，但離可以 linear classfication 的程度還是有些遠。
（c）經過 prototypical denoising 後四個 feature 被分開，已經可以用兩條線大致區分出四個類別，但同類別的 feature 還是分散的。
（d）可以看出，ProDA 比起左邊的三個版本，可以更好的分開不同類別的 feature，同類別的 feature 也比較聚集。

# Conclusion
筆者在這篇文章中帶大家了解目前 UDA semantic segmentation 領域表現最好的模型是用了哪些方法，也詳細介紹了各個公式，希望大家對 ProDA 有更深的了解。
在 ProDA 提出的方法中，筆者認為知識蒸餾的概念最為有趣，已經訓練好的模型經過近一步的「蒸餾」後竟然能讓表現變得更好，不知道現階段其他的模型架構是否也能透過同樣的方式讓表現有所提升？就等大家來嘗試啦！

Reference：
  - https://zhuanlan.zhihu.com/p/102038521
  - [Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Segmantic Segmentation](https://arxiv.org/pdf/2101.10979.pdf)

{% note blue 'far fa-snowflake' modern %}
有任何問題都歡迎在下面提出，喜歡這篇文章的話可以幫我點一個讚，
祝福各位能在機器學習領域走出屬於自己的一條路。
{% endnote %}
