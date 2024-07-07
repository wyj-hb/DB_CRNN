## Introduction

![image-20240707132917942](C:/Users/王喻杰/AppData/Roaming/Typora/typora-user-images/image-20240707132917942.png)

This is a PyToch implementation trained jointly by DBNet and CRNN through the Bridge structure. Through the Bridge structure, the recognizer and detector can be connected to improve detection performance while maintaining modularity.

## Main Results

**Total-Text:**

| Method  | Det-P | Det-R | Det-F | E2E-P | E2E-R | E2E-F |
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: |
| DB-CRNN | 0.84  | 0.74  | 0.78  | 0.76  | 0.67  | 0.72  |

## reference

* [Bridge]([2404.04624 (arxiv.org)](https://arxiv.org/pdf/2404.04624))

* [DBnet]([Real-time Scene Text Detection with Differentiable Binarization (arxiv.org)](https://arxiv.org/pdf/1911.08947))

* [CRNN]([[1507.05717\] An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (arxiv.org)](https://arxiv.org/abs/1507.05717))

  
