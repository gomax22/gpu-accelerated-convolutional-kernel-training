# GPU-Accelerated Convolutional Kernel Training for Vertical Edge Detection
<p align="center">
  <img src="https://github.com/gomax22/gpu-accelerated-convolutional-kernel-training/blob/main/frontcover.jpg" width="750" height="750" />
</p>
## Abstract
Convolutional layers are a key component of several deep learning models, such
as Convolutional Neural Networks (CNNs), which have been particularly successful
in image and video processing tasks, achieving state-of-the-art results.
In this report, we focus on a simplified task, such as vertical edge detection,
for training a convolutional kernel on a synthetic dataset of labeled images,
in which the convolution is performed on a GPU architecture, according to a
proper parallelization strategy.
<p align="center">
  <img src="https://github.com/gomax22/gpu-accelerated-convolutional-kernel-training/blob/main/ParallelizationStrategyConvolution.png" width="750" height="750" />
</p>

---
## Report description
The report covers several theoretical and implementative aspects of the solution, such as:
* **Problem definition and analysis**: definition of convolution in neural networks, convolutional layers and their training phase. 
* **Input and Output**: definition of expected input and output of the problem
* **Algorithm description**: definition of the parallelization strategy for performing the convolution as dot product
* **Implemented routines**: implementation of the proposed parallelization strategy 
* **Performance analysis**: discussion of the experimental results
* **Usage examples**: usage description of the software (some examples are reported)

#### [Link to the report](https://github.com/gomax22/gpu-accelerated-convolutional-kernel-training/blob/main/HPC_Report_Massimiliano_GiordanoOrsini.pdf)


*DISCLAIMER: the provided code can be optimized in order to get a faster implementation. However, this hands-on project leverages on educational purposes: clarifying some concepts about convolutional layers, both from theoretical and practical aspects, and playing with basic CUDA routines.*


