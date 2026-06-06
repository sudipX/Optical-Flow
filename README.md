# Optical Flow Estimation: A Comparative Study of Classical and Deep Learning Methods

A self-directed comparative study of three optical flow estimation methods evaluated on autonomous driving scenes from the KITTI 2015 dataset. This project implements Lucas-Kanade (sparse, classical), Farneback (dense, classical), and RAFT (dense, deep learning) from first principles in Python, and benchmarks them using the standard End-Point Error (EPE) metric alongside structured qualitative failure analysis.

---

## Table of Contents

- [Overview](#overview)
- [Methods Implemented](#methods-implemented)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Results Summary](#results-summary)
- [Failure Mode Analysis](#failure-mode-analysis)
- [Key Findings](#key-findings)
- [Setup and Installation](#setup-and-installation)
- [Running the Notebooks](#running-the-notebooks)
- [Evaluation Metric](#evaluation-metric)
- [Dependencies](#dependencies)
- [Future Directions](#future-directions)
- [References](#references)
- [Author](#author)

---

## Overview

Optical flow estimates the per-pixel apparent velocity field between two consecutive video frames. It is a foundational problem in computer vision with direct applications in autonomous driving, action recognition, video stabilization, and visual odometry.

The core research question this project addresses:

> **How do classical and deep learning optical flow methods compare on real autonomous driving scenes in terms of accuracy, speed, and failure characteristics?**

All implementations are built from first principles using OpenCV 4.x for classical methods and PyTorch 2.x with the official RAFT repository for the deep learning component. No high-level flow libraries were used. The complete implementation is organized as three annotated Jupyter notebooks.

---

## Methods Implemented

### 1. Lucas-Kanade (Sparse, Classical)

Lucas-Kanade resolves the aperture problem by assuming constant flow within a local pixel window. The method solves a least-squares system using spatial and temporal image gradients:

```math
A^T A [u, v]^T = -A^T b
```

Key implementation details:
- Shi-Tomasi corner detection as a preprocessing step to find well-conditioned keypoints
- Pyramid-based (Bouguet) extension with 3 Gaussian pyramid levels to handle larger displacements
- Integration window size of 15x15 pixels
- Produces **sparse** flow only at detected corner locations

### 2. Farneback (Dense, Classical)

Farneback achieves dense flow estimation by approximating each pixel neighborhood with a quadratic polynomial and equating polynomial coefficients between frames to compute displacement:

```math
d = -(A1 + A2)^-1 (b2 - b1)
```

Key implementation details:
- Pyramid scale: 0.5
- Pyramid levels: 3
- Window size: 15
- Polynomial neighborhood: n=5, sigma=1.2
- Produces a **dense** flow field covering every pixel

### 3. RAFT (Dense, Deep Learning)

RAFT (Recurrent All-Pairs Field Transforms) introduces three core architectural innovations:

**4D Correlation Volume:** Feature maps are extracted by a shared ResNet encoder. An all-pairs dot product produces a 4D correlation volume over all source-target pixel pairs, enabling the model to reason about displacements of any magnitude without pyramid-based assumptions.

**Recurrent Update Operator:** A ConvGRU iteratively refines the flow estimate by attending to the correlation volume at the current displacement and computing a flow correction at each step. This study uses K=20 refinement iterations at inference time.

**Single-resolution processing:** Unlike prior pyramid networks, RAFT operates entirely at 1/8th resolution for features, avoiding errors introduced by coarse-to-fine warping.

Key implementation details:
- Official Princeton pretrained weights fine-tuned on KITTI (`raft-kitti.pth`, 5.3M parameters)
- DataParallel prefix stripping for weight loading
- Images padded to multiples of 8 prior to inference and unpadded post-inference

---

## Dataset

**KITTI 2015 Optical Flow Benchmark**

The KITTI 2015 dataset provides stereo image pairs captured from a camera mounted on a driving vehicle in Karlsruhe, Germany. Ground truth flow is provided as sparse annotations derived from a 3D LiDAR sensor projected into image space, encoded in 16-bit PNG format.

Ground truth flow is decoded as:

```math
f_u = \frac{I_R - 2^{15}}{64}
```

```math
f_v = \frac{I_G - 2^{15}}{64}
```

The validity mask (blue channel IB > 0) marks pixels with reliable ground truth annotations. This study evaluates on 194 frame pairs from the training split (`training/image_1`), spanning urban roads, parked and moving vehicles, pedestrians, and background structures.

You can download the dataset from the official KITTI website: [https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

---

## Project Structure

```
optical-flow-comparative-study/
|
|-- classical_flow.ipynb          # Lucas-Kanade and Farneback implementations
|-- raft_implementation.ipynb     # RAFT deep learning implementation and inference
|-- comparative_analysis.ipynb    # Quantitative evaluation, EPE computation, and visualizations
|-- findings.md                   # Detailed findings and analysis write-up
|-- README.md                     # This file
|-- Optical_Flow_CaseStudy.pdf    # Full technical report
|-- OpticalFlow.pdf               # A full comprehensive study material
|

```

---

## Results Summary

Quantitative results averaged over 194 KITTI pairs. EPE = End-Point Error in pixels (lower is better). Inference time averaged per pair on an NVIDIA T4 GPU.

| Method | Type | Avg EPE | Avg Inference Time | Parameters |
|---|---|---|---|---|
| Lucas-Kanade | Classical Sparse | N/A (sparse output) | ~10 ms | 0 |
| Farneback | Classical Dense | 22.99 px | ~150 ms | 0 |
| RAFT | Deep Learning Dense | 2.14 px | ~380 ms | 5.3M |

Key takeaways from the numbers:
- RAFT achieves roughly **10.7x lower EPE** than Farneback on the same scenes
- Lucas-Kanade is approximately **26x faster** than RAFT but produces only sparse flow, making direct EPE comparison inapplicable
- The Farneback vs. RAFT comparison is the relevant one for dense applications: roughly **3x speed penalty** for substantially better accuracy
- RAFT requires GPU acceleration for practical throughput; classical methods run efficiently on CPU

---

## Failure Mode Analysis

Three failure categories were systematically identified and spatially localized across KITTI scenes.

### Failure Mode 1: Large Motion

Regions where inter-frame displacement exceeds roughly 20 pixels correspond to fast-moving vehicles. Lucas-Kanade breaks down here because the pyramid assumption that motion is small at each scale fails at high velocities. Farneback produces blurred, averaged flow. RAFT correctly segments vehicle boundaries due to its all-pairs 4D correlation volume, which explicitly models displacements of any magnitude.

### Failure Mode 2: Texture-less Regions

Approximately 15% of KITTI pixels fall in regions with image gradient magnitude below the 15th percentile, including the sky, flat road surface, and uniform vehicle panels. Lucas-Kanade detects no Shi-Tomasi corners in these areas and provides no flow estimate. Farneback produces noisy polynomial fits. RAFT produces spatially smooth but potentially incorrect flow by leveraging its context encoder to propagate information from nearby textured regions, a form of learned spatial regularization absent in classical methods.

### Failure Mode 3: Occlusion Boundaries

Regions where inter-frame brightness change exceeds the 85th percentile identify likely occlusion or dis-occlusion events. All methods exhibit increased error here. Farneback shows characteristic boundary blurring due to polynomial smoothness assumptions. Lucas-Kanade loses tracking near occluded boundaries. RAFT maintains sharper boundaries through learned feature matching but is not explicitly occlusion-aware in the version evaluated.

The EPE difference map confirms that RAFT's advantage over Farneback is concentrated at occlusion boundaries and large-displacement regions (>15 px), precisely the conditions most critical for autonomous driving safety.

---

## Key Findings

**Finding 1: Speed-Accuracy Trade-off is Non-Linear**

The relevant dense comparison (Farneback vs. RAFT) shows a roughly 3x speed penalty for a 10x+ improvement in EPE. For real-time embedded deployment, this motivates research into RAFT distillation (RAFT-Small reduces to 1M parameters with modest accuracy loss).

**Finding 2: Large Motion is the Critical Differentiator**

The performance gap between classical and deep learning dense methods is concentrated in large-displacement regions. These are precisely the scenarios most safety-critical in autonomous driving, directly motivating the application of RAFT-family methods in production perception systems.

**Finding 3: Iterative Refinement is Effective but Non-Linear**

Flow quality saturates rapidly during RAFT's GRU refinement. Iterations 1 through 5 account for the majority of improvement; iterations 6 through 20 yield diminishing returns. This suggests an adaptive iteration strategy could reduce inference time without meaningful accuracy loss.

**Finding 4: Interpretability Remains a Practical Concern**

For safety-critical deployment, the black-box nature of RAFT creates certification and debugging challenges. Classical methods retain practical relevance in constrained environments where mathematical guarantees are required.

---

## Setup and Installation

### Prerequisites

- Python 3.10 or above
- CUDA 11.8 compatible GPU (recommended for RAFT)
- Google Colab (used for original experiments, T4 GPU)

### Clone the Repository

```bash
git clone https://github.com/<your-username>/optical-flow-comparative-study.git
cd optical-flow-comparative-study
```

### Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib numpy scipy
```

### Download RAFT Pretrained Weights

The RAFT model uses official Princeton pretrained weights fine-tuned on KITTI. Download from the official RAFT repository:

```bash
# Clone the official RAFT repo for model architecture files
git clone https://github.com/princeton-vl/RAFT.git

# Download pretrained weights
bash RAFT/download_models.sh
```

Place `raft-kitti.pth` in the `models/` directory.

### Download the KITTI Dataset

Register and download the KITTI 2015 optical flow training data from:
[https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

Extract into the `data/` directory following the structure described in the Project Structure section above.

---

## Running the Notebooks

The notebooks are designed to be run in order. Each is self-contained with detailed inline annotations.

### 1. Classical Flow (`classical_flow.ipynb`)

Implements and visualizes Lucas-Kanade and Farneback optical flow on KITTI image pairs.

- Loads KITTI frame pairs
- Runs Shi-Tomasi corner detection
- Computes pyramid-based Lucas-Kanade sparse flow and visualizes tracked points with displacement arrows
- Computes Farneback dense flow and visualizes with HSV color encoding and magnitude heatmaps
- Saves flow outputs for downstream evaluation

### 2. RAFT Implementation (`raft_implementation.ipynb`)

Loads the pretrained RAFT model and runs inference on KITTI image pairs.

- Handles image preprocessing (padding to multiples of 8)
- Loads RAFT weights with DataParallel prefix stripping
- Runs 20-iteration GRU refinement
- Visualizes iterative refinement progression (iterations 1, 3, 5, 10, 15, 20)
- Saves predicted flow fields for evaluation

### 3. Comparative Analysis (`comparative_analysis.ipynb`)

Loads ground truth annotations and computes quantitative and qualitative comparisons.

- Parses 16-bit KITTI ground truth flow PNG files
- Computes EPE for Farneback and RAFT over all valid pixels
- Generates side-by-side HSV flow visualizations
- Generates EPE difference maps (RAFT vs. Farneback)
- Performs structured failure mode analysis across three categories
- Produces summary bar charts for EPE and inference time

---

## Evaluation Metric

**End-Point Error (EPE)** measures the mean Euclidean distance between predicted and ground truth flow vectors over all valid annotated pixels:

```math
\text{EPE} = \frac{1}{|V|} \sum_{(x,y)\in V}
\sqrt{(u_{\text{pred}} - u_{\text{gt}})^2 + (v_{\text{pred}} - v_{\text{gt}})^2}
```

where V is the set of pixels with valid ground truth annotations (validity mask IB > 0). Lower EPE indicates higher accuracy.

EPE is only computable for dense methods (Farneback and RAFT). Lucas-Kanade produces a sparse output and cannot be evaluated with this metric directly.

---

## Dependencies

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1 |
| OpenCV | 4.8 |
| CUDA | 11.8 |
| NumPy | 1.24+ |
| Matplotlib | 3.7+ |
| SciPy | 1.10+ |

All experiments were originally conducted on Google Colab with an NVIDIA T4 GPU (16 GB VRAM).

---

## Future Directions

This study reveals several open research questions for further investigation:

**Transformer-based Architectures**
FlowFormer replaces RAFT's cost volume lookup with a Transformer encoder over the correlation volume, enabling global reasoning. A natural extension is evaluating whether attention-based cost encoding improves performance in the texture-less and large-motion regions identified as challenging for RAFT.

**Real-time Model Distillation**
Can RAFT be distilled into a lighter student model that preserves its large-motion advantages while meeting real-time constraints for edge deployment in autonomous vehicles? RAFT-Small (1M parameters) is a starting point, but custom architectures tailored to specific failure modes may offer better trade-offs.

**Scene-adaptive Method Selection**
The failure analysis shows that classical and deep methods have complementary failure modes. An intelligent scene-adaptive router that selects the appropriate method based on estimated motion magnitude could achieve both speed and accuracy across diverse deployment conditions.

**Explicit Occlusion Handling**
The evaluated RAFT version is not explicitly occlusion-aware. Incorporating occlusion estimation as an auxiliary output could directly address the third failure mode identified in this study.

---

## References

1. B. D. Lucas and T. Kanade, "An iterative image registration technique with an application to stereo vision," Proc. DARPA Image Understanding Workshop, pp. 121-130, 1981.
2. G. Farneback, "Two-frame motion estimation based on polynomial expansion," Proc. SCIA, LNCS vol. 2749, pp. 363-370, 2003.
3. Z. Teed and J. Deng, "RAFT: Recurrent all-pairs field transforms for optical flow," Proc. ECCV, pp. 402-419, 2020.
4. M. Menze and A. Geiger, "Object scene flow for autonomous vehicles," Proc. IEEE CVPR, pp. 3061-3070, 2015.
5. J. Shi and C. Tomasi, "Good features to track," Proc. IEEE CVPR, pp. 593-600, 1994.
6. J.-Y. Bouguet, "Pyramidal implementation of the affine Lucas Kanade feature tracker," Intel Corporation Tech. Rep., 2001.
7. B. K. P. Horn and B. G. Schunck, "Determining optical flow," Artificial Intelligence, vol. 17, pp. 185-203, 1981.
8. Z. Huang et al., "FlowFormer: A transformer architecture for optical flow," Proc. ECCV, pp. 668-685, 2022.
9. D. Sun et al., "PWC-Net: CNNs for optical flow using pyramid, warping, and cost volume," Proc. IEEE CVPR, pp. 8934-8943, 2018.
10. A. Dosovitskiy et al., "FlowNet: Learning optical flow with convolutional networks," Proc. IEEE ICCV, pp. 2758-2766, 2015.

---

## Author

**Sudip Basnet**
078BCT092
Pulchowk Campus, IOE, Tribhuvan University
078bct092.sudip@pcampus.edu.np