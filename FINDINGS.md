# Research Findings: Classical vs Deep Learning Optical Flow
## A Comparative Study on the KITTI Autonomous Driving Dataset

## Research Question
How do classical (Lucas-Kanade, Farneback) and deep learning (RAFT) optical flow
methods compare on real autonomous driving scenes in terms of accuracy, speed,
and failure characteristics?

## Methods Evaluated

| Method        | Type                  | Parameters | Training Data          |
|---------------|-----------------------|------------|------------------------|
| Lucas-Kanade  | Classical Sparse      | 0          | None                   |
| Farneback     | Classical Dense       | 0          | None                   |
| RAFT          | Deep Learning Dense   | 5.3M       | FlyingChairs + KITTI   |

## Key Quantitative Findings

### Speed (Inference Time)
- Lucas-Kanade : 10.8ms average
- Farneback    : 150.1ms average
- RAFT         : 376.5ms average

### Accuracy (End-Point Error, lower is better)
- Farneback EPE : 22.99px average
- RAFT EPE      : 2.14px average
- RAFT achieves 10.7x lower EPE than Farneback

## Key Qualitative Findings

### Finding 1: Speed-Accuracy Trade-off
Lucas-Kanade is fastest but produces only sparse flow, which is insufficient for
dense scene understanding in autonomous driving. RAFT is slowest but provides the
most complete and accurate flow estimation.

### Finding 2: Large Motion is the Critical Differentiator
Fast-moving vehicles in KITTI create large displacements between frames. Classical
methods struggle here because the brightness constancy assumption breaks down at
large displacements. RAFT's all-pairs correlation handles this without per-frame
assumptions.

### Finding 3: Texture-less Regions Challenge All Methods
Sky, roads, and uniform surfaces remain difficult for all methods. Lucas-Kanade
avoids these regions entirely since no keypoints are detected there. Farneback
produces noisy estimates. RAFT produces smoother results but remains uncertain
in low-texture areas.

### Finding 4: Interpretability vs Performance Trade-off
Classical methods are mathematically transparent; every computation is
interpretable. RAFT achieves better performance but behaves as a black box.
For safety-critical applications like autonomous driving, this interpretability
gap is a genuine research concern.

### Finding 5: Connection to Traffic Flow Modeling
City-scale traffic flow modeling requires understanding motion at multiple scales.
This study suggests RAFT is the appropriate foundation for accuracy-critical
applications, while classical methods may suffice for real-time edge deployment.

## Limitations
- Evaluated on 194 KITTI pairs
- RAFT uses pretrained weights, not trained from scratch
- Lucas-Kanade EPE not computed (sparse flow incompatible with dense GT)
- Adverse conditions such as night driving and rain not evaluated

## Future Directions
1. FlowFormer evaluation: does attention-based cost encoding outperform RAFT's CNN correlation?
2. Conformer-Flow hypothesis: could a hybrid CNN+Transformer architecture improve
   optical flow by combining local CNN features with global attention?
3. Real-time optimization: can RAFT be distilled into a faster model without
   significant accuracy loss for edge deployment?

## Conclusion
This study demonstrates that RAFT substantially outperforms classical methods on
real driving scenes, particularly for large motion. Classical methods retain value
in interpretability and zero-shot generalization. The gap motivates exploration of
Transformer-based architectures like FlowFormer that bring global reasoning to
optical flow estimation.
