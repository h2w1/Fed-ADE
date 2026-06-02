# Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift

> **CVPR 2026** | [📄 Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Park_Fed-ADE_Adaptive_Learning_Rate_for_Federated_Post-adaptation_under_Distribution_Shift_CVPR_2026_paper.html) | [💻 Code](https://github.com/h2w1/Fed-ADE)

**Heewon Park, Mugon Joe, Miru Kim, Kyungjin Im, Minhae Kwon†

Sungkyunkwan University
(† Corresponding author: minhae.kwon@skku.edu)

---

## Overview

Federated learning (FL) models deployed on edge devices — smartphones, IoT sensors, autonomous systems — continuously encounter **non-stationary data streams** that degrade performance over time. The core challenge: how do you choose the right learning rate for each client, at each timestep, *without ground-truth labels*?

**Fed-ADE** (Federated Adaptation with Distribution Shift Estimation) solves this with two lightweight, label-free estimators that jointly detect *how fast* each client's data is shifting — and automatically sets the learning rate accordingly.

<p align="center">
  <img src="assets/overview.png" alt="Fed-ADE Overview" width="85%"/>
</p>

> Each client receives a pre-trained model, splits it into **shared** (ψ_c) and **personalized** (φ_c) layers, and adapts to unlabeled, distribution-shifting data using an adaptive learning rate driven by distribution dynamics estimation.

---

## Key Idea

Fed-ADE introduces a **Distribution Dynamics Signal** S_c^t ∈ [0, 1] that combines two complementary estimators:

### 1. Uncertainty Dynamics Estimation (S_unc)
Tracks temporal changes in the model's predictive confidence by measuring cosine distance between consecutive batch-averaged softmax vectors:

$$\mathcal{S}^t_{\text{unc}} = 1 - \cos(\mathbf{q}_c^{t-1},\ \mathbf{q}_c^t) \in [0, 1]$$

- Requires storing only the previous batch mean → **O(|I|) memory**
- Captures label distribution shift without any labels

### 2. Representation Dynamics Estimation (S_rep)
Detects embedding-level feature drift via cosine distance between consecutive ℓ₂-normalized batch feature means:

$$\mathcal{S}^t_{\text{rep}} = \frac{1}{2}\left(1 - \cos(\mathbf{z}_c^{t-1},\ \mathbf{z}_c^t)\right) \in [0, 1]$$

- Only stores the previous batch feature mean → **O(d) memory**
- Sensitive to covariate shift invisible to label-space signals

### Adaptive Learning Rate
The two signals are averaged into a unified dynamics signal, which linearly maps to a per-client, per-timestep learning rate:

$$\mathcal{S}^t_c = \frac{1}{2}(\mathcal{S}^t_{\text{unc}} + \mathcal{S}^t_{\text{rep}}), \qquad \eta^t_c = \eta_{\min} + (\eta_{\max} - \eta_{\min})\,\mathcal{S}^t_c$$

| Signal | Low (≈ 0) | High (≈ 1) |
|--------|-----------|------------|
| **S_c^t** | Stable distribution → small η (conservative update) | Strong shift → large η (fast adaptation) |

---

## Theoretical Guarantees

Fed-ADE comes with three formal results:

| Theorem | Result |
|---------|--------|
| **Thm. 1** — Uncertainty Surrogate Error | S̄_unc tracks true label distribution path length up to O(Σ φ_t) calibration error |
| **Thm. 2** — Representation Surrogate Error | S̄_rep tracks feature-level drift up to O(K_h · Σ φ'_t) Lipschitz error |
| **Thm. 3** — Dynamic Regret Bound | E[Reg_T] = **O(S̄_c^{1/3} · T^{2/3})**, matching the min–max optimal rate for non-stationary online learning |

The adaptive learning rate rule achieves **min–max optimal dynamic regret** — no fixed learning rate can do better in the worst case.

---

## Simulation Results

### Datasets & Shift Scenarios

| Scenario | Datasets | Shift Schedules |
|----------|----------|-----------------|
| **Label Shift** | Tiny ImageNet, CIFAR-10, LAMA (text) | Linear, Sine, Square, Bernoulli |
| **Covariate Shift** | CIFAR-10-C, CIFAR-100-C | Linear, Sine, Square, Bernoulli |

All experiments: **100 clients**, **100 timesteps**, unlabeled adaptation data only.

---

### Main Results (Table 1)

**Label Shift — Average Accuracy (%)**

| Dataset | Shift | FTH | ATLAS | Fed-POE | FedCCFA | FixLR(Mid) | **Fed-ADE** |
|---------|-------|-----|-------|---------|---------|-----------|------------|
| Tiny ImageNet | Lin. | 78.2 | 76.5 | 87.1 | 84.7 | 88.2 | **89.1** |
| Tiny ImageNet | Sin. | 77.9 | 76.8 | 87.5 | 84.8 | 88.0 | **88.9** |
| Tiny ImageNet | Squ. | 77.2 | 78.5 | 86.4 | 83.0 | 88.2 | **88.9** |
| Tiny ImageNet | Ber. | 78.2 | 77.6 | 86.5 | 83.8 | 87.8 | **88.7** |
| CIFAR-10 | Lin. | 31.4 | 36.5 | 71.3 | 65.8 | 70.8 | **73.8** |
| CIFAR-10 | Sin. | 40.3 | 43.7 | 71.4 | 65.8 | 70.5 | **73.6** |
| LAMA | Lin. | 68.3 | 79.5 | 85.4 | 95.6 | 95.2 | **95.8** |
| LAMA | Squ. | 70.5 | 79.8 | 84.2 | 92.0 | 95.4 | **96.4** |

**Covariate Shift — Average Accuracy (%)**

| Dataset | Shift | UDA | UNIDA | Fed-POE | FedCCFA | FixLR(Mid) | **Fed-ADE** |
|---------|-------|-----|-------|---------|---------|-----------|------------|
| CIFAR-10-C | Lin. | 45.7 | 43.3 | 44.5 | 43.1 | 63.9 | **64.4** |
| CIFAR-10-C | Ber. | 42.3 | 42.7 | 48.7 | 42.0 | 64.4 | **65.8** |
| CIFAR-100-C | Lin. | 35.6 | 34.3 | 27.3 | 24.3 | 43.4 | **45.8** |
| CIFAR-100-C | Sin. | 35.7 | 34.1 | 27.5 | 24.5 | 42.1 | **46.7** |

**Wall Time (sec.)**

| FTH | ATLAS | Fed-POE | FedCCFA | **Fed-ADE** |
|-----|-------|---------|---------|------------|
| 2631 | 1959 | 131 | 211 | **109** |

> Fed-ADE achieves the **best accuracy across all settings** while running **17–24× faster** than localized methods and **~2× faster** than FedCCFA.

---

### Robustness to Pre-training Distribution (Table 2)

Fed-ADE maintains strong performance even when the pre-training data distribution is non-uniform (Gaussian or Exponential Decay), with only minor accuracy variation compared to the uniform baseline.

| Distribution Shift | Fed-ADE | FedCCFA | Fed-POE |
|-------------------|---------|---------|---------|
| Gaussian — Lin. | **69.2** | 62.3 | 65.7 |
| Gaussian — Ber. | **67.1** | 63.0 | 60.9 |
| Exp. Decay — Lin. | **67.4** | 54.1 | 61.1 |
| Exp. Decay — Ber. | **66.7** | 53.0 | 57.5 |

---

### Ablation: Similarity Measure (Table 3)

Cosine similarity outperforms KL divergence, Wasserstein distance, and Bayesian CPD across all shift schedules on CIFAR-10, due to its bounded, direction-based nature that is robust to noisy pseudo-labels and class imbalance.

| Measure | Lin. | Sin. | Squ. | Ber. |
|---------|------|------|------|------|
| **Cosine (Fed-ADE)** | **73.8** | **73.6** | **72.2** | **72.9** |
| KL Divergence | 63.2 | 62.4 | 61.1 | 61.4 |
| Wasserstein | 70.8 | 69.9 | 68.8 | 66.5 |
| Bayesian CPD | 71.5 | 70.8 | 69.7 | 69.0 |

---

### Ablation: Estimator Components (Table 4)

Both S_unc and S_rep are necessary — removing either one hurts performance, and a fixed learning rate performs worst with the highest variance.

| S_unc | S_rep | Variant | Label Shift (%) | Covariate Shift (%) |
|-------|-------|---------|----------------|---------------------|
| ✓ | ✓ | **Fed-ADE (full)** | **73.8 ± 0.6** | **64.4 ± 0.2** |
| ✗ | ✓ | w/o S_unc | 71.3 ± 0.3 | 64.2 ± 0.1 |
| ✓ | ✗ | w/o S_rep | 73.1 ± 0.4 | 64.0 ± 0.3 |
| ✗ | ✗ | Fixed LR | 70.8 ± 2.1 | 63.9 ± 0.3 |

---

## Method Summary

```
Fed-ADE Algorithm (per round r):
  ┌─ For each client c:
  │   1. Build unsupervised risk F̂_c^t  (BBSE label estimation)
  │   2. Compute batch summaries q_c^t (softmax mean), z_c^t (feature mean)
  │   3. Estimate S_unc^t (uncertainty dynamics)
  │   4. Estimate S_rep^t (representation dynamics)
  │   5. Combine → S_c^t = (S_unc^t + S_rep^t) / 2
  │   6. Set adaptive LR: η_c^t = η_min + (η_max − η_min) · S_c^t
  │   7. Update shared + personalized layers {ψ_c, φ_c}
  │   8. Send shared layers ψ_c to server
  └─
  Server: Aggregate ψ̄ = weighted avg of {ψ_c}
  ┌─ For each client c:
  │   1. Replace ψ_c ← ψ̄
  │   2. Update personalized layers φ_c only
  │   3. Cache (q_c^t, z_c^t) for next round
  └─
```

---

## Citation

```bibtex
@InProceedings{Park_2026_CVPR,
    author    = {Park, Heewon and Joe, Mugon and Kim, Miru and Im, Kyungjin and Kwon, Minhae},
    title     = {Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2026},
    pages     = {24587-24597}
}
```

