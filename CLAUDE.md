# DeepLearning Practice

## 프로젝트 목적

PyTorch로 딥러닝 모델을 직접 구현하고 학습시키며 핵심 개념을 체득하는 것이 목표다.
논문을 읽고 → scratch 구현 → 학습 → 실험의 사이클을 반복한다.

## 학습자 배경

- MLP를 순수 텐서 연산(`torch.tensor`, matmul 등)으로 구현한 경험 있음
- `nn.Module` 기반 PyTorch API는 아직 익히는 중
- CNN, RNN 구현 경험 없음 → Phase 0부터 시작

## 구현 패턴 (모든 모델 공통)

```
1. 논문 읽기    →  abstract + architecture 섹션 위주
2. 구현         →  PyTorch scratch (nn.Module 기반)
3. 학습 확인    →  작은 데이터셋으로 overfitting 먼저 확인
4. 실험         →  hyperparameter 조정, ablation
```

## 디렉토리 구조

```
DeepLearning_Practice/
├── phase0_foundations/
│   ├── 01_mlp/
│   ├── 02_cnn_resnet/
│   └── 03_lstm_gru/
├── phase1_transformer/
│   ├── 04_transformer/
│   ├── 05_t5/
│   ├── 06_bert/
│   └── 07_gpt/
├── phase2_vision/
│   ├── 08_vit/
│   ├── 09_mae/
│   ├── 10_simclr_moco/
│   └── 11_clip/
├── phase3_generative/
│   ├── 12_ae/
│   ├── 13_vae/
│   ├── 14_realnvp/
│   ├── 15_gan_dcgan/
│   ├── 16_ddpm/
│   └── 17_ddim/
├── phase4_rl/
│   ├── 18_dqn/
│   ├── 19_reinforce/
│   ├── 20_ppo/
│   └── 21_sac/
└── phase5_llm/
    ├── 22_llm_finetuning/
    ├── 23_lora_peft/
    ├── 24_rlhf_dpo/
    └── 25_mamba/
```

각 모델 폴더 구성 (Colab 기준):
```
<model>/
└── <번호>_<모델명>.ipynb   # 모델 정의 + 학습 + 실험 전부
```

notebook 내부 셀 구성:
```
[1] 라이브러리 import
[2] 모델 정의
[3] 데이터셋 로딩
[4] 학습 루프
[5] 결과 시각화
```

notebook 이름 목록:
```
phase0_foundations/
├── 01_mlp/           → 01_mlp.ipynb
├── 02_cnn_resnet/    → 02_cnn_resnet.ipynb
└── 03_lstm_gru/      → 03_lstm_gru.ipynb

phase1_transformer/
├── 04_transformer/   → 04_transformer.ipynb
├── 05_t5/            → 05_t5.ipynb
├── 06_bert/          → 06_bert.ipynb
└── 07_gpt/           → 07_gpt.ipynb

phase2_vision/
├── 08_vit/           → 08_vit.ipynb
├── 09_mae/           → 09_mae.ipynb
├── 10_simclr_moco/   → 10_simclr_moco.ipynb
└── 11_clip/          → 11_clip.ipynb

phase3_generative/
├── 12_ae/            → 12_ae.ipynb
├── 13_vae/           → 13_vae.ipynb
├── 14_realnvp/       → 14_realnvp.ipynb
├── 15_gan_dcgan/     → 15_gan_dcgan.ipynb
├── 16_ddpm/          → 16_ddpm.ipynb
└── 17_ddim/          → 17_ddim.ipynb

phase4_rl/
├── 18_dqn/           → 18_dqn.ipynb
├── 19_reinforce/     → 19_reinforce.ipynb
├── 20_ppo/           → 20_ppo.ipynb
└── 21_sac/           → 21_sac.ipynb

phase5_llm/
├── 22_llm_finetuning/ → 22_llm_finetuning.ipynb
├── 23_lora_peft/      → 23_lora_peft.ipynb
├── 24_rlhf_dpo/       → 24_rlhf_dpo.ipynb
└── 25_mamba/          → 25_mamba.ipynb
```

## 전체 커리큘럼

### Phase 0 — 기반
| # | 모델 | 핵심 개념 | 데이터셋 |
|---|------|----------|---------|
| 1 | MLP | nn.Module API, optimizer, loss | MNIST |
| 2 | CNN / ResNet | conv, pooling, skip connection, BatchNorm | CIFAR-10 |
| 3 | LSTM / GRU | hidden state, vanishing gradient | 텍스트 분류 |

### Phase 1 — Attention & Transformer
| # | 모델 | 핵심 개념 | 데이터셋 |
|---|------|----------|---------|
| 4 | Transformer | multi-head attention, positional encoding | WMT |
| 5 | T5 | encoder-decoder, prefix LM | 번역/요약 |
| 6 | BERT | masked LM, encoder-only | WikiText |
| 7 | GPT | causal masking, autoregressive | tiny Shakespeare |

### Phase 2 — Vision & Multimodal
| # | 모델 | 핵심 개념 | 데이터셋 |
|---|------|----------|---------|
| 8 | ViT | patch embedding | CIFAR-100 |
| 9 | MAE | masked image modeling | ImageNet-subset |
| 10 | SimCLR / MoCo | contrastive self-supervised | STL-10 |
| 11 | CLIP | dual encoder, zero-shot | CC3M subset |

### Phase 3 — Generative Models
| # | 모델 | 핵심 개념 | 데이터셋 |
|---|------|----------|---------|
| 12 | AE | bottleneck representation | MNIST |
| 13 | VAE | reparameterization trick, ELBO | CelebA |
| 14 | RealNVP | normalizing flow, change of variables | 2D toy / CelebA |
| 15 | GAN / DCGAN | adversarial training, mode collapse | CelebA |
| 16 | DDPM | forward/reverse diffusion, noise schedule | CIFAR-10 |
| 17 | DDIM | deterministic sampling | — |

### Phase 4 — Reinforcement Learning
| # | 모델 | 핵심 개념 | 환경 |
|---|------|----------|-----|
| 18 | DQN | Q-learning, replay buffer, target network | CartPole / Atari |
| 19 | REINFORCE | policy gradient, baseline | LunarLander |
| 20 | PPO | clipped objective, GAE | MuJoCo / Gym |
| 21 | SAC | entropy regularization, off-policy | continuous control |

### Phase 5 — LLM & Modern
| # | 모델/기법 | 핵심 개념 |
|---|----------|---------|
| 22 | LLM fine-tuning | instruction tuning, SFT |
| 23 | LoRA / PEFT | low-rank adaptation |
| 24 | RLHF / DPO | reward model, alignment |
| 25 | Mamba | selective SSM, linear-time sequence |

## 현재 진행 상태

- [ ] Phase 0 — 기반
- [ ] Phase 1 — Transformer 계열
- [ ] Phase 2 — Vision & Multimodal
- [ ] Phase 3 — Generative Models
- [ ] Phase 4 — Reinforcement Learning
- [ ] Phase 5 — LLM & Modern

## 핵심 논문 목록

| 모델 | 논문 |
|------|------|
| ResNet | Deep Residual Learning for Image Recognition (He et al., 2015) |
| Transformer | Attention Is All You Need (Vaswani et al., 2017) |
| BERT | BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018) |
| GPT | Language Models are Unsupervised Multitask Learners (Radford et al., 2019) |
| T5 | Exploring the Limits of Transfer Learning with T5 (Raffel et al., 2019) |
| ViT | An Image is Worth 16x16 Words (Dosovitskiy et al., 2020) |
| MAE | Masked Autoencoders Are Scalable Vision Learners (He et al., 2021) |
| SimCLR | A Simple Framework for Contrastive Learning (Chen et al., 2020) |
| CLIP | Learning Transferable Visual Models From Natural Language (Radford et al., 2021) |
| VAE | Auto-Encoding Variational Bayes (Kingma & Welling, 2013) |
| RealNVP | Density estimation using Real-valued Non-Volume Preserving (Dinh et al., 2016) |
| GAN | Generative Adversarial Networks (Goodfellow et al., 2014) |
| DDPM | Denoising Diffusion Probabilistic Models (Ho et al., 2020) |
| DDIM | Denoising Diffusion Implicit Models (Song et al., 2020) |
| DQN | Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013) |
| PPO | Proximal Policy Optimization Algorithms (Schulman et al., 2017) |
| SAC | Soft Actor-Critic (Haarnoja et al., 2018) |
| LoRA | LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) |
| DPO | Direct Preference Optimization (Rafailov et al., 2023) |
| Mamba | Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023) |

## 개발 환경

- Google Colab (GPU 런타임 사용)
- Python 3.x, PyTorch (Colab 기본 제공)
- 주요 라이브러리: torchvision, transformers (참고용), datasets, matplotlib, numpy
