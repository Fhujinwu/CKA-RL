<h1 align="center">
     <br>Continual Knowledge Adaptation for Reinforcement Learning
<p align="center">
    <a href="https://openreview.net/pdf?id=QRlVickNdN">
        <img alt="Static Badge" src="https://img.shields.io/badge/Paper-NeurIPS-red">
    </a>
</p>

<h4 align="center"></a>
     
>[Jinwu Hu](https://scholar.google.com/citations?user=XmqjPi0AAAAJ&hl=en), [Zihao Lian](https://scholar.google.com/citations?user=mDGNr5cAAAAJ&hl=en), [Zhiqun Wen](https://scholar.google.com/citations?hl=en&user=w_O5JUYAAAAJ), ChenghaoLi, [Guohao Chen](https://scholar.google.com/citations?user=HZbzdNEAAAAJ&hl=en&oi=ao), Xutao Wen, [Bin Xiao](https://faculty.cqupt.edu.cn/xiaobin/zh_CN/index.htm), [Mingkui Tan](https://tanmingkui.github.io/)\
<sub>South China University of Technology, Pazhou Laboratory</sub>

<p align="center">
  <img src="./assets/CKA_RL.png" alt="CKA-RL" width="700" align="center">
</p>

> **TL;DR:** **CKA-RL** introduces a **Continual Knowledge Adaptation** framework for reinforcement learning that maintains a **pool of task-specific knowledge vectors** and **dynamically reuses historical knowledge** to adapt to new tasks.  

---

## ✨ Abstract
Reinforcement Learning enables agents to learn optimal behaviors through interactions with environments. However, real-world environments are typically non-stationary, requiring agents to continuously adapt to new tasks and changing conditions. Although Continual Reinforcement Learning facilitates learning across multiple tasks, existing methods often suffer from catastrophic forgetting and inefficient knowledge utilization. To address these challenges, we propose **Continual Knowledge Adaptation for Reinforcement Learning (CKA-RL)**, which enables the accumulation and effective utilization of historical knowledge. Specifically, we introduce a **Continual Knowledge Adaptation** strategy, which involves maintaining a task-specific **knowledge vector pool** and dynamically using historical knowledge to adapt the agent to new tasks. This process mitigates catastrophic forgetting and enables efficient knowledge transfer across tasks by preserving and adapting critical model parameters.  Additionally, we propose an **Adaptive Knowledge Merging** mechanism that combines similar knowledge vectors to address scalability challenges, reducing memory requirements while ensuring the retention of essential knowledge. Experiments on three benchmarks demonstrate that **CKA-RL** outperforms state-of-the-art methods, achieving an improvement of **4.20%** in overall performance and **8.02%** in forward transfer.

---

## 🚀 Quick Start 

1. **Clone the repo**
    ```bash
    git clone https://github.com/Fhujinwu/CKA-RL.git
    cd CKA-RL
    ```

2. **Create environments**
    >💡 Install the environment only for the experiments you plan to run.  
    >The project provides two separate requirement files for different experiment groups:  
    > - `experiments/atari/requirements.txt` -- dependencies for Atari experiments
    > - `experiments/meta-world/requirements.txt` –- dependencies for Meta-World experiments
    ```bash
    # === Environment setup for 🕹️ Atari experiments ===
    conda create -n cka-rl-atari python=3.10 -y
    conda activate cka-rl-atari
    pip install -r experiments/atari/requirements.txt

    # === Environment setup for 🌍 Meta-World experiments ===
    conda create -n cka-rl-meta python=3.10 -y
    conda activate cka-rl-meta
    pip install -r experiments/meta-world/requirements.txt
    ```

## ▶️ Usage

### 🕹️ Atari Experiments
1. **Run CKA-RL on a specific Atari environment**

    ```bash
    # Example: Run CKA-RL on the Atari environment "Freeway"
    ATARI_ENV="ALE/Freeway-v5"     # Options: ALE/Freeway-v5, ALE/SpaceInvaders-v5
    SEED=42

    cd experiments/atari/
    python run_experiments.py \
        --method_type CKA-RL \
        --env $ATARI_ENV \
        --first-mode 0 \
        --last-mode 7 \             # Use --last-mode 9 for SpaceInvaders
        --seed $SEED \
        --tag main
    ```
    > 🗂️ Training logs and results will be automatically saved to: 
    > `./experiments/atari/data/Freeway/main`

2. **Process Results**

    > ⚠️ Note: Before running the following scripts, ensure that all methods (including baselines) have been executed; otherwise, `process_results.py` may fail due to missing result files.
    ```bash
    # Step 1. Gather all results from experiment logs
    python gather_rt_results.py \
        --base_dir ./experiments/atari/data/Freeway/main \
        --env Freeway             # Options: Freeway, SpaceInvaders

    # Step 2. Process the aggregated results
    python process_results.py \
        --data-dir ./experiments/atari/data/Freeway/main
    ```

### 🌍 Meta-World Experiments
1. **Run CKA-RL on Meta-World environment**

    ```bash
    # Example: Run CKA-RL
    python run_experiments.py \
            --algorithm cka-rl \
            --tag main

    ```
    > 🗂️ Training logs and results will be automatically saved to: 
    > `./experiments/meta-world/runs/main`

2. **Process Results**

    > ⚠️ Note: Before running the following scripts, ensure that all methods (including baselines) have been executed; otherwise, `process_results.py` may fail due to missing result files.
    ```bash
    # Step 1. Gather all results from experiment logs
    python extract_results.py

    # Step 2. Process the aggregated results
    python process_results.py
    ```

## 💬 Citation
If you find our work useful, please consider citing:

```bibtex
@inproceedings{hu2025CKARL,
  title={Continual Knowledge Adaptation for Reinforcement Learning},
  author={Hu, Jinwu and Lian, Zihao and Wen, Zhiquan and Li, Chenghao and Chen, Guohao and Wen, Xutao and Xiao, Bin and Tan, Mingkui},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 🙏 Acknowledgements
We gratefully acknowledge the following open-source contributions:  

- [componet](https://github.com/mikelma/componet) for baseline codebase
- [loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity) for CbpNet implementation
- [mask-lrl](https://github.com/dlpbc/mask-lrl) for MaskNet implementation
- [crelu-pytorch](https://github.com/timoklein/crelu-pytorch) for CReLU implementation

Our work builds upon these open-source efforts; we thank the authors for their valuable contributions.

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=Fhujinwu/CKA-RL&type=Date)