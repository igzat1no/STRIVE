# STRIVE: Structured Representation Integrating VLM Reasoning for Efficient Object Navigation

---

Vision-Language Models (VLMs) have been increasingly integrated into object navigation tasks for their rich prior knowledge and strong reasoning abilities.  However, applying VLMs to navigation presents two key challenges: effectively parsing and structuring complex environment information and determining *when and how* to query VLMs. Insufficient environment understanding and over-reliance on VLMs (e.g. querying at every step) can easily lead to unnecessary backtracking and reduced navigation efficiency, especially in large continuous environments.

To address these challenges, we propose a novel framework that incrementally constructs a multi-layer environment representation consisting of viewpoints, object nodes, and room nodes during navigation. Viewpoints and object nodes facilitate intra-room exploration and accurate target localization, while room nodes support efficient inter-room planning.

Building on this structured representation, we propose a novel two-stage navigation policy, integrating high-level planning guided by VLM reasoning with low-level VLM-assisted exploration to efficiently and reliably locate a goal object.

We evaluated our approach on four simulated benchmarks (HM3D v1\&v2, RoboTHOR, and MP3D), and achieved state-of-the-art performance on both the success rate (SR\% $\mathord{\uparrow}\, 13.1\%$) and navigation efficiency (SPL%$\mathord{\uparrow}\, 6.2\%$). We further validate our method on a real robot platform, demonstrating strong robustness across 120 episodes in 10 different indoor environments.

Please refer to more details in our paper: [STRIVE: Structured Representation Integrating VLM Reasoning for Efficient Object Navigation](https://arxiv.org/abs/2505.06729)

### 🔥 News:

- 2026.1.31: Our paper is accepted by ICRA 2026.

---

## 1. Installation

### 1.1 Clone this repository

```bash
git clone git@github.com:igzat1no/STRIVE.git
cd STRIVE
```

### 1.2 Create a conda environment (Python 3.12)

```bash
conda create -n strive python=3.12 -y
conda activate strive
```

### 1.3 Install pip dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Install habitat-sim and habitat-lab from source

We slightly modified the code of habitat and fixed some bugs. Follow the instructions in [habitat-sim](https://github.com/zwandering/habitat-sim) and [habitat-lab](https://github.com/zwandering/habitat-lab) to install them. Checkout to v0.3.2 branch.

### 1.5 Install Segment Anything Model (SAM)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download the SAM ViT-H checkpoint:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 1.6 Set up MMDetection (GroundingDINO)

```bash
mim install mmengine
mim install mmcv==2.1.0
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

Download the GroundingDINO Swin-L checkpoint:

```bash
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

---

## 3. Required Environment Variables

Do **not** commit real secrets to the repository.
Set them in your shell config (for example, `~/.zshrc`).

```bash
export GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>"
export HABITAT_LAB_PATH="/path/to/habitat-lab/"
export SAM_CHECKPOINT="/path/to/sam_vit_h_4b8939.pth"
export GROUNDING_DINO_PATH="/path/to/mmdetection/"
export GROUNDING_DINO_CHECKPOINT="/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
export HM3D_DATA_PATH="/path/to/HM3D_v2/"
export MP3D_DATA_PATH="/path/to/MP3D/"
```

Then reload your shell:

```bash
source ~/.zshrc
```

Quick check:

```bash
python -c "import os; keys=['GEMINI_API_KEY','HABITAT_LAB_PATH','SAM_CHECKPOINT','GROUNDING_DINO_PATH','GROUNDING_DINO_CHECKPOINT','HM3D_DATA_PATH','MP3D_DATA_PATH']; print({k: bool(os.getenv(k)) for k in keys})"
```

---

## 4. Run

Run HM3D evaluation (default):

```bash
python objnav_benchmark_with_process_obs.py
```

---

## BibTex
Please cite our paper if you find it helpful :)
```
@misc{zhu2025strivestructuredrepresentationintegrating,
      title={STRIVE: Structured Representation Integrating VLM Reasoning for Efficient Object Navigation},
      author={Haokun Zhu and Zongtai Li and Zhixuan Liu and Wenshan Wang and Ji Zhang and Jonathan Francis and Jean Oh},
      year={2025},
      eprint={2505.06729},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.06729},
}
```
