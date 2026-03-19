# STRIVE

This repository contains the STRIVE project.
The current main entry point is:

- `objnav_benchmark_with_process_obs.py`

---

## 1. Dependencies

Recommended Python version: **3.10+**.

Core dependencies include:

- `habitat-lab`
- `habitat-sim`
- `open3d`
- `openai`
- other packages listed in project requirements

Please make sure `habitat-lab` and `habitat-sim` can be imported in your current Python environment.

---

## 2. Installation

```bash
git clone <your_repo_url>
cd STRIVE
pip install -r requirements.txt
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
python objnav_benchmark_with_process_obs.py --save_dir demo_run
```

Common example with arguments:

```bash
python objnav_benchmark_with_process_obs.py \
  --eval_episodes 100 \
  --start_episode 0 \
  --save_dir exp_hm3d \
  --vlm gemini
```

---

## 5. Output

By default, outputs are written to:

- `logs/<save_dir>/episode-*/`
- `logs/<save_dir>/metrics.csv`

These include per-episode logs, trajectories, graph artifacts, and intermediate visualizations.

---

## 6. Troubleshooting

- **`HABITAT_LAB_PATH is not set`**
  - The environment variable is missing or your shell has not been reloaded.

- **Cannot resolve `habitat` / `mmdet` in IDE**
  - Usually an interpreter or analysis-path issue. First verify runtime import works.

- **`GEMINI_API_KEY is not set`**
  - Check whether `GEMINI_API_KEY` is exported correctly.

---
