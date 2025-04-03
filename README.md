# MLOps_Spring_2025
# Project Proposal: LLaRA++ — A cold-start music recommendation system that integrates behavior and text

## LLaRA++: Beyond Language Modeling — Hybrid Embedding and User-Aware Curriculum for Cold-Start Music Recommendation
LLaRA++ is a cold-start music recommendation system that unifies graph-based collaborative filtering (LightGCN) and text-based embeddings (BERT/DistilBERT) to address minimal interaction data. An MLP projector (SR2LLM) aligns LightGCN embeddings into a language model space, enabling LLaRA with curriculum prompt tuning for enhanced predictions. Deployed via ONNX QInt8 quantization in a FastAPI service and orchestrated by a CI/CD pipeline, LLaRA++ offers scalable, low-latency inference. Its primary users are music streaming platforms and ML teams seeking advanced yet practical solutions to cold-start problems.

## Contributors
| Name             | Responsible for                                      | Link to their commits in this repo             |
|------------------|-------------------------------------------------------|------------------------------------------------|
| All team members | Overall system design, integration, and documentation |       |
| Muyuan Zhang        | Model training (LightGCN, Projector, LLaRA)           |  |
| Siqi Xu       | Model quantization, FastAPI deployment & monitoring   |  |
| Yufei Wang         | Data preprocessing, lyric/title embedding, cold start |  |
| Steven Wang       | CI/CD, pipeline integration, dashboard                |  |

## System diagram
![LLaRA++ Pipeline Diagram](System-diagram.png)

## Summary of outside materials
| Dataset / Model                         | How it was created                                            | Conditions of use |
|----------------------------------------|----------------------------------------------------------------|-------------------|
| [Million Playlist Dataset (MPD)](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)         | Collected from US Spotify users 2010–2017, 1M playlists        | Research only, under AIcrowd challenge license|
| [DistilBERT / BERT](https://github.com/Steven-wyf/MLOps_Spring_2025/blob/main/bert-powered%20book%20genre%20classfication.pdf)                      | Pretrained on Wikipedia + BookCorpus                         | Open under Apache 2.0       |
| [LightGCN Enhanced (from paper)](https://github.com/Steven-wyf/MLOps_Spring_2025/blob/main/Embedding%20Enhancement%20Method%20for%20LightGCN%20in%20Recommendation%20Information%20Systems.pdf)       | Constructed from playlist-track-user graph                   | Public research use         |
| [LLaRA code from SIGIR ’24 paper](https://github.com/Steven-wyf/MLOps_Spring_2025/blob/main/llara.pdf)      | Hybrid prompt & curriculum tuning idea                       | MIT License on GitHub       |

## Summary of Infrastructure Requirements

| Requirement       | How Many / When                                      | Justification                                                                 |
|-------------------|------------------------------------------------------|-------------------------------------------------------------------------------|
| `m1.medium` VMs   | 3 VMs for the entire project duration                | For running lightweight services including data preprocessing, genre tagging, and API server components. |
| `gpu_mi100`       | 4-hour blocks, twice per week (total ~32 hours)     | Required for training LLaRA with LoRA adapters, tuning LightGCN embeddings, and running curriculum-based prompt tuning. |
| Floating IPs      | 1 persistent, 1 temporary during staging deployments | Persistent IP for stable API endpoint access; a second IP for temporary staging/canary testing. |
| Persistent Storage| 100GB volume attached for the full project duration | To store datasets, training artifacts, model checkpoints, ONNX binaries, and streaming logs across pipelines. |
| Ray Cluster       | Dynamically scaled during training & evaluation     | Used for scheduling training jobs and distributed hyperparameter tuning (Ray Tune). |
| MLflow Server     | 1 containerized instance                             | Required for tracking experiments and managing model metadata during development. |


## Detailed design plan

### Model training and training platforms
- Use **LightGCN** to model user-item sequence graph behavior embeddings【43†Embedding Enhancement Method for LightGCN】
- Use **BERT/DistilBERT** to embed textual features (title, artist, lyrics)【39†Exploring Genre with DistilBERT】【43†bert-powered book genre classfication.pdf】
- Train an MLP projector (SR2LLM) to align LightGCN ID embedding into LLM space
- Use **LLaRA** as core model: start with text-only prompting, then curriculum prompt tuning to inject behavior features【43†llara.pdf】
- Use **LoRA** for efficient fine-tuning
- Track experiments with **MLflow**

### Model serving and monitoring platforms
- Quantize final LLaRA model using **ONNX + QInt8**【40†LLM Quantization】
- Serve with **FastAPI** with support for batching
- Log latency and throughput metrics
- Set up 3 deployment stages: staging, canary, production

### Data pipeline
- Use Python scripts to extract playlist → seed tracks → target for next-track prediction
- Lyrics/title embedding and genre classification for cold start
- Streaming simulation for inference testing
- Persistent volume stores track embeddings, models, test logs
- Preprocessing scripts + evaluation pipeline containerized

### Continuous X
- Use **Terraform/Helm** to define infrastructure
- Set up GitHub Actions CI/CD to automate data → train → deploy
- Stage model updates using ArgoCD
- Auto-deploy passing models to canary → production if metrics are met
- Dashboard monitoring + alerting for model drift & service anomalies

---
