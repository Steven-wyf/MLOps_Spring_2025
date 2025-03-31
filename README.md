# MLOps_Spring_2025
# Project Proposal: LLaRA++ — A cold-start music recommendation system that integrates behavior and text

## Title of project
LLaRA++: Beyond Language Modeling — Hybrid Embedding and User-Aware Curriculum for Cold-Start Music Recommendation

## Contributors
| Name             | Responsible for                                      | Link to their commits in this repo             |
|------------------|-------------------------------------------------------|------------------------------------------------|
| All team members | Overall system design, integration, and documentation |       |
| Member A         | Model training (LightGCN, Projector, LLaRA)           |  |
| Member B         | Model quantization, FastAPI deployment & monitoring   |  |
| Member C         | Data preprocessing, lyric/title embedding, cold start |  |
| Member D         | CI/CD, pipeline integration, dashboard                |  |

## System diagram
(See diagram in the repository `main/system_diagram.png` — this includes the MPD input → LightGCN & BERT/DistilBERT → SR2LLM → LLaRA + curriculum prompt tuning → quantized ONNX model → API server → evaluation dashboard.)

## Summary of outside materials
| Dataset / Model                         | How it was created                                            | Conditions of use |
|----------------------------------------|----------------------------------------------------------------|-------------------|
| Million Playlist Dataset (MPD)         | Collected from US Spotify users 2010–2017, 1M playlists        | Research only, under AIcrowd challenge license 【44†license.txt】|
| DistilBERT / BERT                      | Pretrained on Wikipedia + BookCorpus                         | Open under Apache 2.0       |
| LightGCN Enhanced (from paper)         | Constructed from playlist-track-user graph                   | Public research use         |
| LLaRA code from SIGIR ’24 paper        | Hybrid prompt & curriculum tuning idea                       | MIT License on GitHub       |

## Summary of infrastructure requirements
| Requirement     | How many/when                                     | Justification                         |
|-----------------|---------------------------------------------------|--------------------------------------|
| `m1.medium` VMs | 3 throughout the project                          | LightGCN & data preprocessing        |
| `gpu_mi100`     | 4-hour block 2x per week                          | Training LLaRA model + tuning        |
| Floating IPs    | 1 always-on + 1 temporary                         | API serving + dashboard staging      |
| Persistent Disk | 100GB for data, embeddings, and logs             | Reuse across stages, tracked volume  |

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
