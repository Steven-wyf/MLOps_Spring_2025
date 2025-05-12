# LLaRA++ — A Cold-Start Music Recommendation System

##  Value Proposition (Unit 1)

**Customer**: Music streaming platforms (e.g., Spotify) targeting new users with sparse interaction histories.

**Value**: LLaRA++ helps recommend personalized songs to new users by combining user-item interaction graphs and song metadata (title, artist, lyrics). This enables better discovery and retention from day one.

**Scale**:

* **Dataset**: 1M playlists (≈20GB), >2M unique tracks.
* **Model size**: Final ONNX QInt8 LLaRA model ≈500MB.
* **Deployment**: Up to 1K inference requests/hour in demo.

##  Cloud-Native Infrastructure (Unit 2/3 - Steven Wang)

* Provisioned using Terraform + KVM in [`tf/`](./tf/).
* MLflow & MinIO deployed via Helm + ArgoCD in [`k8s/platform`](./k8s/platform).
* Services staged in `staging`, `canary`, `production` under [`k8s/`](./k8s).
* CI/CD orchestration in [`ansible/`](./ansible).

##  Persistent Storage + Data (Unit 8 - Data Person: Yufei Wang)

### **Storage**: `/mnt/object/`, mounted to the container on CHI@TACC, named mlops_project9_persistant

  * [`data_processing/`](https://github.com/Steven-wyf/MLOps_Spring_2025/tree/main/data_processing): raw and preprocessed data scripts, dataset overview and inspection script

  * `experiment_tracking/`: MLflow + artifacts

* **Training Data**: [`data_processing/`](./data_processing)

  * MPD: million\_playlist\_dataset.json
  * Example sample:

    ```json
    {
      "name": "musical",
      "tracks": [
        {"track_name": "Finalement", "artist_name": "Degiheugi"}, ...
      ]
    }
    ```

* **Pipeline**: Extract → Clean → Embed lyrics/title → Train

* **ETL + Embedding Scripts**: [`data_processing/`](./data_processing), [`train/bert_encoding.py`](./train/bert_encoding.py)

##  Model Training (Unit 4/5 - Muyuan Zhang)

* Inputs: Playlist (tracks), lyrics, artist names
* Outputs: Recommended next-track logits
* Models:

  * [`train/llara_train.py`](./train/llara_train.py) for full training
  * Matrix_factorazation: `matrix_factorization.py`
  * MLP Projector: `mlp_train.py`
* Re-training done via `docker-compose-training.yml`
* Experiment tracking: [`mlflow`](http://129.114.25.37:5000)

##  Model Serving (Unit 6/7 - Siqi Xu)

* Quantized to ONNX QInt8 (see `evaluation/templates`)
* API in [`inference/inference_api.py`](./inference/inference_api.py)
* Served with FastAPI + Docker Compose: [`docker-compose-inference.yml`](./inference/docker-compose-inference.yml)
* Input: `{ "tracks": [ {"track_name": ..., "artist_name": ... }, ... ] }`
* Output: `{ "recommended_tracks": [...] }`
* Offline evaluation: [`evaluation`](./evaluation) folder
* Load test results: logged via MLflow

##  Evaluation and Monitoring (Siqi Xu)

* Offline metrics: Recall\@xx, MRR\@xx, Precision\@xx  (🎯)
* Online logging: API response latency, failure rates (FastAPI middleware)
* Business-specific evaluation: genre diversity vs. retention rate (🎯)

##  CI/CD and Continuous Training (Unit 2/3 - Steven Wang)

* Infrastructure as Code: [`tf/`](./tf), [`k8s/`](./k8s), [`ansible/`](./ansible)
* GitHub Actions trigger training/deployment (see CI script)
* Model promotion:

  * Train → Staging → Canary → Production (ArgoCD apps)
  * Use MLflow metrics to auto-promote

##  Online Inference + Feedback Loop (🎯) Data or Evaluation person? 

* `/predict` endpoint accepts user playlist input
* Feedback (clicks, skips) logged for monitoring
* Future loop: feedback → retraining pipeline

---

##  Folder Map

| Folder                 | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `data_processing/`     | Preprocessing scripts, data embedding              |
| `train/`               | All training code: LightGCN, BERT, MLP, LLaRA      |
| `inference/`           | FastAPI server, inference logic                    |
| `evaluation/`          | Offline tests, metrics computation                 |
| `experiment_tracking/` | MLflow + MinIO configuration                       |
| `ansible/`             | Automation playbooks for platform & app deployment |
| `k8s/`                 | Helm values and ArgoCD templates                   |
| `tf/`                  | Terraform configuration for VMs, block storage     |

---

##  Run the System on Chameleon

```bash
# Mount volumes (rclone/block)
# Provision infra:
cd tf/kvm
terraform init && terraform apply

# Run platform deployment:
cd ../ansible
ansible-playbook -i inventory.yml argocd/argocd_add_platform.yml

# Start MLflow + MinIO:
kubectl port-forward svc/mlflow 5000:5000 -n mlops-platform &
kubectl port-forward svc/minio 9000:9000 -n mlops-platform &

# Run training:
cd ../train
docker-compose -f docker-compose-training.yml up

# Build + push model:
cd ../workflows
argo submit --from workflowtemplate/train-model -p model-version=0

# Serve API:
cd ../inference
docker-compose -f docker-compose-inference.yml up
```

---
