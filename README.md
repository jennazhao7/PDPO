# PDPO

Privacy-preserving DPO research codebase. See [COLLABORATOR.md](COLLABORATOR.md) for setup.

## Google Cloud GPUs

Run on GCP with a GPU VM + SSH: **[docs/GCP.md](docs/GCP.md)**

```bash
cp scripts/gcp/gcp.env.example scripts/gcp/gcp.env   # set your project ID
gcloud auth login
bash scripts/gcp/setup_gcp.sh
```

