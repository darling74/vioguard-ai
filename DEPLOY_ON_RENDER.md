# VIOGUARD AI — Render Deployment

This deployment setup targets Render's Python web service.

## Deploy
1. Extract this ZIP on your computer.
2. Create a GitHub repository and upload the extracted project contents (including `render.yaml`, `.python-version`, and the `models/model_config.json` file). Do not upload `.env`, secrets, private videos, or model weights.
3. In Render, choose **New + → Blueprint** and connect the GitHub repository containing `render.yaml`.
4. Review the service plan and any charges, then create/deploy the service. The build installs dependencies and runs `python download_model.py` to download and verify the pretrained model.
5. When the deploy finishes, open the public `onrender.com` URL and test `/api/health`, then upload a short supported MP4/AVI/MOV video.

## Important
- The project config supports `HOST` and `PORT`; Render is configured to bind to `0.0.0.0` and use its assigned `PORT`.
- PyTorch + video inference may exceed free-tier RAM/CPU limits. The included blueprint selects a paid standard instance; review current plan pricing before deploying.
- Uploaded videos are temporary and the app deletes them after analysis. Job state is in memory, so this is a single-instance prototype, not a multi-instance production service.
- The model repository README describes the weights as research/non-commercial. Do not use this deployment for commercial use without confirming the model licence.
- Violence predictions can be wrong. Do not use them as the sole basis for decisions about people.
