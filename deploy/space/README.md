---
title: Clinical Diagnostic API
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Chest X-ray triage API (Normal, Pneumonia, Tuberculosis)
---

# Clinical Diagnostic API

The FastAPI backend of the [Clinical Diagnostic Portal](https://github.com/adi13apr/Clinical-Diagnostics-). It classifies a chest X-ray as Normal, Pneumonia or Tuberculosis with a distilled EfficientNet-B0, returns temperature-calibrated probabilities, flags uncertain results for clinician review, and serves Grad-CAM maps and an LLM-written report.

> **Not a medical device.** This is a research/portfolio project. Its output is a screening aid, not a diagnosis.

- **Interactive docs:** `/docs`
- **Health check (no login):** `/health` should report `model_loaded: true` and `calibrated: true`
- Every other endpoint needs `Authorization: Bearer <Supabase access token>`.

**This Space sleeps when idle.** On the free tier it stops after a period without traffic, so the first request after a pause takes a minute or two while the container starts and loads the model.

This repository is a generated snapshot. Don't edit it here: change the GitHub repo and redeploy with `scripts/deploy_space.ps1`.
