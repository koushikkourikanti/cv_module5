# Video-LLM Classroom Assistant

A multimodal video-understanding prototype developed for a Computer Vision Challenge Assignment inspired by the **CVPR 2025 Video-LLM Workshop**.

The project demonstrates how sampled video frames can be combined with a multimodal Large Language Model to perform:

- Video summarization
- Object recognition
- Action recognition
- Temporal reasoning
- Before/after reasoning
- Natural-language video question answering

---

## Project Overview

Traditional video-analysis systems often focus on individual frames or predefined visual categories.

This project explores a more interactive approach using a **Video-LLM-style pipeline**, where frames sampled from a video are analyzed collectively by a multimodal language model.

The prototype allows a user to:

1. Upload a short video.
2. Extract representative frames using OpenCV.
3. Send the frames to a multimodal LLM.
4. Generate a summary of the video.
5. Identify important objects and actions.
6. Infer the chronological sequence of events.
7. Ask natural-language questions about the video.
8. Run a five-question feasibility evaluation.

---

## Challenge Assignment

The prototype was developed for a challenge assignment based on the:

**CVPR 2025 Video-LLM Workshop**

### Keynote Inspiration

**Advancing Video Understanding: From Training-Free to Streaming Video LLM**

The project is inspired by ideas related to:

- Video-LLMs
- Multimodal reasoning
- Temporal video understanding
- Streaming video intelligence
- Visual question answering
- Long-form video understanding

This project is a feasibility prototype and does **not** claim to reproduce the complete StreamBridge or SF-LLaVA architectures.

---

## Problem Statement

Students and users watching instructional or activity videos may need to understand important actions, objects, and event sequences without repeatedly reviewing the entire video.

The problem explored in this project is:

> Can a multimodal LLM analyze representative frames from a video, understand the major visual events and their temporal order, and answer natural-language questions about the video?

---

## Proposed Solution

The system uses a simple Video-LLM-inspired processing pipeline:

```text
Input Video
     |
     v
Video Metadata Extraction
     |
     v
Uniform Frame Sampling
     |
     v
8 Chronological Frames
     |
     v
Multimodal LLM
     |
     v
Visual + Temporal Reasoning
     |
     +----------------------+
     |                      |
     v                      v
Video Summary          Video Question Answering
     |
     v
Feasibility Evaluation