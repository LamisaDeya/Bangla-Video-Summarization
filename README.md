<div align="center"> 

## Abstractive Summarization of Bengali Academic Videos Based on Audio Subtitles

</div>

<div align="center"> 

<a href="https://arxiv.org/abs/YOUR_ARXIV_ID">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red" />
</a>
<a href="https://github.com/LamisaDeya/Bangla-Video-Summarization">
    <img src="https://img.shields.io/badge/GitHub-Repository-blue" />
</a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" />
</a>

</div>

## Introduction

The rapid growth of academic video content makes it difficult for students and educators to find relevant information efficiently. This is especially challenging for low-resource languages like Bengali due to the lack of video summarization tools. This work presents the **first end-to-end pipeline** for abstractive summarization of Bengali academic videos based on audio subtitles.

Our system integrates audio preprocessing, speech-to-text transcription, context-aware smart chunking, abstractive summarization, and title generation in a unified pipeline. The summaries include timestamps for easy video navigation, making educational content more accessible.

**Key Features:**
- End-to-end pipeline from video to timestamped summary
- Audio preprocessing to improve transcription quality
- Smart chunking with sentence-level overlap
- Fine-tuned BanglaT5 for summarization
- Fine-tuned mT5-multilingual-XLSum for title generation
- Two benchmark datasets for Bengali academic content


## Pipeline Overview

The pipeline consists of the following stages:
1. **Audio Extraction & Preprocessing**: Extract audio and apply adaptive amplification
2. **Speech-to-Text**: Google's Universal Speech Model (USM) for transcription
3. **Text Chunking**: Smart chunking with 1-sentence overlap (512 tokens max)
4. **Summarization**: Fine-tuned BanglaT5 model
5. **Title Generation**: Fine-tuned mT5-multilingual-XLSum model
6. **Post-processing**: Timestamp integration and regex-based formula conversion

## Environment

Create and activate the environment:
```bash
conda create -n bengali-video-sum python=3.8
conda activate bengali-video-sum
pip install torch transformers pandas openpyxl
pip install moviepy pydub google-cloud-speech
pip install banglanlptoolkit
```

## Datasets

We provide two benchmark datasets:

| Dataset | Pairs | Videos | Topics | Subjects |
|---------|-------|--------|--------|----------|
| **Summarization** | 10,029 | 213 | 46 | 6 |
| **Title Generation** | 1,005 | 335 | 54 | 6 |

**Dataset Statistics:**
- **Summarization Dataset**: Text chunks (401-422 words) paired with summaries (150-199 words)
- **Title Generation Dataset**: Summaries (1,003-1,011 words) paired with titles (7-19 words)
- **Subject Distribution (Summarization Dataset)**: Physics (29%), Chemistry (20%), Biology (14%), Math & ICT (14%), CSE (13%), Other (13%)

**Download:**
- Datasets are available in `Summarization_dataset.xlsx` and `titles_dataset.xlsx`
- Dataset statistics in `stat_of_dataset.xlsx`

## Model Zoo

### Summarization Models
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | MoverScore |
|-------|---------|---------|---------|-----------|------------|
| **BanglaT5** | **0.4018** | **0.1645** | **0.2579** | **0.8795** | **0.6036** |
| NLLB-200 | 0.3629 | 0.1187 | 0.2249 | 0.8734 | 0.5899 |
| mBART-50 | 0.3438 | 0.1044 | 0.2016 | 0.8686 | 0.5812 |
| mT5 | 0.3400 | 0.1139 | 0.2015 | 0.8694 | 0.5830 |

### Title Generation Models
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| **mT5-XLSum** | **0.4476** | **0.2129** | **0.3720** |
| BanglaT5 | 0.1033 | 0.0136 | 0.0942 |


## Training

### Summarization
Training code is provided in the following files:
- `BanglaT5_training.txt` - BanglaT5 training script
- `mBART_training.txt` - mBART training script
- `NLLB_code.txt` - NLLB training script
- `mt5_training.ipynb` - mT5 training notebook

**Training Configuration:**
- Model: BanglaT5 (247M parameters)
- Epochs: 10
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Batch Size: 4 (train and eval)
- Hardware: Single NVIDIA T4 GPU

**To train:**
```python
# Load your dataset
train_data = pd.read_excel('Summarization_dataset.xlsx', sheet_name='train')

# Fine-tune BanglaT5
# See BanglaT5_training.txt for complete code
```

### Title Generation
```python
# Load summary-title pairs
title_data = pd.read_excel('titles_dataset.xlsx', sheet_name='train')

# Fine-tune mT5-multilingual-XLSum
# See mt5_training.ipynb for complete code
```

## Inference

### Individual Model Inference
Use the provided notebooks for inference:
- `banglat5_inference.ipynb` - BanglaT5 inference
- `NLLB_inference.ipynb` - NLLB inference
- `mBART_inference.ipynb` - mBART inference
- `mt5_inference.ipynb` - mT5 inference

### Full Pipeline
Run the complete end-to-end pipeline using `Full_pipeline.ipynb`:

```python
# 1. Upload your trained models to Google Colab
# 2. Place models in: /content/models/
# 3. Run Full_pipeline.ipynb

# Input: Academic video file (.mp4)
# Output: 
#   - Title
#   - Timestamped summary
#   - Compression ratio
```

**Pipeline Stages:**
1. Audio extraction and preprocessing
2. Speech-to-text transcription (Google USM API)
3. Punctuation addition
4. Smart chunking (1-sentence overlap)
5. Summary generation (BanglaT5)
6. Title generation (mT5-XLSum)
7. Timestamp integration
8. Regex-based post-processing

## Evaluation

Evaluate models using `Performance_metrices.ipynb`:

```python
# Metrics implemented:
# - ROUGE (1, 2, L)
# - BERTScore
# - MoverScore
# - Factual Consistency (FactCC, SummaC, DAE)
# - Bootstrap confidence intervals
```

**Transcription Quality:**
| Model | WER | CER |
|-------|-----|-----|
| Google ASR (preprocessed) | **0.2825** | **0.3378** |
| Google ASR | 0.3672 | 0.3380 |
| Facebook MMS | 0.5537 | 0.9390 |
| Whisper Large v3 | 1.3446 | 1.6920 |

## Cross-Domain Performance

The fine-tuned BanglaT5 model shows strong zero-shot capabilities:

| Content Type | ROUGE-L | BERTScore | MoverScore |
|--------------|---------|-----------|------------|
| **Other Academic Videos** | **0.2095** | **0.8712** | **0.6089** |
| News Content | 0.1743 | 0.8974 | 0.5505 |
| Podcasts | 0.1353 | 0.8814 | 0.5505 |
| Tech-Related Videos | 0.1632 | 0.8837 | 0.5318 |

## Results Highlights

- **23% improvement** in transcription WER with audio preprocessing
- **BanglaT5 outperforms** multilingual models by 10.7% (ROUGE-1)
- **Strong generalization** to other academic content
- **Fastest inference**: 2.54s for BanglaT5 vs 78.32s for Qwen
- **Human evaluation**: Generated summaries achieve Likert scores of 2.23-2.46 (scale 1-3)

## Repository Structure

```
Bangla-Video-Summarization/
├── BanglaT5_training.txt           # BanglaT5 training code
├── banglat5_inference.ipynb        # BanglaT5 inference
├── mBART_training.txt              # mBART training code
├── mBART_inference.ipynb           # mBART inference
├── NLLB_code.txt                   # NLLB training code
├── NLLB_inference.ipynb            # NLLB inference
├── mt5_training.ipynb              # mT5 training
├── mt5_inference.ipynb             # mT5 inference
├── Full_pipeline.ipynb             # Complete end-to-end pipeline
├── Performance_metrices.ipynb      # Evaluation metrics
├── Summarization_dataset.xlsx      # Summarization dataset (10,029 pairs)
├── titles_dataset.xlsx             # Title generation dataset (1,005 pairs)
├── stat_of_dataset.xlsx            # Dataset statistics
└── README.md                       # This file
```

## Citation

If you use this work, please cite:

## Acknowledgements

This codebase builds upon the following resources:
- [BanglaT5](https://github.com/csebuetnlp/BanglaT5) for Bengali text generation
- [mT5](https://github.com/google-research/multilingual-t5) for multilingual modeling
- [Google Speech Recognition API](https://cloud.google.com/speech-to-text) for transcription
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for model implementations


## Contact

For questions or collaboration:
- **Lamisa Bintee Mizan Deya**: deya1907049@stud.kuet.ac.bd
- **Farhatun Shama**: farhatunshama@gmail.com
- **Repository**: [https://github.com/LamisaDeya/Bangla-Video-Summarization](https://github.com/LamisaDeya/Bangla-Video-Summarization)

---

**Note:** This is a research-level repository and may contain issues/bugs. Please contact the authors for any queries.
