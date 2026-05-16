# EmoVerse Dataset Card

## Dataset Summary

EmoVerse is a large-scale visual emotion dataset for interpretable visual emotion analysis. It provides categorical emotion states, dimensional emotion representations, natural-language descriptions, Background-Attribute-Subject triplets, and region-level visual grounding.

## Intended Use

EmoVerse is designed for research on:

- visual emotion classification,
- emotion explanation generation,
- interpretable multimodal learning,
- emotion-aware image editing and generation,
- affective data augmentation,
- continuous emotion representation learning.

## Data Fields

Public records are expected to include:

| Field | Description |
| --- | --- |
| `id` | Sample identifier |
| `image_path` | Relative path to the image |
| `description` | Brief visual and affective description |
| `emotion` | CES emotion category |
| `confidence` | Emotion confidence or intensity score |
| `background` | Background or scene context |
| `attribute` | Affective attribute |
| `subject` | Emotion-bearing subject |
| `B-A-S` | Background-Attribute-Subject triplet |
| `DES` | 1024-dimensional Dimensional Emotion Space representation |
| `bbox` | Subject-level bounding boxes |
| `mask_path` | Optional subject mask path |

## Emotion Labels

The categorical label space follows an 8-class visual emotion setting:

- Amusement
- Anger
- Awe
- Contentment
- Disgust
- Excitement
- Fear
- Sadness

## Annotation Pipeline

The annotation pipeline combines multimodal large language models, emotion-specific verification models, Grounding DINO, SAM, Critic Agent review, and manual sampling. The goal is to balance scale, reliability, and interpretability.

## Known Considerations

Visual emotion perception is subjective and culturally sensitive. Labels should be interpreted as dataset annotations rather than universal truth. Models trained on this dataset should be evaluated for robustness, bias, and domain transfer before deployment.

## Distribution

The dataset is not distributed through Git. Public download instructions will be announced after license and storage review.
