# RAVDESS Speech Emotion Recognition — Model Comparison

A speaker-independent deep learning pipeline that classifies emotion
(neutral, calm, happy, sad, angry, fear, disgust, surprise) from speech
audio, comparing five architectures on the same data and evaluation
pipeline: **LSTM, 1D-CNN, CNN-LSTM, FNN, and VGG16 (transfer learning on
log-Mel spectrograms)**.

Dataset: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) — 24 actors, 8 emotions, 1,440 clips.

## Why this is different from a typical SER notebook

Most RAVDESS tutorials split the dataset by *recording*, not by *speaker* —
since each actor has multiple recordings, a random split lets the same
voice appear in both train and test, inflating accuracy because the model
partly learns to recognize the speaker instead of the emotion.

This pipeline splits by **actor ID** instead, with automated checks
confirming no actor appears in more than one partition before training
begins. As a result, the numbers reported here tend to be lower than
typical RAVDESS results that use a random split, since the model can no
longer rely on partially recognizing the speaker's voice to predict the
emotion.

## Results

Speaker-independent test set (300 clips, actors never seen in training;
chance level = 12.5%):

| Model    | Accuracy | F1 (macro) | AUC (macro) |
|----------|---------:|------------:|-------------:|
| VGG16    | 53.7%    | 0.536       | 0.877        |
| FNN      | 42.3%    | 0.412       | 0.804        |
| CNN-LSTM | 30.3%    | 0.291       | 0.734        |
| LSTM     | 23.7%    | 0.208       | 0.694        |
| CNN      | 13.3%    | 0.030       | 0.619        |

**VGG16 wins** because it's the only model operating on real
time-frequency structure (spectrogram images), where 2D convolution is a
well-motivated choice. **FNN beats CNN and LSTM** despite being the
simplest architecture — the engineered feature vector (ZCR + RMS + MFCC
concatenated) isn't a real time series, so treating it as one (via Conv1D
or LSTM) imposes structure that doesn't exist in the data. **The CNN
collapsed entirely** (predicting one class almost every time, best
checkpoint from epoch 1) — left in as an instructive negative result
rather than removed.

## Setup & run

```bash
pip install -r requirements.txt
export RAVDESS_ROOT=/path/to/ravdess-emotional-speech-audio  # optional, defaults to Kaggle path
```

Open `ravdess_speech_emotion_recognition.ipynb` and run all cells. GPU
recommended, not required — falls back to CPU automatically.

## Acknowledgments

Core audio feature extraction (ZCR, RMSE, MFCC) is adapted from Shivam
Burnwal's [Speech Emotion Recognition](https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition)
notebook on Kaggle. The speaker-independent split with leakage checks, the
VGG16 spectrogram branch, GPU/CPU fallback, and the five-model comparison
with macro-averaged metrics are original to this implementation.
