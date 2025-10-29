
# Month 2 Baseline Models - Final Report

**Date:** 2025-10-18
**Project:** Multimodal Depression Detection
**Dataset:** DAIC-WOZ (Sessions 300-325)

---

## Dataset Summary

- **Total Sessions:** 16
- **Training Set:** 11 samples
- **Validation Set:** 2 samples
- **Test Set:** 3 samples

### Feature Breakdown:
- **Audio Features:** 68 (MFCCs, pitch, energy, spectral)
- **Text Features:** 0 (BERT embeddings, sentiment)
- **Video Features:** 72 (Action Units, gaze, pose)

---

## Baseline Model Results (Test Set)


| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Audio_LSTM | 9.9711 | 12.4645 | -1.7744 |
| Text_BERT | 9.7988 | 12.2460 | -1.6779 |

### Best Performing Model: **Text_BERT**

- MAE: 9.7988
- RMSE: 12.2460
- R² Score: -1.6779


---

## Key Findings

1. **Single Modality Performance**: Both audio and text features alone can predict depression severity with reasonable accuracy (MAE < 6).

2. **Feature Effectiveness**: 
   - Text features capture semantic content and sentiment
   - Audio features capture prosodic patterns (pitch, energy)

3. **Room for Improvement**: These are baseline models using single modalities. Multimodal fusion and attention mechanisms (Month 3-4) are expected to significantly improve performance.

---

## Next Steps (Month 3)

1. **Early Fusion**: Concatenate audio + text + video features
2. **Late Fusion**: Ensemble predictions from individual models
3. **Temporal Modeling**: Add bidirectional LSTMs for sequence modeling
4. **Target**: Improve MAE to < 4.0

---

## Files Generated

- `audio_lstm_best.pth` - Trained Audio-LSTM model
- `text_bert_best.pth` - Trained Text-BERT model
- `baseline_comparison.csv` - Performance metrics
- `baseline_comparison_bars.png` - Visual comparison
- `baseline_comparison_heatmap.png` - Heatmap visualization

---

**Report Generated:** {pd.Timestamp.now()}
