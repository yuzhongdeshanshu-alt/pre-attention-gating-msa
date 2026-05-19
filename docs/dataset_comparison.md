This document briefly compares several multimodal affective computing datasets considered for this project. CMU-MOSEI was selected as the main benchmark because it is large, widely used, and directly aligned with utterance-level multimodal sentiment regression.

| Dataset | Main focus | Scale / labels | Reason for selection or exclusion |
|---|---|---|---|
| CMU-MOSI | Multimodal sentiment analysis in opinion videos | Smaller utterance-level corpus; sentiment intensity labels | Relevant to multimodal sentiment analysis, but smaller than CMU-MOSEI, making it less suitable for comparing multiple model variants across random seeds. |
| CMU-MOSEI | Large-scale multimodal sentiment and emotion analysis in opinion videos | 23,453 segments; sentiment scores from -3 to +3 | Selected because it is large, widely used, and directly aligned with utterance-level sentiment regression. |
| IEMOCAP | Multimodal emotion recognition in acted dyadic interactions | Around 12 hours; mainly discrete emotion labels | Not selected because its primary focus is emotion recognition rather than opinion-video sentiment intensity regression. |
| MELD | Multimodal emotion recognition in multi-party conversations | About 13,000 utterances; emotion and sentiment labels | Not selected because dialogue context and speaker interaction are central to the dataset, differing from the present utterance-level MSA setting. |
| M3ED | Chinese multimodal emotional dialogue analysis | 24,449 utterances; seven emotion categories | Not selected because it focuses on Chinese emotional dialogue and emotion recognition rather than English MSA sentiment regression. |

## References

- Zadeh et al. (2016). CMU-MOSI: Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos.
- Zadeh et al. (2018). Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph.
- Busso et al. (2008). IEMOCAP: Interactive Emotional Dyadic Motion Capture Database.
- Poria et al. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.
- Zhao et al. (2022). M3ED: Multi-modal Multi-scene Multi-label Emotional Dialogue Database.