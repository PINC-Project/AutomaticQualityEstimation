# AutomaticQualityEstimation

Various scripts for assessing translation quality with no/minimum manual effort.

## Plan

1. automatic transcription of audio files
2. sentence splitting, eg:
   - NLTK – https://www.nltk.org/api/nltk.tokenize.sent_tokenize.html
   - SpaCy - https://spacy.io/api/sentencizer/
   - Silero Text Enchancement - https://habr.com/ru/articles/581960/ 
3. sentence alignment, eg:
  - https://arxiv.org/pdf/2311.08982 – https://github.com/steinst/sentalign/
  - https://github.com/bfsujason/bertalign
4. compute automatic quality estimation metrics
  - https://github.com/Unbabel/COMET
  - https://huggingface.co/facebook/blaser-2.0-qe
  - https://tharindu.co.uk/TransQuest/models/sentence_level_pretrained/
  - compute AQE metric with ChatGPT https://aclanthology.org/2024.eamt-1.28.pdf
5. export to CSV
6. human quality assesment UI
  - content faithfulness, language naturalness, delivery
