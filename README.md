# Automatic Quality Estimation

## Things done so far

- imported PINC corpus into pandas
    - stored result in parquet file `corpus.parquet`
    - used only 1-to-1 mappiung of paragraphs
    - stored recording name and paragraph number so we can retreive meta-data later, if needed
- computed Comet score on corpus
    - stored in `corpus+comet.parquet`
    - removed fillers and hesitations before scoring
    - score is on each paragraph pair individually
    - translation is not perfectly aligned to source, so score might be better for complete text than each paragraph individually
- did some preliminary machine translation experiments
    - used `sdadas/mt5-base-translator-en-pl` (and `...pl-en`) from Huggingface
    - used `Helsinki-NLP/opus-mt-pl-en` as the basic MarianMT model
    - there was no Helsinki-NLP version in the other direction
    - used `pumad/pumadic-en-pl` for the other direction, instead
    - also used `facebook/nllb-200-3.3B` which is a large SOTA NMT model
    - computed Comet scores on them as well - for comparison
    - machine translation was done without context, so comparison to human translation is not accurate
    - ideally we would translate full recordings and then do alignment
    - I'm worried there may be performance issues for longer texts (need to check this)
- each parquet file converted to XLSX for inspection
    - look in the `xlsx` folder - click on a file and look for the download button
