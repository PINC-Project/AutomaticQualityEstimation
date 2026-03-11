import httpx
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PINC")


logger.info("Loading corpus online...")
r = httpx.get("https://pinc-project.gitlab.io/pinc-browser/corpus_1t1.json")
data = r.json()


rows = []
logger.info("Processing corpus...")
for uid, doc in data.items():
    for pn, p in enumerate(doc["source"]["paragraph"]):
        src = p["words"]
        if len(p["links"]) > 0:
            if len(p["links"]) > 1:
                logger.warning(f"Multiple links per paragraph ({uid}:{pn})")
            trn = doc["translation"]["paragraph"][p["links"][0]]["words"]
            rows.append(
                {
                    "UID": uid,
                    "par_num": pn,
                    "source": src,
                    "translation": trn,
                }
            )

logger.info("Saving dataframe..")
df = pd.DataFrame(rows)
df.to_parquet("corpus.parquet")

logger.info("Done!")
