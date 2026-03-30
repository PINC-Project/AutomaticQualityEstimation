import httpx
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PINC")


def fix_capitalization(text: str):
    sstart = True
    for c in text:
        if sstart and c != " ":
            yield c.upper()
            sstart = False
        else:
            if c in set((".", "?", "!")):
                sstart = True
            yield (c)


def remove_disfluency(t: str) -> str:
    ret = " ".join(filter(lambda w: "~" not in w and "+" not in w, t.split()))
    return "".join(fix_capitalization(ret))


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
                    "source": remove_disfluency(src),
                    "translation": remove_disfluency(trn),
                }
            )

logger.info("Saving dataframe..")
df = pd.DataFrame(rows)
df.to_parquet("exp/corpus.parquet")

logger.info("Done!")
