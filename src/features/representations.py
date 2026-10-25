"""The three text representations under comparison, behind one interface.

Each representation is a class with `fit(texts)` and `transform(texts)`. They
are fitted **inside each CV fold on that fold's training half only** - fitting a
vectorizer on all the data before splitting leaks test information into the
features and inflates every score.

    TfidfWord      word 1-2 grams. The reference baseline.
    TfidfChar      character 3-5 grams inside word boundaries.
    FastTextDoc    subword embeddings trained on the unlabeled corpus,
                   averaged over the document.
    TransformerDoc frozen XLM-R, mean-pooled over tokens.

H3 is the reason TfidfChar is here as well as TfidfWord: if fastText holds up
better on mixed text, the honest question is whether that is fastText or simply
character n-grams. Splitting the baseline answers it.

Heavy dependencies (gensim, torch, transformers) are imported lazily, so the
TF-IDF baselines run on a machine where the transformer stack is not installed.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "interim" / "corpus_clean.jsonl"
MANIFEST = ROOT / "data" / "interim" / "batches" / "manifest.csv"
CACHE = ROOT / "models_cache"
XLMR_MODEL = "FacebookAI/xlm-roberta-base"
XLMR_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"

TOKEN_RE = re.compile(r"[Ѐ-ӿa-zA-Z]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class TfidfWord:
    name = "tfidf_word"

    def __init__(self, max_features=50000):
        self.vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, max_features=max_features,
            lowercase=True, token_pattern=r"[Ѐ-ӿa-zA-Z]+")

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def transform(self, texts):
        return self.vec.transform(texts)


class TfidfChar:
    name = "tfidf_char"

    def __init__(self, max_features=100000):
        # char_wb keeps n-grams inside word boundaries, which is what makes this
        # a morphology probe rather than a phrase detector.
        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=max_features, lowercase=True)

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def transform(self, texts):
        return self.vec.transform(texts)


class FastTextDoc:
    """Subword embeddings trained on the *unlabeled* corpus, then frozen.

    All sampled comments are excluded from embedding training. When
    training_videos is set, additional comments are restricted to those videos.
    """

    name = "fasttext"

    def __init__(self, dim=100, epochs=10, min_count=2, corpus=CLEAN, training_videos=None):
        self.dim, self.epochs, self.min_count, self.corpus = (
            dim, epochs, min_count, Path(corpus))
        self.model = None
        self.training_videos = training_videos

    def _corpus_sentences(self):
        if not self.corpus.exists():
            raise SystemExit(
                "{} not found - run build_corpus first.".format(self.corpus))
        if not MANIFEST.exists():
            raise SystemExit(f"{MANIFEST} not found - sampled IDs are needed to avoid leakage")
        import csv
        with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
            sampled = {r["comment_id"] for r in csv.DictReader(f)}
        with open(self.corpus, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row["comment_id"] in sampled:
                        continue
                    if self.training_videos is not None and row["video_id"] not in self.training_videos:
                        continue
                    toks = tokenize(row["text"])
                    if toks:
                        yield toks

    def fit(self, texts=None):
        """`texts` is ignored on purpose - see the class docstring."""
        if self.model is not None:
            return self
        try:
            from gensim.models import FastText
        except ImportError:
            raise SystemExit(
                "gensim is not installed.\n  pip install gensim\n"
                "(the official fasttext wheel has no cp313 build; gensim's "
                "implementation is the same algorithm)")
        CACHE.mkdir(exist_ok=True)
        fingerprint = hashlib.sha256()
        for path in (self.corpus, MANIFEST):
            if not path.exists():
                raise SystemExit(f"{path} not found")
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    fingerprint.update(chunk)
        fingerprint.update(f"{self.dim}:{self.epochs}:{self.min_count}:2026".encode())
        cached = CACHE / "fasttext_{}.model".format(fingerprint.hexdigest()[:16])
        if self.training_videos is None and cached.exists():
            self.model = FastText.load(str(cached))
            return self
        sents = list(self._corpus_sentences())
        if not sents:
            raise ValueError("No unsampled comments on the training videos for fastText")
        self.model = FastText(
            vector_size=self.dim, window=5, min_count=self.min_count,
            sg=1, epochs=self.epochs, workers=1, seed=2026)
        self.model.build_vocab(sents)
        self.model.train(sents, total_examples=len(sents), epochs=self.epochs)
        # Training-video sets change every fold. Keep those models in memory
        # only, so the whole-corpus cache remains specific to its own protocol.
        if self.training_videos is None:
            self.model.save(str(cached))
        return self

    def transform(self, texts):
        wv = self.model.wv
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            if not toks:
                continue
            # fastText answers for OOV words too, via character n-grams - that
            # is exactly the property H3 is about, so no token is skipped.
            out[i] = np.mean([wv[w] for w in toks], axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class TransformerDoc:
    """Frozen multilingual encoder, mean-pooled over non-padding tokens.

    Frozen encoder + linear probe, not fine-tuning. Fine-tuning a 270M-parameter
    model on ~1,000 examples would mostly measure regularisation luck, and it
    would confound the thing being compared: the question is which *representation*
    suits code-switched text, not which one adapts fastest.
    """

    name = "xlmr"

    def __init__(self, model_name=XLMR_MODEL, revision=XLMR_REVISION,
                 batch_size=8, max_len=256, device=None):
        self.model_name, self.revision = model_name, revision
        self.batch_size, self.max_len, self.device = batch_size, max_len, device
        self.tok = self.model = None

    def fit(self, texts=None):
        if self.model is not None:
            return self
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise SystemExit(
                "transformers/torch are not installed.\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install transformers")
        cache_dir = CACHE / "hf"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.tok = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision, cache_dir=str(cache_dir))
        self.model = AutoModel.from_pretrained(
            self.model_name, revision=self.revision,
            cache_dir=str(cache_dir)).eval()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._torch = torch
        return self

    def transform(self, texts):
        if self.model is None:
            raise ValueError("Call fit() before transform()")
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        torch = self._torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i:i + self.batch_size])
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
                hid = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                # Mean over real tokens only; padding would drag vectors toward
                # the pad embedding and penalise short comments.
                pooled = (hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(pooled.cpu().numpy())
        arr = np.vstack(out).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)


REPRESENTATIONS = {
    "tfidf_word": TfidfWord,
    "tfidf_char": TfidfChar,
    "fasttext": FastTextDoc,
    "xlmr": TransformerDoc,
}
