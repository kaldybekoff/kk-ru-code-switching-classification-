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

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "interim" / "corpus_clean.jsonl"
CACHE = ROOT / "models_cache"

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

    Training on all 43k unlabeled comments rather than on the ~1k labeled ones
    is deliberate and is not leakage: no label is involved, and the same frozen
    model is applied to every fold. It is what gives the subword vectors enough
    text to be worth anything.
    """

    name = "fasttext"

    def __init__(self, dim=100, epochs=10, min_count=2, corpus=CLEAN):
        self.dim, self.epochs, self.min_count, self.corpus = (
            dim, epochs, min_count, Path(corpus))
        self.model = None

    def _corpus_sentences(self):
        if not self.corpus.exists():
            raise SystemExit(
                "{} not found - run build_corpus first.".format(self.corpus))
        with open(self.corpus, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    toks = tokenize(json.loads(line)["text"])
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
        cached = CACHE / "fasttext_d{}_e{}.model".format(self.dim, self.epochs)
        if cached.exists():
            self.model = FastText.load(str(cached))
            return self
        sents = list(self._corpus_sentences())
        self.model = FastText(
            vector_size=self.dim, window=5, min_count=self.min_count,
            sg=1, epochs=self.epochs, workers=4, seed=2026)
        self.model.build_vocab(sents)
        self.model.train(sents, total_examples=len(sents), epochs=self.epochs)
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

    def __init__(self, model_name="xlm-roberta-base", batch_size=16, max_len=128):
        self.model_name, self.batch_size, self.max_len = (
            model_name, batch_size, max_len)
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
        CACHE.mkdir(exist_ok=True)
        self.tok = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=str(CACHE))
        self.model = AutoModel.from_pretrained(
            self.model_name, cache_dir=str(CACHE)).eval()
        self._torch = torch
        return self

    def transform(self, texts):
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i:i + self.batch_size])
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
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
