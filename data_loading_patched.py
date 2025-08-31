# data_loading_patched.py — fast streaming loader for Siamese training (Keras 3–friendly)
import os, glob, re, random
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from skimage import exposure
from keras.utils import Sequence  # NOTE: keras.utils (not tf.keras.utils)

ALLOWED_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
TOK_HAND = ("left", "right")
TOK_FINGER = ("index", "middle", "ring")

# ---------- robust path expanders ----------
def _list_images(pattern: str) -> List[str]:
    """Works with dir paths, **/*.* or **/*.bmp; falls back to os.walk if glob finds none."""
    pattern = pattern.replace("\\", "/")

    def _glob_all(base):
        out = []
        for ext in ALLOWED_EXTS:
            out.extend(glob.glob(base + ext, recursive=True))
        return out

    # Directory given?
    if os.path.isdir(pattern):
        files = []
        for r, _, fns in os.walk(pattern):
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in ALLOWED_EXTS:
                    files.append(os.path.join(r, fn))
        return sorted(files)

    # **/*.* catch-all
    if pattern.endswith("**/*.*"):
        base = pattern[:-4]
        files = _glob_all(base)
        if files: return sorted(files)
        root = base.split("/**", 1)[0]
        return sorted(_list_images(root))  # recurse as dir

    # specific extension
    lower = pattern.lower()
    if any(lower.endswith(ext) for ext in ALLOWED_EXTS):
        files = glob.glob(pattern, recursive=True)
        if files: return sorted(files)
        # fallback walk
        root = pattern.split("/**", 1)[0]
        if os.path.isdir(root):
            return _list_images(root)
        return []

    # generic: treat as dir base
    base = pattern.rstrip("/") + "/**/*"
    files = _glob_all(base)
    if files: return sorted(files)
    root = base.split("/**", 1)[0]
    return _list_images(root) if os.path.isdir(root) else []

# ---------- metadata parsing ----------
def _find_subject_id(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if parts and "." in parts[-1]: parts = parts[:-1]
    for p in reversed(parts):
        if re.fullmatch(r"\d+", p): return p
    return parts[-1] if parts else "unknown"

def _parse_hand_finger(path: str) -> Tuple[str, str]:
    p = path.replace("\\", "/").lower()
    hand = "anyhand"
    for h in TOK_HAND:
        if f"/{h}/" in p or f"_{h}_" in p or p.endswith(f"_{h}.bmp"):
            hand = h; break
    finger = "anyfinger"
    for f in TOK_FINGER:
        if f"/{f}/" in p or f"_{f}_" in p or p.endswith(f"_{f}.bmp"):
            finger = f; break
    return hand, finger

def _index_by_key(files: List[str]) -> Dict[Tuple[str,str,str], List[str]]:
    by = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in ALLOWED_EXTS: continue
        sid = _find_subject_id(f); hand, finger = _parse_hand_finger(f)
        by.setdefault((sid, hand, finger), []).append(f)
    for k in by: by[k].sort()
    return by

# ---------- image IO & preprocessing ----------
def _load_preprocess(path: str, target_size: Optional[Tuple[int,int]]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]))  # (W,H)
    arr = np.asarray(img, dtype="float32")
    arr = exposure.equalize_hist(arr).astype("float32")  # [0,1]
    if arr.ndim == 2: arr = arr[..., None]
    return arr

# ---------- streaming sequence ----------
class StreamingPairSequence(Sequence):
    """On-the-fly balanced pairs; caches decoded images for speed."""
    def __init__(self, groups: Dict[Tuple[str,str,str], List[str]],
                 batch_size: int, steps_per_epoch: int,
                 target_size: Optional[Tuple[int,int]] = (96,96),
                 pos_fraction: float = 0.5,
                 same_type_neg_ratio: float = 0.8,
                 seed: int = 42, cache_images: bool = True, **kwargs):
        super().__init__(**kwargs)  # Keras 3 expects this
        from collections import defaultdict
        self.groups = groups
        self.batch = batch_size
        self.steps = steps_per_epoch
        self.target_size = target_size
        self.pos_fraction = pos_fraction
        self.same_type_neg_ratio = same_type_neg_ratio
        self.rng = random.Random(seed)
        self.cache = {} if cache_images else None

        self.keys = list(groups.keys())
        self.pos_keys = [k for k in self.keys if len(groups[k]) >= 2]
        # group by (hand,finger) for “same-type” negatives
        self.by_type = defaultdict(list)
        for (sid, hand, finger), lst in groups.items():
            self.by_type[(hand, finger)].append((sid, lst))
        self.type_keys = list(self.by_type.keys())

        if not self.pos_keys:
            raise RuntimeError("No groups with >=2 images to form positives.")

    def __len__(self): return self.steps

    def _img(self, p):
        if self.cache is not None:
            if p not in self.cache:
                self.cache[p] = _load_preprocess(p, self.target_size)
            return self.cache[p]
        return _load_preprocess(p, self.target_size)

    def __getitem__(self, _):
        n = self.batch
        n_pos = int(n * self.pos_fraction)
        n_neg = n - n_pos
        X1 = np.empty((n, self.target_size[0], self.target_size[1], 1), dtype="float32")
        X2 = np.empty_like(X1)
        Y  = np.empty((n, 1), dtype="float32")

        # positives
        for i in range(n_pos):
            k = self.rng.choice(self.pos_keys)
            a, b = self.rng.sample(self.groups[k], 2)
            X1[i], X2[i], Y[i] = self._img(a), self._img(b), 1.0

        # negatives (mix same-type and cross-type)
        i = n_pos
        n_same = int(n_neg * self.same_type_neg_ratio)
        # same-type (different subject)
        for _ in range(n_same):
            hf = self.rng.choice(self.type_keys)
            subj_lists = self.by_type[hf]
            if len(subj_lists) < 2:
                continue
            (s1, L1), (s2, L2) = self.rng.sample(subj_lists, 2)
            a, b = self.rng.choice(L1), self.rng.choice(L2)
            X1[i], X2[i], Y[i] = self._img(a), self._img(b), 0.0
            i += 1
            if i >= n: break
        # cross-type
        while i < n:
            k1, k2 = self.rng.sample(self.keys, 2)
            if k1 == k2: continue
            a, b = self.rng.choice(self.groups[k1]), self.rng.choice(self.groups[k2])
            X1[i], X2[i], Y[i] = self._img(a), self._img(b), 0.0
            i += 1

        # shuffle within batch
        idx = np.random.permutation(n)
        # IMPORTANT: return a TUPLE of inputs, not a list (Keras 3)
        return (X1[idx], X2[idx]), Y[idx]

# ---------- convenience: fixed arrays for final eval ----------
def _make_pairs(by: Dict[Tuple[str,str,str], List[str]], total_pairs: int,
                target_size: Optional[Tuple[int,int]], seed: int,
                same_type_neg_ratio: float = 0.8):
    rng = random.Random(seed)
    keys = list(by.keys())
    pos_keys = [k for k in keys if len(by[k]) >= 2]
    if not pos_keys: raise RuntimeError("No groups with >=2 images to form positives.")

    from collections import defaultdict
    by_type = defaultdict(list)
    for (sid, hand, finger), lst in by.items():
        by_type[(hand, finger)].append((sid, lst))
    type_keys = list(by_type.keys())

    n_pos = total_pairs // 2
    n_neg = total_pairs - n_pos
    n_same = int(n_neg * same_type_neg_ratio)
    n_cross = n_neg - n_same

    X1, X2, Y = [], [], []
    for _ in range(n_pos):
        k = rng.choice(pos_keys)
        a, b = rng.sample(by[k], 2)
        X1.append(_load_preprocess(a, target_size)); X2.append(_load_preprocess(b, target_size)); Y.append(1.0)
    for _ in range(n_same):
        hf = rng.choice(type_keys)
        subj_lists = by_type[hf]
        if len(subj_lists) < 2:
            n_cross += 1; continue
        (s1, L1), (s2, L2) = rng.sample(subj_lists, 2)
        a, b = rng.choice(L1), rng.choice(L2)
        X1.append(_load_preprocess(a, target_size)); X2.append(_load_preprocess(b, target_size)); Y.append(0.0)
    for _ in range(n_cross):
        k1, k2 = rng.sample(keys, 2)
        if k1 == k2: continue
        a, b = rng.choice(by[k1]), rng.choice(by[k2])
        X1.append(_load_preprocess(a, target_size)); X2.append(_load_preprocess(b, target_size)); Y.append(0.0)
    X1, X2, Y = map(np.array, (X1, X2, Y))
    idx = np.random.permutation(len(Y))
    return np.stack([X1[idx], X2[idx]], 1), Y[idx].astype("float32")

# ---------- public helpers ----------
def build_stream(train_glob: str, test_glob: str,
                 pairs_train: int, pairs_test: int,
                 batch_size: int, target_size: Optional[Tuple[int,int]],
                 seed: int = 42, same_type_neg_ratio: float = 0.8,
                 cache_images: bool = True):
    train_files = _list_images(train_glob)
    test_files  = _list_images(test_glob)
    if len(train_files) == 0: raise RuntimeError(f"No training images found for: {train_glob}")
    if len(test_files)  == 0: raise RuntimeError(f"No test images found for: {test_glob}")

    by_tr = _index_by_key(train_files)
    by_te = _index_by_key(test_files)

    steps_tr = max(1, pairs_train // batch_size)
    steps_te = max(1, pairs_test  // batch_size)

    tr_seq = StreamingPairSequence(by_tr, batch_size, steps_tr, target_size, 0.5, same_type_neg_ratio, seed,   cache_images)
    va_seq = StreamingPairSequence(by_te, batch_size, steps_te, target_size, 0.5, same_type_neg_ratio, seed+1, cache_images)

    # Fixed test set (arrays) for calibration
    te_pairs, te_y = _make_pairs(by_te, pairs_test, target_size, seed+2, same_type_neg_ratio)
    return tr_seq, va_seq, te_pairs, te_y
