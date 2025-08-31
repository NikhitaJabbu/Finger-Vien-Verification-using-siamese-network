# keras_siamese_MLP1_loop_patched.py — faster training w/ streaming + separable conv + mixed precision
import argparse, os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.optimizers import Adam

from data_loading_patched import build_stream  # streaming API

def set_seed(seed=42):
    import random
    np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)

def maybe_enable_mixed_precision():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[MP] Mixed precision enabled (float16 on GPU).")
        else:
            print("[MP] CPU only: mixed precision not enabled.")
    except Exception as e:
        print("[MP] Skipped mixed precision:", e)

def sep_block(x, c):
    x = layers.SeparableConv2D(c, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return x

def create_branch(input_shape, embedding_dim=128):
    inp = layers.Input(shape=input_shape)
    x = sep_block(inp, 24)
    x = sep_block(x,   48)
    x = sep_block(x,   96)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    emb = layers.Dense(embedding_dim, activation=None, name="embedding")(x)
    emb = layers.Lambda(lambda t: K.l2_normalize(t, axis=1), name="l2norm")(emb)
    return Model(inp, emb, name="branch")

def euclidean_distance(v):
    a, b = v
    return tf.cast(K.sqrt(K.maximum(K.sum(K.square(a - b), axis=1, keepdims=True), K.epsilon())), tf.float32)

def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, y_pred.dtype)
        pos = y_true * K.square(y_pred)
        neg = (1.0 - y_true) * K.square(K.maximum(margin - y_pred, 0.0))
        return K.mean(pos + neg)
    return loss

def distance_accuracy(threshold=1.0):
    def acc(y_true, y_pred):
        y_pred_bin = K.cast(y_pred < threshold, y_pred.dtype)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.mean(K.cast(K.equal(y_true, y_pred_bin), y_pred.dtype))
    return acc

def compute_accuracy(y_true, distances, threshold=1.0):
    y_hat = (distances < threshold).astype(np.float32).flatten()
    return float((y_hat == y_true.astype(np.float32)).mean())

def best_threshold(d, y, n=400):
    d = d.ravel(); y = y.astype(np.float32).ravel()
    ts = np.linspace(d.min(), d.max(), n)
    accs = [(((d < t).astype(np.float32) == y).mean(), t) for t in ts]
    accs.sort(reverse=True)
    return accs[0][1], accs[0][0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="D:/Projects/FingerVeinVerification_2.0/data/train", help="Glob or directory for training images")
    ap.add_argument("--test",  default="D:/Projects/FingerVeinVerification_2.0/data/test",  help="Glob or directory for test images")
    ap.add_argument("--pairs-train", type=int, default=12000)
    ap.add_argument("--pairs-test",  type=int, default=3000)
    ap.add_argument("--resize", type=int, default=96, help="Square resize; 0 = no resize")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--same-type-neg", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="outputs_fv_siamese_fast")
    args = ap.parse_args()

    set_seed(42)
    maybe_enable_mixed_precision()
    os.makedirs(args.outdir, exist_ok=True)
    target = (args.resize, args.resize) if args.resize and args.resize > 0 else None

    print("[0/5] Devices:", tf.config.list_physical_devices())
    print("[1/5] Building streaming datasets...")
    tr_seq, va_seq, te_pairs, te_y = build_stream(
        train_glob=args.train, test_glob=args.test,
        pairs_train=args.pairs_train, pairs_test=args.pairs_test,
        batch_size=args.batch, target_size=target,
        seed=42, same_type_neg_ratio=args.same_type_neg, cache_images=True
    )

    input_shape = (target[0], target[1], 1) if target else te_pairs.shape[2:]
    left  = layers.Input(shape=input_shape, name="left")
    right = layers.Input(shape=input_shape, name="right")
    branch = create_branch(input_shape)
    dist = layers.Lambda(euclidean_distance, name="euclidean_distance")([branch(left), branch(right)])
    siamese = Model([left, right], dist, name="siamese_contrastive")

    siamese.compile(optimizer=Adam(args.lr),
                    loss=contrastive_loss(args.margin),
                    metrics=[distance_accuracy(args.threshold)])

    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ]

    print("[2/5] Training (streaming)...")
    hist = siamese.fit(tr_seq, validation_data=va_seq, epochs=args.epochs, verbose=2, callbacks=cbs)

    print("[3/5] Evaluating on fixed test set...")
    d_test = siamese.predict([te_pairs[:,0], te_pairs[:,1]], batch_size=args.batch, verbose=0)
    acc_cfg = compute_accuracy(te_y, d_test, threshold=args.threshold)
    thr_best, acc_best = best_threshold(d_test, te_y)
    pos = d_test[te_y==1].ravel(); neg = d_test[te_y==0].ravel()
    print(f"Accuracy @ threshold={args.threshold:.3f}: {acc_cfg:.4f}")
    print(f"Best threshold ≈ {thr_best:.3f} | Accuracy @best = {acc_best:.4f}")
    print(f"Distance means (pos, neg): {pos.mean():.3f}, {neg.mean():.3f}")

    print("[4/5] Saving...")
    model_path = os.path.join(args.outdir, "siamese_fv.keras")
    history_path = os.path.join(args.outdir, "history.json")
    with open(history_path, "w") as f: json.dump(hist.history, f)
    siamese.save(model_path)
    with open(os.path.join(args.outdir, "calibration.json"), "w") as f:
        json.dump({
            "threshold_configured": float(args.threshold),
            "accuracy_at_configured": float(acc_cfg),
            "threshold_best": float(thr_best),
            "accuracy_at_best": float(acc_best),
            "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean())
        }, f, indent=2)
    print("Saved:", model_path)
    print("       ", history_path)
    print("       ", os.path.join(args.outdir, "calibration.json"))

if __name__ == "__main__":
    main()
