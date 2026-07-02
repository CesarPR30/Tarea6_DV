"""
Generate data for "Visualizing the Hidden Activity of ANNs" (Rauber et al. 2017).

Architecture & protocol matched to the paper:
  - MLP: input -> 4 ReLU hidden layers of 1000 neurons -> softmax(10)   (N_LAYERS=4, HIDDEN=1000)
  - Inter-epoch (T1): snapshots at epochs 0, 20, 40, 60, 80, 100  (steps of 20, like Fig. 11)
  - Inter-layer (T2): the 4 hidden layers at the final epoch              (like Fig. 10)
  - Datasets: MNIST, SVHN, CIFAR-10                                       (the paper's three)

Per stage we also compute CLASSIFICATION analytics so the UI can answer
"how is the model classifying at this epoch / layer":
  - pred  : per-observation predicted class
            * T1 -> the network's real prediction at that epoch (it evolves)
            * T2 -> k-NN(k=6) prediction from the 2-D projection of that layer
                    (how separable / classifiable the representation is)
  - acc   : accuracy of those predictions on the projected subset
  - nh     : neighborhood hit (k=6)
The confusion matrix / per-class bars are derived in the browser from (label, pred),
so they also react to lasso/zoom selections.

Output: ../static/data/<ds>_{t1,t2,meta}.json  +  <ds>_sprite.png
"""
import json, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from PIL import Image

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(os.cpu_count() or 4)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "static", "data")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_SUBSET = 1000
TRAIN_SUBSET = 15000     # train on a representative subset (keeps 100 epochs feasible on CPU)
HIDDEN = 1000
N_LAYERS = 4
EPOCH_SNAPS = [0, 20, 40, 60, 80, 100]   # paper: epochs 0..100 in steps of 20
SPRITE_COLS = 40
KNN = 6


class MLP(nn.Module):
    def __init__(self, in_dim, h=HIDDEN, n_layers=N_LAYERS, out=10):
        super().__init__()
        dims = [in_dim] + [h] * n_layers
        self.hidden = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers))
        self.out = nn.Linear(h, out)
        self.drops = nn.ModuleList(nn.Dropout(min(0.2 + 0.1 * i, 0.5)) for i in range(n_layers))

    def forward(self, x, return_acts=False):
        acts = []
        for lin, drop in zip(self.hidden, self.drops):
            x = F.relu(lin(x)); acts.append(x); x = drop(x)
        logits = self.out(x)
        return (logits, acts) if return_acts else logits


def load_dataset(name):
    root = os.path.join(HERE, "_torch_data")
    flat = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    if name == "mnist":
        tr = datasets.MNIST(root, True, download=True, transform=flat)
        te = datasets.MNIST(root, False, download=True, transform=flat)
        classes = [str(i) for i in range(10)]; shape = (1, 28, 28); in_dim = 784
    elif name == "svhn":
        tr = datasets.SVHN(root, "train", download=True, transform=flat)
        te = datasets.SVHN(root, "test", download=True, transform=flat)
        classes = [str(i) for i in range(10)]; shape = (3, 32, 32); in_dim = 3072
    elif name == "cifar10":
        tr = datasets.CIFAR10(root, True, download=True, transform=flat)
        te = datasets.CIFAR10(root, False, download=True, transform=flat)
        classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]; shape = (3, 32, 32); in_dim = 3072
    else:
        raise ValueError(name)
    return tr, te, classes, shape, in_dim


def get_subset(test):
    idx = np.random.RandomState(42).permutation(len(test))[:N_SUBSET]
    xs = torch.stack([test[i][0] for i in idx])
    ys = np.array([int(test[i][1]) for i in idx])
    return xs.to(DEVICE), ys, idx


@torch.no_grad()
def extract(model, xs):
    model.eval()
    logits, acts = model(xs, return_acts=True)
    probs = F.softmax(logits, 1)
    score, pred = probs.max(1)
    return ([a.cpu().numpy() for a in acts],
            pred.cpu().numpy(), score.cpu().numpy())


def train_and_snapshot(model, loader, xs):
    opt = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-5)
    snaps_last, snaps_pred, snaps_score = {}, {}, {}
    max_ep = max(EPOCH_SNAPS)

    def snap(ep):
        acts, preds, score = extract(model, xs)
        snaps_last[ep] = acts[-1]; snaps_pred[ep] = preds; snaps_score[ep] = score

    snap(0)
    for ep in range(1, max_ep + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(xb), yb).backward(); opt.step()
        if ep in EPOCH_SNAPS:
            snap(ep)
        if ep % 10 == 0:
            print(f"    epoch {ep}/{max_ep}", flush=True)
    final_acts, final_preds, _ = extract(model, xs)
    return snaps_last, snaps_pred, snaps_score, final_acts, final_preds


def tsne_of(data, init=None, perplexity=30):
    red = PCA(n_components=min(50, data.shape[1]), random_state=0).fit_transform(data)
    params = dict(n_components=2, perplexity=perplexity, max_iter=750, random_state=0,
                  init=(init if init is not None else "pca"))
    return TSNE(**params).fit_transform(red)


def norm01(P):
    P = np.asarray(P, float); mn, mx = P.min(0), P.max(0)
    return (P - mn) / np.where(mx - mn == 0, 1, mx - mn)


def neighborhood_hit(P, ys, k=KNN):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(P)
    _, idx = nn.kneighbors(P)
    return float(np.mean([np.mean(ys[idx[i][1:]] == ys[i]) for i in range(len(P))]))


def knn_proj_pred(P, ys, k=KNN):
    """Leave-one-out k-NN prediction + vote-fraction score from the 2-D projection."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(P)
    _, idx = nn.kneighbors(P)
    preds, scores = [], []
    for i in range(len(P)):
        counts = np.bincount(ys[idx[i][1:]], minlength=int(ys.max()) + 1)
        p = int(counts.argmax())
        preds.append(p); scores.append(counts[p] / k)
    return np.array(preds), np.array(scores)


def build_t1(snaps_last, snaps_pred, snaps_score, ys):
    eps = sorted(snaps_last.keys())
    proj = norm01(tsne_of(np.concatenate([snaps_last[e] for e in eps], 0)))
    N = len(ys); stages = []
    for k, e in enumerate(eps):
        pts = proj[k * N:(k + 1) * N]; preds = snaps_pred[e]
        stages.append({
            "epoch": int(e),
            "points": [[round(float(a), 4), round(float(b), 4)] for a, b in pts],
            "pred": [int(p) for p in preds],
            "score": [round(float(s), 3) for s in snaps_score[e]],
            "nh": round(neighborhood_hit(pts, ys), 4),
            "acc": round(float(np.mean(preds == ys)), 4),
        })
    return {"labels": [int(v) for v in ys], "stages": stages, "predKind": "network"}


def build_t2(final_acts, ys):
    base = norm01(tsne_of(final_acts[0])) * 4 - 2
    stages = []
    for li, acts in enumerate(final_acts):
        pts = norm01(tsne_of(acts, init=base.astype(np.float32)))
        kp, ks = knn_proj_pred(pts, ys)
        stages.append({
            "layer": li + 1,
            "points": [[round(float(a), 4), round(float(b), 4)] for a, b in pts],
            "pred": [int(p) for p in kp],
            "score": [round(float(s), 3) for s in ks],
            "nh": round(neighborhood_hit(pts, ys), 4),
            "acc": round(float(np.mean(kp == ys)), 4),
        })
    return {"labels": [int(v) for v in ys], "stages": stages, "predKind": "knn-projection"}


def make_sprite(test, idx, name, shape):
    c, h, w = shape; n = len(idx)
    rows = math.ceil(n / SPRITE_COLS)
    mode = "L" if c == 1 else "RGB"
    sheet = Image.new(mode, (SPRITE_COLS * w, rows * h), color=(255 if c == 1 else (255, 255, 255)))
    for k, i in enumerate(idx):
        arr = test[i][0].view(c, h, w).numpy()
        if c == 1:
            img = Image.fromarray((255 - arr[0] * 255).astype(np.uint8), "L")
        else:
            img = Image.fromarray((np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8), "RGB")
        r, cc = divmod(k, SPRITE_COLS)
        sheet.paste(img, (cc * w, r * h))
    sheet.save(os.path.join(OUT, f"{name}_sprite.png"))
    return {"cols": SPRITE_COLS, "rows": rows, "tw": w, "th": h}


def run(name):
    print(f"=== {name} ===", flush=True)
    tr, te, classes, shape, in_dim = load_dataset(name)
    if TRAIN_SUBSET and len(tr) > TRAIN_SUBSET:
        sub = np.random.RandomState(1).permutation(len(tr))[:TRAIN_SUBSET]
        tr = Subset(tr, sub.tolist())
    loader = DataLoader(tr, batch_size=256, shuffle=True, num_workers=0)
    xs, ys, idx = get_subset(te)

    model = MLP(in_dim).to(DEVICE)
    snaps_last, snaps_pred, snaps_score, final_acts, final_preds = train_and_snapshot(model, loader, xs)

    print("  t-SNE T1...", flush=True); t1 = build_t1(snaps_last, snaps_pred, snaps_score, ys)
    print("  t-SNE T2...", flush=True); t2 = build_t2(final_acts, ys)
    sprite = make_sprite(te, idx, name, shape)
    meta = {"name": name, "classes": classes, "n": int(len(ys)), "hidden": HIDDEN,
            "n_layers": N_LAYERS, "epochs": EPOCH_SNAPS, "sprite": sprite,
            "final_acc": round(float(np.mean(final_preds == ys)), 4)}

    json.dump(t1, open(os.path.join(OUT, f"{name}_t1.json"), "w"))
    json.dump(t2, open(os.path.join(OUT, f"{name}_t2.json"), "w"))
    json.dump(meta, open(os.path.join(OUT, f"{name}_meta.json"), "w"))
    print(f"  done. final acc={meta['final_acc']}", flush=True)


if __name__ == "__main__":
    for ds in ["mnist", "svhn", "cifar10"]:
        run(ds)
    print("ALL DONE", flush=True)
