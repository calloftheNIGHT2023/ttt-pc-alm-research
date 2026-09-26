"""Real multi-attribute data for report 449 (TTT x PC-ALM on real tasks).

--domain nlp  : MRC Psycholinguistic Database (HF StephanAkkerman/MRC-psycholinguistic-database), 13 real word
               attributes; word embeddings from sentence-transformers/all-MiniLM-L6-v2 (mean pooled).
--domain graph: QM9 (PyG), 19 real molecular targets. A 4-layer GIN encoder is trained ONLY on the train targets
               (multi-task regression), frozen, and its pooled 128-d embedding is exported.
Output npz: E (N x D0 embeddings), A (N x P attributes, NaN = missing, each standardised over its available items),
names, train_idx, test_idx (attribute indices; test attributes are never used by any encoder or outer loop).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--domain', choices=['nlp', 'graph'], required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--nlp_encoder', default='minilm', choices=['minilm', 'glove'])
args = ap.parse_args()
dev = args.device
torch.manual_seed(0)

if args.domain == 'nlp':
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel
    ds = load_dataset('StephanAkkerman/MRC-psycholinguistic-database', split='train')
    cols = ['Number of Letters', 'Number of Phonemes', 'Number of Syllables', 'KF Written Frequency',
            'KF Number of Categories', 'KF Number of Samples', 'Thorndike-Lorge Frequency', 'Brown Verbal Frequency',
            'Familiarity', 'Concreteness', 'Imageability', 'Meaningfulness: Coloradao Norms', 'Age of Acquisition Rating']
    logcols = {'KF Written Frequency', 'KF Number of Categories', 'KF Number of Samples', 'Thorndike-Lorge Frequency',
               'Brown Verbal Frequency'}
    words, rows = {}, []
    for r in ds:
        w = str(r['Word']).strip().lower()
        if not w.isalpha() or len(w) < 2:
            continue
        vals = []
        for c in cols:
            v = r[c]
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = 0.0
            if c in ('Number of Letters',):
                vals.append(v if v > 0 else np.nan)
            elif c in logcols:
                vals.append(np.log1p(v) if v > 0 else np.nan)
            else:
                vals.append(v if v > 0 else np.nan)
        vals = np.array(vals)
        n_sem = np.isfinite(vals[8:]).sum()                      # require at least one rated (semantic) norm
        if n_sem == 0:
            continue
        if w in words:                                           # keep the entry with more available attributes
            j = words[w]
            if np.isfinite(vals).sum() <= np.isfinite(rows[j][1]).sum():
                continue
            rows[j] = (w, vals)
        else:
            words[w] = len(rows); rows.append((w, vals))
    W = [r[0] for r in rows]; A = np.stack([r[1] for r in rows])
    if args.nlp_encoder == 'glove':
        from sentence_transformers import SentenceTransformer
        name = 'sentence-transformers/average_word_embeddings_glove.6B.300d'
        st = SentenceTransformer(name, device=dev)
        E = st.encode(W, batch_size=2048, convert_to_numpy=True).astype(np.float64)
        keep = np.abs(E).sum(1) > 0                                  # drop out-of-vocabulary words (all-zero vectors)
        W = [w for w, k in zip(W, keep) if k]; A = A[keep]; E = E[keep]
    else:
        name = 'sentence-transformers/all-MiniLM-L6-v2'
        tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).to(dev).eval()
        E = []
        with torch.no_grad():
            for i in range(0, len(W), 1024):
                b = tok(W[i:i + 1024], padding=True, return_tensors='pt').to(dev)
                h = model(**b).last_hidden_state; m = b['attention_mask'].unsqueeze(-1).float()
                E.append(((h * m).sum(1) / m.sum(1)).cpu())
        E = torch.cat(E).numpy().astype(np.float64)
    names = cols
    test_names = ['Imageability', 'Age of Acquisition Rating', 'Number of Syllables', 'Brown Verbal Frequency']
    meta = dict(n_words=len(W), source='StephanAkkerman/MRC-psycholinguistic-database', encoder=name)
else:
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_add_pool
    ds = QM9('/workspace/data/qm9')
    Y = getattr(ds, '_data', None).y.numpy().astype(np.float64)[:, :19]
    names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'cv',
             'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
    test_names = ['mu', 'gap', 'cv', 'r2', 'B']
    A = Y.copy()
    A = (A - np.nanmean(A, 0)) / np.nanstd(A, 0)
    A = np.clip(A, -6, 6)                                       # rotational constants A/B/C have extreme tails
    tr = [i for i, n in enumerate(names) if n not in test_names]
    ymu = torch.tensor(np.nanmean(Y, 0), dtype=torch.float32); ysd = torch.tensor(np.nanstd(Y, 0), dtype=torch.float32)

    class GIN(torch.nn.Module):
        def __init__(s, h=128, L=4):
            super().__init__()
            s.inp = torch.nn.Linear(ds.num_node_features, h)
            s.convs = torch.nn.ModuleList([GINEConv(torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, h)),
                                                    edge_dim=ds.num_edge_features) for _ in range(L)])
            s.norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(h) for _ in range(L)])
            s.head = torch.nn.Linear(h, len(tr))
        def embed(s, b):
            x = s.inp(b.x)
            for c, nrm in zip(s.convs, s.norms):
                x = x + torch.relu(nrm(c(x, b.edge_index, b.edge_attr)))
            return global_add_pool(x, b.batch) / 10.0
        def forward(s, b):
            return s.head(s.embed(b))
    net = GIN().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    idx = torch.randperm(len(ds))
    train_set = ds[idx[:110000]]
    loader = DataLoader(train_set, batch_size=256, shuffle=True)
    for ep in range(8):
        tot = 0.0
        for b in loader:
            b = b.to(dev); p = net(b)
            t = ((b.y[:, :19].cpu() - ymu) / ysd).clamp(-6, 6)[:, tr].to(dev)   # labels carried by the batch (QM9 idx != position)
            loss = ((p - t) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * b.num_graphs
        print('gin epoch', ep, tot / len(train_set), flush=True)
    net.eval(); E = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=1024):
            E.append(net.embed(b.to(dev)).cpu())
    E = torch.cat(E).numpy().astype(np.float64)
    W = None
    meta = dict(n_molecules=len(ds), encoder='GINE 4x128 trained on train targets only (8 epochs, 110k molecules)',
                gin_final_train_mse=tot / len(train_set))

names = list(names)
test_idx = [names.index(n) for n in test_names]
train_idx = [i for i in range(len(names)) if i not in test_idx]
if args.domain == 'nlp':
    mu = np.nanmean(A, 0); sd = np.nanstd(A, 0); A = (A - mu) / sd
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, E=E, A=A, names=np.array(names), train_idx=np.array(train_idx), test_idx=np.array(test_idx))
meta.update(names=names, train=[names[i] for i in train_idx], test=[names[i] for i in test_idx],
            available=[int(np.isfinite(A[:, j]).sum()) for j in range(A.shape[1])], E_shape=list(E.shape))
Path(str(out) + '.json').write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
