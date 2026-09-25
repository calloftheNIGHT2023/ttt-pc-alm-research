"""Extract real representations from small real models in three domains (Exp C, report 440).

NLP  : sentence-transformers/all-MiniLM-L6-v2 mean-pooled embeddings of AG News texts (384-d)
CV   : torchvision ResNet-18 (ImageNet weights) penultimate features of CIFAR-10 images (512-d)
Graph: 2-layer GCN trained on PubMed node classification, hidden-layer node embeddings (256-d)
Also saves the real labels of each dataset for real-label tasks.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--domains', nargs='+', default=['nlp', 'cv', 'graph'])
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--n_nlp', type=int, default=30000)
ap.add_argument('--n_cv', type=int, default=30000)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
dev = args.device
torch.manual_seed(0)

if 'nlp' in args.domains:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel
    ds = load_dataset('fancyzhx/ag_news', split='train').shuffle(seed=0).select(range(args.n_nlp))
    name = 'sentence-transformers/all-MiniLM-L6-v2'
    tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).to(dev).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            b = tok(ds[i:i + 512]['text'], padding=True, truncation=True, max_length=128, return_tensors='pt').to(dev)
            h = model(**b).last_hidden_state
            m = b['attention_mask'].unsqueeze(-1).float()
            feats.append(((h * m).sum(1) / m.sum(1)).cpu())
    F = torch.cat(feats).numpy().astype(np.float64)
    np.savez(out / 'nlp_agnews_minilm.npz', X=F, label=np.array(ds['label']))
    print('nlp', F.shape, flush=True)

if 'cv' in args.domains:
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    from datasets import load_dataset
    hf = load_dataset('uoft-cs/cifar10', split='train').shuffle(seed=0).select(range(args.n_cv))

    class HFSet(torch.utils.data.Dataset):
        def __len__(s): return len(hf)
        def __getitem__(s, i):
            r = hf[i]; return tf(r['img'].convert('RGB')), r['label']
    loader = torch.utils.data.DataLoader(HFSet(), batch_size=256, num_workers=8)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity(); net = net.to(dev).eval()
    feats, labs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            feats.append(net(xb.to(dev)).cpu()); labs.append(yb)
    F = torch.cat(feats).numpy().astype(np.float64)
    np.savez(out / 'cv_cifar10_resnet18.npz', X=F, label=torch.cat(labs).numpy())
    print('cv', F.shape, flush=True)

if 'graph' in args.domains:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv
    ds = Planetoid('/workspace/data/planetoid', 'PubMed'); data = ds[0].to(dev)

    class GCN(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.c1 = GCNConv(ds.num_features, 256); s.c2 = GCNConv(256, ds.num_classes)
        def forward(s, x, ei):
            h = torch.relu(s.c1(x, ei)); return s.c2(torch.nn.functional.dropout(h, 0.5, s.training), ei), h
    net = GCN().to(dev); opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(200):
        net.train(); opt.zero_grad()
        o, _ = net(data.x, data.edge_index)
        torch.nn.functional.cross_entropy(o[data.train_mask], data.y[data.train_mask]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        o, h = net(data.x, data.edge_index)
        acc = (o.argmax(-1)[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    F = h.cpu().numpy().astype(np.float64)
    np.savez(out / 'graph_pubmed_gcn.npz', X=F, label=data.y.cpu().numpy())
    print('graph', F.shape, 'gcn test acc', round(acc, 4), flush=True)
print('DONE')
