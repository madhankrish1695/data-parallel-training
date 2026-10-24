"""
train_ddp.py - Synchronous data-parallel training of ResNet-18 on CIFAR-10
using PyTorch DistributedDataParallel (DDP) with the NCCL (GPU) or Gloo (CPU) backend.

Launch (single node, N GPUs):
    torchrun --nproc_per_node=N train_ddp.py --scaling strong --tag strong_N

What it records (rank 0 writes results/<tag>.json):
  * per-epoch wall-clock time (data loading + compute + gradient sync)  -> "Response time"
  * test accuracy after every epoch and cumulative training time        -> convergence curve
  * communication cost: measured in a profiling phase before training
        - t_compute : steps run under model.no_sync()  (no all-reduce)
        - t_full    : normal DDP steps (all-reduce overlapped with backward)
        - exposed communication per step = t_full - t_compute
        - raw all-reduce time of a gradient-sized tensor (micro-benchmark)
"""
import argparse, json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torchvision, torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# ----------------------------------------------------------------- model
class BasicBlock(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False);      self.b2 = nn.BatchNorm2d(o)
        self.sh = nn.Sequential()
        if stride != 1 or i != o:
            self.sh = nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.sh(x))

class ResNet18(nn.Module):
    """ResNet-18 adapted to 32x32 inputs: 3x3 stem conv, no max-pool (standard CIFAR variant)."""
    def __init__(self, nc=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        cfg, layers, i = [(64, 1), (128, 2), (256, 2), (512, 2)], [], 64
        for o, s in cfg:
            layers += [BasicBlock(i, o, s), BasicBlock(o, o, 1)]; i = o
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, nc)
    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ----------------------------------------------------------------- helpers
def sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()

def get_data(args, rank, world):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    # rank 0 downloads first, others wait (avoids file corruption)
    if rank == 0: torchvision.datasets.CIFAR10(args.data, True, download=True)
    dist.barrier()
    train = torchvision.datasets.CIFAR10(args.data, True, transform=tr)
    test = torchvision.datasets.CIFAR10(args.data, False, transform=te)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    per_worker_bs = args.batch_size // world if args.scaling == "strong" else args.batch_size
    loader = DataLoader(train, per_worker_bs, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    test_loader = DataLoader(test, 500, shuffle=False, num_workers=2)
    return loader, test_loader, sampler, per_worker_bs

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); correct = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        correct += (model(x).argmax(1) == y).sum().item()
    model.train()
    return 100.0 * correct / len(loader.dataset)

def one_step(model, opt, x, y, scaler, amp):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=x.device.type, enabled=amp,
                        dtype=torch.float16 if x.device.type == "cuda" else torch.bfloat16):
        loss = F.cross_entropy(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    return loss

def profile_comm(ddp_model, loader, dev, opt, scaler, amp, steps=30, warm=10):
    """Exposed communication per step = t(sync step) - t(no_sync step). Same batches/model."""
    it = iter(loader); batches = []
    for _ in range(steps + warm):
        try: batches.append(next(it))
        except StopIteration: it = iter(loader); batches.append(next(it))
    def timed(use_nosync):
        times = []
        for k, (x, y) in enumerate(batches):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            sync(dev); dist.barrier(); t = time.perf_counter()
            if use_nosync:
                with ddp_model.no_sync(): one_step(ddp_model, opt, x, y, scaler, amp)
            else:
                one_step(ddp_model, opt, x, y, scaler, amp)
            sync(dev)
            if k >= warm: times.append(time.perf_counter() - t)
        return sum(times) / len(times)
    t_comp = timed(True); t_full = timed(False)
    # micro-benchmark: raw all-reduce of a tensor the size of the gradients (no overlap)
    n = sum(p.numel() for p in ddp_model.parameters())
    buf = torch.randn(n, device=dev); res = []
    for k in range(25):
        sync(dev); dist.barrier(); t = time.perf_counter()
        dist.all_reduce(buf); sync(dev)
        if k >= 5: res.append(time.perf_counter() - t)
    return t_comp, t_full, sum(res) / len(res), n * 4 / 1e6


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="weak scaling: per-worker batch; strong scaling: GLOBAL batch")
    ap.add_argument("--scaling", choices=["weak", "strong"], default="weak")
    ap.add_argument("--lr", type=float, default=0.1, help="base lr for a global batch of 128")
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--amp", action="store_true", help="mixed precision")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    dev = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda: torch.cuda.set_device(dev)
    dist.init_process_group("nccl" if use_cuda else "gloo")
    torch.manual_seed(args.seed); torch.backends.cudnn.benchmark = True

    loader, test_loader, sampler, bs = get_data(args, rank, world)
    global_bs = bs * world
    model = ResNet18().to(dev).to(memory_format=torch.channels_last) if use_cuda else ResNet18()
    ddp = DDP(model, device_ids=[local] if use_cuda else None, bucket_cap_mb=25)

    peak_lr = args.lr * global_bs / 128          # linear scaling rule (Goyal et al. 2017)
    opt = torch.optim.SGD(ddp.parameters(), lr=peak_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps_per_epoch = len(loader); total = args.epochs * steps_per_epoch
    warm = args.warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + __import__("math").cos(3.14159265 * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler(enabled=args.amp and use_cuda)

    # --- communication profiling (state is restored afterwards so training is unaffected)
    state = {k: v.clone() for k, v in ddp.state_dict().items()}
    t_comp, t_full, t_ar, mb = profile_comm(ddp, loader, dev, opt, scaler, args.amp)
    ddp.load_state_dict(state); opt.state.clear()
    for g in opt.param_groups: g["lr"] = peak_lr / warm
    if rank == 0:
        print(f"[comm] compute-only {t_comp*1e3:.1f} ms | with sync {t_full*1e3:.1f} ms | "
              f"exposed comm {(t_full-t_comp)*1e3:.1f} ms | raw all-reduce({mb:.1f} MB) {t_ar*1e3:.1f} ms", flush=True)

    log = dict(args=vars(args), world=world, per_worker_bs=bs, global_bs=global_bs, peak_lr=peak_lr,
               steps_per_epoch=steps_per_epoch, grad_mb=mb, t_compute_step=t_comp, t_full_step=t_full,
               t_allreduce_raw=t_ar, epochs=[])
    cum = 0.0
    for ep in range(args.epochs):
        sampler.set_epoch(ep); ddp.train()
        sync(dev); dist.barrier(); t0 = time.perf_counter(); dl_wait = 0.0; tl = time.perf_counter()
        for x, y in loader:
            dl_wait += time.perf_counter() - tl                      # time blocked on data loader
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            one_step(ddp, opt, x, y, scaler, args.amp); sched.step()
            tl = time.perf_counter()
        sync(dev); dist.barrier(); t_ep = time.perf_counter() - t0
        cum += t_ep
        acc = evaluate(ddp.module, test_loader, dev) if rank == 0 else 0.0
        if rank == 0:
            log["epochs"].append(dict(epoch=ep + 1, time=t_ep, cum_time=cum, test_acc=acc, data_wait=dl_wait))
            print(f"ep {ep+1:3d} | {t_ep:6.1f}s | cum {cum:7.1f}s | acc {acc:5.2f}% | data-wait {dl_wait:5.1f}s", flush=True)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        json.dump(log, open(f"{args.out}/{args.tag}.json", "w"), indent=1)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
