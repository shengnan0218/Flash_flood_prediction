from __future__ import annotations
import argparse,time
from pathlib import Path
import torch
from config import load_config
from data.device import resolve_device
from datasets import SyntheticEventDataset,collate_graph_events
from models import HybridFloodModel

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--batch-size",type=int,default=2);p.add_argument("--history",type=int,default=12);p.add_argument("--horizon",type=int,default=6);p.add_argument("--repeats",type=int,default=2);a=p.parse_args()
    for name in ("hunan_e1_pure_ai.yaml","hunan_e2_physics_runoff.yaml","hunan_e3_physics_routing.yaml","hunan_e4.yaml"):
        cfg=load_config(Path("configs")/name);cfg["batch_size"]=a.batch_size;cfg["history_length"]=a.history;cfg["forecast_horizon"]=a.horizon;dev=resolve_device(cfg["device"],cfg["gpu_id"]);item=SyntheticEventDataset(a.batch_size,a.history,a.horizon,cfg["dynamic_dim"],cfg["node_static_dim"],cfg["edge_static_dim"]);batch=collate_graph_events([item[i] for i in range(a.batch_size)]).to(dev);m=HybridFloodModel(cfg,6).to(dev);times=[]
        for _ in range(a.repeats):
            if dev.type=="cuda":torch.cuda.reset_peak_memory_stats(dev);torch.cuda.synchronize()
            t=time.perf_counter();o=m(batch);(o["q"].mean()+o["z"].mean()).backward()
            if dev.type=="cuda":torch.cuda.synchronize()
            times.append(time.perf_counter()-t);m.zero_grad(set_to_none=True)
        print(name,"parameters",sum(x.numel() for x in m.parameters()),"forward_backward_s",sum(times)/len(times),"peak_gpu_bytes",torch.cuda.max_memory_allocated(dev) if dev.type=="cuda" else 0)
if __name__=="__main__":main()
