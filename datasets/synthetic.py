"""Small six-node converging river tree with missing labels."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from data.schema import GraphEventBatch

EDGE_INDEX=torch.tensor([[0,1,2,3,4],[2,2,3,4,5]],dtype=torch.long) # 0+1 -> 2 -> 3 -> 4 -> 5

class SyntheticEventDataset(Dataset[GraphEventBatch]):
    def __init__(self,count:int=16,history:int=12,horizon:int=6,dynamic_dim:int=4,node_static_dim:int=3,edge_static_dim:int=2,seed:int=42)->None:
        self.items=[]; g=torch.Generator().manual_seed(seed); n=6
        node=torch.rand(n,node_static_dim,generator=g); node[:,0]=torch.linspace(20,120,n) # drainage area km2
        edge=torch.rand(5,edge_static_dim,generator=g); edge[:,0]=torch.linspace(8000,16000,5); edge[:,1]=torch.linspace(0.0005,0.003,5)
        for _ in range(count):
            rain=torch.rand(history+horizon,n,1,generator=g)*8; dyn=torch.rand(history,n,dynamic_dim,generator=g); qh=torch.rand(history,n,generator=g)*20; zh=0.8+0.15*torch.sqrt(qh)
            # Smooth learnable target, not claimed as real hydrology.
            future=rain[history:].squeeze(-1); qt=0.65*qh[-1:].expand(horizon,-1)+future*node[:,0][None]*0.02; zt=0.8+0.15*torch.sqrt(qt)
            qm=torch.rand(history,n,generator=g)>.15; zm=torch.rand(history,n,generator=g)>.2; qtm=torch.rand(horizon,n,generator=g)>.1; ztm=torch.rand(horizon,n,generator=g)>.15
            self.items.append(GraphEventBatch(dyn,rain,node,EDGE_INDEX,edge,qh,zh,qm,zm,qt,zt,qtm,ztm,torch.ones(n,dtype=torch.bool),torch.tensor(True)))
    def __len__(self)->int:return len(self.items)
    def __getitem__(self,i:int)->GraphEventBatch:return self.items[i]

def collate_graph_events(items:list[GraphEventBatch])->GraphEventBatch:
    static={"node_static","edge_index","edge_static"}; kwargs={}
    for key in GraphEventBatch.__dataclass_fields__:
        vals=[getattr(x,key) for x in items]
        if vals[0] is None: kwargs[key]=None
        elif key in static: kwargs[key]=vals[0]
        else: kwargs[key]=torch.stack(vals)
    return GraphEventBatch(**kwargs)
