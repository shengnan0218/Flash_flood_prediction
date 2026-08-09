from __future__ import annotations
import tempfile,unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import load_config
from data.schema import validate_batch
from datasets import SyntheticEventDataset,collate_graph_events
from models.runoff import WaterBalanceLSTMCell
from models.routing import KinematicWaveGNN
from models import HybridFloodModel
from trainers import Trainer

ROOT=Path(__file__).parents[1]
def batch_and_cfg(name="hunan_e4.yaml",count=2):
    c=load_config(ROOT/"configs"/name);c["solver"]["dx"]=1000.;ds=SyntheticEventDataset(count,c["history_length"],c["forecast_horizon"],c["dynamic_dim"],c["node_static_dim"],c["edge_static_dim"]);return c,ds,collate_graph_events([ds[i] for i in range(count)])

class TestSystem(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_water_balance_and_nonnegative(self):
        cell=WaterBalanceLSTMCell(3,8);b=12;z=torch.zeros(b,8);s=(z,z.clone(),torch.zeros(b),torch.zeros(b));rain=torch.rand(b)*20
        for _ in range(8):
            runoff,s,d=cell(torch.rand(b,3),rain,s);self.assertLess(d["residual"].abs().max().item(),5e-6);self.assertTrue((runoff>=0).all() and (d["storage_fast"]>=0).all() and (d["storage_slow"]>=0).all())
    def test_routing_mass_cfl_confluence_and_direction(self):
        c,ds,b=batch_and_cfg();r=KinematicWaveGNN(c["node_static_dim"],c["edge_static_dim"],8,c["physical_bounds"],c["solver"]);ql=torch.ones(2,4,6);q,d=r(ql,b.node_static,b.edge_index,b.edge_static);self.assertLess(d["routing_mass_balance_residual"].abs().max().item(),2e-2);self.assertTrue((q>=0).all());self.assertTrue(d["explicit_equivalent_substeps"].max()>=1);self.assertLessEqual(d["implicit_relative_residual"].max(),c["solver"]["implicit_residual_tolerance"]);self.assertTrue((q[:,:,2]>=ql[:,:,2]).all())
        changed=ql.clone();changed[:,:,5]+=1000;q2,_=r(changed,b.node_static,b.edge_index,b.edge_static);self.assertTrue(torch.allclose(q2[:,:,:5],q[:,:,:5],atol=1e-5))
    def test_cfl_is_diagnostic_only(self):
        c,_,b=batch_and_cfg();strict=dict(c["solver"]);strict["cfl"]=0.1;r1=KinematicWaveGNN(c["node_static_dim"],c["edge_static_dim"],8,c["physical_bounds"],c["solver"]);r2=KinematicWaveGNN(c["node_static_dim"],c["edge_static_dim"],8,c["physical_bounds"],strict);r2.load_state_dict(r1.state_dict());lateral=torch.full((2,6,6),100.);q1,d1=r1(lateral,b.node_static,b.edge_index,b.edge_static);q2,d2=r2(lateral,b.node_static,b.edge_index,b.edge_static);torch.testing.assert_close(q1,q2);torch.testing.assert_close(d1["edge_storage"],d2["edge_storage"]);self.assertGreater(d2["explicit_equivalent_substeps"].max(),d1["explicit_equivalent_substeps"].max());self.assertGreater(d2["explicit_cfl_exceedance_count"].item(),0)
    def test_four_modes_backward_and_edge_gradient(self):
        for name in ("hunan_e1_pure_ai.yaml","hunan_e2_physics_runoff.yaml","hunan_e3_physics_routing.yaml","hunan_e4.yaml"):
            c,_,b=batch_and_cfg(name);validate_batch(b,{k:c[k] for k in ("history_length","forecast_horizon","node_static_dim","edge_static_dim")});m=HybridFloodModel(c,6);o=m(b);loss=o["q"].mean()+o["z"].mean();loss.backward();self.assertEqual(o["q"].shape,b.q_target.shape)
            if hasattr(m.routing,"edge_net"):self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.routing.edge_net.parameters() if p.grad is not None),0)
    def test_checkpoint_and_training_loss(self):
        c,ds,_=batch_and_cfg("hunan_e1_pure_ai.yaml",8);loader=DataLoader(ds,batch_size=2,collate_fn=collate_graph_events);m=HybridFloodModel(c,6)
        with tempfile.TemporaryDirectory() as td:
            c["training"]={"epochs":3,"patience":3,"gradient_clip":1.,"checkpoint":str(Path(td)/"x.pt"),"log_csv":str(Path(td)/"x.csv")};c["debug_max_batches"]=4;t=Trainer(m,c,torch.device("cpu"));hist=t.fit(loader);self.assertTrue(Path(c["training"]["checkpoint"]).exists());before={k:v.clone() for k,v in m.state_dict().items()};t.load_checkpoint(c["training"]["checkpoint"]);self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in m.state_dict().items()));self.assertLessEqual(min(x["loss"] for x in hist),hist[0]["loss"])
if __name__=="__main__":unittest.main()
