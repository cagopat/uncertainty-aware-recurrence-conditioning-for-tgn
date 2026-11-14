"""Explicit state isolation and candidate validation for diagnostic evaluation."""
import hashlib
import numpy as np
import torch

def validate_prediction_frame(frame, nodes, *, directed, self_loops, selected_pairs=None):
    nodes=np.asarray(sorted(set(nodes)),dtype=np.int64)
    if len(nodes)==0:raise ValueError('Empty candidate-node universe')
    src=frame.src_node.to_numpy(np.int64);dst=frame.dst_node.to_numpy(np.int64);p=frame.prediction.to_numpy()
    if not np.isin(src,nodes).all() or not np.isin(dst,nodes).all():raise ValueError('Out-of-universe candidate node')
    if not self_loops and (src==dst).any():raise ValueError('Unexpected self-loop candidate')
    if not directed and (src>dst).any():raise ValueError('Undirected candidates must have canonical orientation')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Non-finite or invalid probabilities')
    base=int(nodes.max())+1;keys=src*base+dst
    if len(np.unique(keys))!=len(keys):raise ValueError('Duplicate candidate pairs')
    expected=len(nodes)*(len(nodes)-1)//(1 if directed else 2)+int(self_loops)*len(nodes)
    if selected_pairs is not None:
        selected=np.asarray(selected_pairs,dtype=np.int64);wanted=selected[:,0]*base+selected[:,1]
        if not np.array_equal(np.sort(keys),np.sort(wanted)):raise ValueError('Incomplete selected candidate universe')
        expected=len(selected)
    if len(frame)!=expected:raise ValueError(f'Incomplete predictions: expected {expected}, got {len(frame)}')
    return {'expected_rows':expected,'actual_rows':len(frame),'nodes':len(nodes),'directed':directed,
            'self_loops':self_loops,'complete':True,
            'candidate_sha256':hashlib.sha256(np.sort(keys).astype('<i8').tobytes()).hexdigest()}

def ingest_tgn_snapshot(backbone, src, dst, times, edge_ids):
    """Ingest one snapshot using TGN's delayed-message convention.

    First commit the prior pending snapshot atomically, using the configured
    ``snapshot_mean`` aggregation in the prospective protocol.  Then construct
    every current-snapshot message from that same pre-current memory and leave
    those messages pending for the next snapshot/query.  Thus simultaneous event
    reordering is neutral under ``snapshot_mean`` and no same-time event can leak
    into another event's current-snapshot representation.
    """
    if backbone.model_name!='tgn':raise ValueError('Snapshot ingestion is implemented only for TGN')
    src=np.asarray(src);dst=np.asarray(dst);times=np.asarray(times,dtype=float);edge_ids=np.asarray(edge_ids)
    if not len(src):return
    if len(np.unique(times))!=1:raise ValueError('Snapshot ingestion requires exactly one timestamp')
    with torch.no_grad():
        bank=backbone.memory_bank;ids=np.concatenate([src,dst])
        backbone.update_memories(ids,bank.node_raw_messages);bank.clear_node_raw_messages(ids)
        for a,b in [(src,dst),(dst,src)]:
            unique,msg=backbone.compute_new_node_raw_messages(a,b,None,times,edge_ids)
            bank.store_node_raw_messages(unique,msg)

def snapshot_index_batches(times):
    times=np.asarray(times)
    if not len(times) or (np.diff(times)<0).any():raise ValueError('Snapshot batches require nonempty chronological data')
    bounds=np.r_[0,np.flatnonzero(np.diff(times))+1,len(times)]
    return [torch.arange(a,b,dtype=torch.long) for a,b in zip(bounds[:-1],bounds[1:])]
