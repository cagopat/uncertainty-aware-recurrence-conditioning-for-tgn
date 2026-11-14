"""Opt-in temporal gates. Legacy KEAT modes remain in MemoryModel unchanged."""
import hashlib
import json
import os
import numpy as np
import torch

def stable_row_permutations(node_ids, query_times, neighbor_ids, edge_ids, gaps, layer, seed):
    """Deterministic changed assignment, independent of batch/order/RNG state.

    Neighborhood events are canonically ordered before assigning the permutation.
    Identical repeated events are exchangeable. Singleton/empty rows are identity;
    all-tied rows may have an index permutation but unchanged numerical weights.
    """
    if os.environ.get("TGL_PROSPECTIVE_UNIFORM_PERMUTATION", "0") == "1":
        from train.prospective_controls import uniform_row_permutations

        raw_context = os.environ.get("TGL_CONTROL_CONTEXT", "prospective-unspecified")
        try:
            parsed_context = json.loads(raw_context)
        except json.JSONDecodeError:
            parsed_context = {"context_id": raw_context}
        if not isinstance(parsed_context, dict):
            raise ValueError("TGL_CONTROL_CONTEXT must encode a JSON object")
        return uniform_row_permutations(
            node_ids,
            query_times,
            neighbor_ids,
            edge_ids,
            gaps,
            layer=int(layer),
            control_seed=int(seed),
            context=parsed_context,
        )

    node_ids=np.asarray(node_ids);query_times=np.asarray(query_times)
    neighbor_ids=np.asarray(neighbor_ids);edge_ids=np.asarray(edge_ids);gaps=np.asarray(gaps)
    if neighbor_ids.shape != edge_ids.shape or gaps.shape != neighbor_ids.shape:
        raise ValueError('Neighborhood ID, edge ID and gap shapes must agree')
    if len(node_ids)!=len(gaps) or len(query_times)!=len(gaps):raise ValueError('Missing query identities')
    out=np.broadcast_to(np.arange(gaps.shape[1]),gaps.shape).copy()
    for row in range(len(gaps)):
        positions=np.flatnonzero(neighbor_ids[row]>0)
        if len(positions)<2:continue
        canonical=sorted(positions,key=lambda j:(int(edge_ids[row,j]),int(neighbor_ids[row,j]),float(gaps[row,j])))
        events=[(int(edge_ids[row,j]),int(neighbor_ids[row,j]),float(gaps[row,j]).hex()) for j in canonical]
        identity=[int(seed),int(layer),int(node_ids[row]),float(query_times[row]).hex(),events]
        digest=hashlib.blake2b(json.dumps(identity,separators=(',',':')).encode(),digest_size=16).digest()
        rng=np.random.Generator(np.random.PCG64(int.from_bytes(digest,'little')))
        permutation=rng.permutation(len(canonical))
        original=np.asarray([float(gaps[row,j]) for j in canonical])
        if len(np.unique(original))>1 and np.array_equal(original,original[permutation]):
            # A random index permutation can be identity, or can move only tied
            # gaps. Stage C2 requires a numerically changed assignment whenever
            # the row actually contains distinguishable temporal weights.
            for shift in range(1,len(canonical)):
                candidate=np.roll(np.arange(len(canonical)),shift)
                if not np.array_equal(original,original[candidate]):
                    permutation=candidate
                    break
        out[row,canonical]=np.asarray(canonical)[permutation]
    return out

def gate_features(gaps, time_features, edge_features, neighbor_ids, *, family, sigma,
                  amplitude=1., strength=1., control='real', query_node_ids=None,
                  query_times=None, neighbor_edge_ids=None, layer=0,
                  control_seed=None):
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('sigma must be finite and positive')
    if not np.isfinite(amplitude) or not 0<=amplitude<=1:raise ValueError('amplitude must be in [0,1]')
    if not np.isfinite(strength) or not 0<=strength<=1:raise ValueError('strength must be in [0,1]')
    d=torch.as_tensor(gaps,dtype=time_features.dtype,device=time_features.device).abs()
    valid=torch.as_tensor(neighbor_ids,device=d.device)>0
    if d.shape!=time_features.shape[:2] or d.shape!=edge_features.shape[:2] or d.shape!=valid.shape:
        raise ValueError('Gaps, masks, time features and edge features have inconsistent shapes')
    if not torch.isfinite(d).all():raise ValueError('Non-finite gaps')
    if family=='none':w=torch.ones_like(d)
    elif family=='laplacian':w=torch.exp(-d/sigma)
    elif family=='rbf':w=torch.exp(-d.square()/sigma**2)
    else:raise ValueError(f'Unknown gate family: {family}')
    w=w*amplitude
    if control=='permute':
        if any(x is None for x in [query_node_ids,query_times,neighbor_edge_ids,control_seed]):
            raise ValueError('Permutation requires query identities, edge IDs and an independent control seed')
        order=stable_row_permutations(query_node_ids,query_times,neighbor_ids,neighbor_edge_ids,
                                      np.asarray(gaps),layer,control_seed)
        w=w.gather(1,torch.as_tensor(order,device=d.device))
    elif control=='row_mean':
        w=((w*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)).expand_as(w)
    elif control=='row_energy':
        energy=time_features.square().sum(-1)+edge_features.square().sum(-1)
        denominator=(energy*valid).sum(1,keepdim=True)
        numerator=(w.square()*energy*valid).sum(1,keepdim=True)
        c=torch.where(denominator>0,(numerator/denominator.clamp_min(torch.finfo(d.dtype).tiny)).sqrt(),torch.ones_like(denominator))
        # Match the forward norm without adding a normalization derivative.
        w=c.detach().expand_as(w)
    elif control!='real':raise ValueError(f'Unknown control: {control}')
    # ``amplitude`` is the historical multiplier a*psi and remains unchanged.
    # Stage C2 uses a separate identity-to-gate interpolation so strength=0 is
    # exactly vanilla and strength=1 is exactly the selected full gate.
    if strength == 0:
        w=torch.ones_like(w)
    elif strength != 1:
        w=1.+strength*(w-1.)
    # Padding is not part of the treatment or any matching statistic.
    w=torch.where(valid,w,torch.ones_like(w))
    return time_features*w.unsqueeze(-1),edge_features*w.unsqueeze(-1),w
