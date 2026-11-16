from .offload_experiment import strategy
from .policy import CONTEXT, MAX_REQUESTS, WIDTH, PARENTS
from .root_shadow import validate_top2

CHAIN_PARENTS = PARENTS
FORK_PARENTS = CHAIN_PARENTS + (0,)
FORK_DEPTHS = tuple(range(WIDTH)) + (1,)


def pack(proposals, contexts):
    if any('growth_plan' in p for p in proposals):
        if not all('growth_plan' in p for p in proposals):raise ValueError('GROWTH_PARTIAL_PLAN_BATCH')
        from .growth import pack as growth_pack, fork_gap_limit
        return growth_pack(proposals,contexts)
    if (not 1 <= len(proposals) <= MAX_REQUESTS or len(proposals) != len(contexts)
            or len({p['id'] for p in proposals}) != len(proposals)):
        raise ValueError('FORK5_BATCH_BINDING')
    out = []
    for p, c in zip(proposals, contexts):
        if (p['id'] != c['id'] or p['prefix_hash'] != c['prefix_hash']
                or len(p['nodes']) != WIDTH or type(c['remaining']) is not int
                or c['remaining'] < 0 or type(c['prefix_length']) is not int
                or c['prefix_length'] < 1 or type(c['ignore_eos']) is not bool
                or any(type(t) is not int or t < 0 for t in p['nodes'])):
            raise ValueError('FORK5_CONTEXT_BINDING')
        nodes = list(p['nodes'])
        if c['source'] == 'edge':
            meta = validate_top2(c['top2'], nodes[1])
            # Near the output/context boundary or natural EOS, preserve the
            # original chain policy. These are planned shapes, not failure fallbacks.
            # Budget follows the actual verification cohort, not HTTP C,
            # active requests or a post-forward failure fallback.
            limit=c.get('fork_gap')
            if limit is None:limit=fork_gap_limit()
            if (strategy() != 'k3' and len(proposals) == 1 and c['remaining'] >= WIDTH
                    and c['ignore_eos'] and c['prefix_length']+WIDTH <= CONTEXT
                    and (limit is None or meta['gap']<=limit)):
                nodes.append(meta['alternate'])
        elif c['source'] not in ('suffix', 'exact_prefix') or c['top2'] is not None:
            raise ValueError('FORK5_SOURCE_BINDING')
        out.append(dict(id=p['id'], prefix_hash=p['prefix_hash'], nodes=nodes))
    return out


def row_parents(nodes):
    if len(nodes) not in (WIDTH, WIDTH+1):
        raise ValueError('FORK5_NODE_COUNT')
    return FORK_PARENTS[:len(nodes)]
