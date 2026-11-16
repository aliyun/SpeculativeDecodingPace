import os

BATCH_MAX = 8
GRAPH_LIMIT = 192
TARGET_GRAPHS = 157
EDGE_GRAPHS = 96  # 72 existing captures + 3 plain tiers x B1..8.
ACTIONS = tuple((k, roots) for k in (3, 5, 7) for roots in (1, 2))


def enabled():
    value = os.environ.get('TRIAGE_BATCH_GROWTH', '0')
    if value not in ('0', '1'):
        raise ValueError('BATCH_GROWTH_MODE')
    # The production catalogue is selected by the single growth switch. Keep
    # the older multi-flag experiment gate for explicit historical runs.
    if os.environ.get('TRIAGE_ADAPTIVE_GROWTH') == '1' and os.environ.get('TRIAGE_GROWTH_EXPERIMENT') != '1':
        return True
    if value == '1' and (os.environ.get('TRIAGE_ADAPTIVE_GROWTH') != '1'
                         or os.environ.get('TRIAGE_GROWTH_EXPERIMENT') != '1'):
        raise ValueError('BATCH_GROWTH_REQUIRES_EXPLICIT_EXPERIMENT')
    return value == '1'


def stream_chain_layouts(ks=None, *, streaming=None):
    """Only the missing even single-root B2..8 shapes; legacy trees stay intact."""
    if ks is None:
        from .growth import KS
        ks = KS
    if streaming is None:
        streaming = os.environ.get('TRIAGE_STREAM_SELECTOR') == '1'
    return tuple((k, b) for k in ks if k not in (3, 5, 7)
                 for b in range(2, BATCH_MAX+1)) if streaming else ()


def target_graphs(ks, *, streaming=False):
    return TARGET_GRAPHS + 2*sum(k not in (3, 5, 7) for k in ks) + len(
        stream_chain_layouts(ks, streaming=streaming))


def layouts():
    """(K, actual B, side-bearing rows); real queries, including all tails."""
    return tuple((k, b, sides) for k in (3, 5, 7)
                 for b in range(1, BATCH_MAX + 1) for sides in range(b + 1))
