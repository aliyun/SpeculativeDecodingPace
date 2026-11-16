from collections import Counter
import json
import math
import os
from pathlib import Path
import struct
import time
from .policy import MAX_REQUESTS


def validate_top2(value, primary):
    if (type(value) is not dict or set(value) != {'primary', 'alternate', 'gap'}
            or type(primary) is not int or primary < 0
            or value['primary'] != primary or type(value['primary']) is not int
            or type(value['alternate']) is not int or value['alternate'] < 0
            or value['alternate'] == primary
            or type(value['gap']) not in (int, float) or not math.isfinite(value['gap'])
            or value['gap'] < 0):
        raise ValueError('ROOT_TOP2_INVALID')
    return dict(primary=primary, alternate=value['alternate'], gap=float(value['gap']))


def decode_top2(primary, alternate, gap_bits):
    # GPU sends one int64 packed result; float32 gap bits retain exact values.
    if type(gap_bits) is not int or not 0 <= gap_bits <= 0xffffffff:
        raise ValueError('ROOT_TOP2_GAP_BITS')
    gap = struct.unpack('!f', struct.pack('!I', gap_bits))[0]
    return validate_top2(dict(primary=primary, alternate=alternate, gap=gap), primary)


class RootTop2Shadow:
    def __init__(self, limit=20000):
        if type(limit) is not int or limit < 1:
            raise ValueError('ROOT_SHADOW_LIMIT_CONFIG')
        self.limit, self.records = limit, []
        self.counts = Counter()
        self.flushes = self.bytes = self.write_ns = 0

    def observe(self, proposals, verified, context):
        """Bind already-completed chain labels; never select from target results."""
        if (not 1 <= len(proposals) <= MAX_REQUESTS or len({p['id'] for p in proposals}) != len(proposals)
                or len(proposals) != len(verified) or len(proposals) != len(context)):
            raise ValueError('ROOT_SHADOW_BATCH_BINDING')
        if len(self.records) + len(proposals) > self.limit:
            raise ValueError('ROOT_SHADOW_LIMIT')
        pending = []
        for p, v, c in zip(proposals, verified, context):
            if (p['id'] != v.get('id') or p['id'] != c['id']
                    or p['prefix_hash'] != v.get('prefix_hash')
                    or p['prefix_hash'] != c['prefix_hash']
                    or len(p['nodes']) != 4 or len(v.get('target_argmax', [])) != 4):
                raise ValueError('ROOT_SHADOW_PREFIX_TARGET_BINDING')
            path, emitted = v['path'], v['emitted']
            if (any(type(t) is not int or t < 0 for t in p['nodes'])
                    or type(c['remaining']) is not int or c['remaining'] < 0
                    or type(c['ignore_eos']) is not bool
                    or type(c['delivered']) is not int or not 0 <= c['delivered'] <= len(emitted)):
                raise ValueError('ROOT_SHADOW_CONTEXT_TYPES')
            if (path not in ([0], [0, 1], [0, 1, 2], [0, 1, 2, 3])
                    or len(path) != len(emitted)
                    or emitted[:-1] != [p['nodes'][i] for i in path[1:]]
                    or any(type(t) is not int or t < 0 for t in v['target_argmax'])
                    or emitted[-1] != v['target_argmax'][path[-1]]):
                raise ValueError('ROOT_SHADOW_NOT_CHAIN_LABELS')
            accepted = len(path)-1
            expected = 0
            while expected < 3 and p['nodes'][expected+1] == v['target_argmax'][expected]:
                expected += 1
            if accepted != expected:
                raise ValueError('ROOT_SHADOW_ACCEPTANCE_BINDING')
            if (v['target_argmax'][0] == p['nodes'][1]) != (accepted >= 1):
                raise ValueError('ROOT_SHADOW_ACCEPTANCE_BINDING')
            meta = c['top2']
            record = dict(id=p['id'], prefix_hash=p['prefix_hash'], source=c['source'],
                          primary=p['nodes'][1], accepted_draft_tokens=accepted,
                          raw_emitted_tokens=len(emitted), delivered_tokens=c['delivered'])
            if meta is None:
                if c['source'] not in ('suffix', 'exact_prefix'):
                    raise ValueError('ROOT_SHADOW_MISSING_EDGE_TOP2')
                record.update(eligible=False, excluded=c['source'], top2=None)
            else:
                if c['source'] != 'edge':
                    raise ValueError('ROOT_SHADOW_WRONG_SOURCE')
                meta = validate_top2(meta, p['nodes'][1])
                eligible = c['remaining'] >= 4 and c['ignore_eos'] and c['delivered'] == len(emitted)
                rescued = v['target_argmax'][0] == meta['alternate']
                if rescued and accepted:
                    raise ValueError('ROOT_SHADOW_RESCUE_ACCEPTED_PRIMARY')
                record.update(top2=meta, target_root=v['target_argmax'][0], eligible=eligible,
                              excluded=None if eligible else 'output_boundary_eos_or_cancel',
                              rescued_root=rescued, lost_third_level=accepted == 3,
                              same_prefix_fork_delta=(int(rescued)-int(accepted == 3)) if eligible else None)
            pending.append(record)
        # Publish only after validation of the whole label batch.
        self.records.extend(pending)
        for r in pending:
            self.counts['observed'] += 1
            self.counts['eligible' if r['eligible'] else 'excluded'] += 1
            if r['eligible']:
                self.counts['rescued_root'] += int(r['rescued_root'])
                self.counts['lost_third_level'] += int(r['lost_third_level'])
                self.counts['same_prefix_delta_sum'] += r['same_prefix_fork_delta']

    @staticmethod
    def require_drained(active, waiting, admitting, failure, proposals, cloud, edge):
        if (active or waiting or admitting or failure
                or any(proposals.get(k, 0) for k in
                       ('live_entries', 'inflight', 'live_requests', 'cleanup_pending', 'root_top2_pending'))):
            raise ValueError('ROOT_SHADOW_NOT_LOGICALLY_DRAINED')
        for worker in (cloud, edge):
            shadow = worker.get('root_top2_shadow') or {}
            if (worker.get('requests') != 0 or shadow.get('enabled') is not True or shadow.get('failed', False)
                    or type(shadow.get('initial_free_blocks')) is not int
                    or worker.get('free_blocks') != shadow['initial_free_blocks']):
                raise ValueError('ROOT_SHADOW_NOT_PHYSICALLY_DRAINED')

    def export(self, directory):
        start = time.monotonic_ns()
        data = ''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in self.records).encode()
        path = Path(directory)/('ROOT_TOP2_SHADOW.%06d.jsonl' % (self.flushes+1))
        fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        records = len(self.records)
        self.records.clear()
        self.flushes += 1;self.bytes += len(data);self.write_ns += time.monotonic_ns()-start
        return dict(file=path.name, records=records, counts=dict(self.counts),
                    flushes=self.flushes, bytes=self.bytes, write_ns=self.write_ns,
                    scope='SHADOW_SAME_PREFIX_LABELS_NOT_FORK_NATIVE_OR_SPEEDUP')
