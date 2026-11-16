from .policy import PARENTS


def validate(parents):
    if not parents or parents[0] != -1:
        raise ValueError('root must be -1')
    for node, parent in enumerate(parents[1:], 1):
        if not 0 <= parent < node:
            raise ValueError('parents must be topologically ordered')


def depths(parents):
    validate(parents)
    out = [0]
    for parent in parents[1:]:
        out.append(out[parent] + 1)
    return out


def path_to(parents, node):
    validate(parents)
    if not 0 <= node < len(parents):
        raise ValueError('node outside tree')
    path = []
    while node >= 0:
        path.append(node)
        node = parents[node]
    return path[::-1]


def ancestor_mask(parents):
    validate(parents)
    return [[j in path_to(parents, i) for j in range(len(parents))]
            for i in range(len(parents))]


def accept(parents, tokens, target_argmax):
    validate(parents)
    if len(parents) != len(tokens) or len(tokens) != len(target_argmax):
        raise ValueError('tree array lengths differ')
    node, path = 0, [0]
    while True:
        matches = [i for i in range(node + 1, len(parents))
                   if parents[i] == node and tokens[i] == target_argmax[node]]
        if not matches:
            return path, [tokens[i] for i in path[1:]] + [target_argmax[node]]
        node = matches[0]
        path.append(node)
