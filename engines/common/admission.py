import asyncio
from collections import deque
from itertools import islice


class AdmissionQueue:
    """Single-event-loop queue whose in-flight admission prefix stays reserved."""
    def __init__(self, maxsize=512):
        self.maxsize=maxsize;self.items=deque()

    def put_nowait(self, row):
        if len(self.items)>=self.maxsize:raise asyncio.QueueFull
        self.items.append(row)

    def get_nowait(self):
        if not self.items:raise asyncio.QueueEmpty
        return self.items.popleft()

    def qsize(self):return len(self.items)
    def empty(self):return not self.items
    def peek(self, count):return list(islice(self.items, count))

    def remove_cancelled(self, cancelled):
        removed=[r['id'] for r in self.items if r['id'] in cancelled]
        self.items=deque(r for r in self.items if r['id'] not in cancelled)
        return removed
