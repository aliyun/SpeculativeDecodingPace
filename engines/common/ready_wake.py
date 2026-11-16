import asyncio
import threading


class CompletionWake:
    def __init__(self):
        self.loop=asyncio.get_running_loop()
        self.event=asyncio.Event()
        self.guard=threading.Lock()
        self.closed=False
        self.error=None

    def notify(self, completed):
        # Future callbacks can run on a worker or inline on the controller.
        # Never inspect/consume the result, perform work, or wait for the owner.
        with self.guard:
            if self.closed:return
            try:self.loop.call_soon_threadsafe(self.event.set)
            except Exception as exc:self.error=exc

    def check(self):
        if self.error is not None:
            raise RuntimeError('GROWTH_READY_WAKE_FAILED') from self.error

    def clear(self):
        self.check()
        self.event.clear()

    async def wait(self, timeout):
        self.check()
        try:await asyncio.wait_for(self.event.wait(),timeout)
        except TimeoutError:pass
        self.check()

    def close(self):
        with self.guard:self.closed=True
