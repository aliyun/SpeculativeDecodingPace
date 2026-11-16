from concurrent.futures import ThreadPoolExecutor
import threading
import time

WORKERS, CAPACITY = 32, 64


class _ReaderJob:
    """The join/is_alive surface used by stream handles, without owning a thread."""
    def __init__(self, done):
        self.done = done
    def join(self, timeout=None):
        self.done.wait(timeout)
    def is_alive(self):
        return not self.done.is_set()


class StreamReaders:
    def __init__(self, *, workers=WORKERS, capacity=CAPACITY):
        if (type(workers) is not int or not 1 <= workers <= WORKERS
                or type(capacity) is not int or not workers <= capacity <= CAPACITY):
            raise ValueError('invalid bounded stream reader pool')
        self.workers, self.capacity = workers, capacity
        self.lock, self.slots = threading.RLock(), threading.BoundedSemaphore(capacity)
        self.executor, self.jobs, self.worker_tids = None, {}, set()
        self.prepared = self.closed = self.finished = False
        self.failure, self.setup = None, None

    def prepare(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('stream readers closed')
            if self.prepared:
                return self.snapshot()
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='edge-reuse')
            gate = threading.Barrier(self.workers+1)
            def warm():
                gate.wait(timeout=10)
                return threading.get_native_id()
            try:
                jobs = [self.executor.submit(warm) for _ in range(self.workers)]
                gate.wait(timeout=10)
                self.worker_tids = {job.result(timeout=10) for job in jobs}
                if len(self.worker_tids) != self.workers:
                    raise RuntimeError('stream reader prewarm incomplete')
            except BaseException:
                self.closed = True
                self.executor.shutdown(wait=True, cancel_futures=False)
                self.finished = True
                raise
            self.setup = dict(start_ns=start, end_ns=time.monotonic_ns(),
                              caller_cpu_ns=time.thread_time_ns()-cpu)
            self.prepared = True
            return self.snapshot()

    def _complete(self, future, done):
        with self.lock:
            try:
                error = future.exception()
                if error is not None and self.failure is None:
                    self.failure = error
            except BaseException as exc:
                if self.failure is None:
                    self.failure = exc
            finally:
                self.jobs.pop(future)
                self.slots.release()
                done.set()

    def submit(self, run, cancel):
        with self.lock:
            if not self.prepared or self.closed or self.failure is not None:
                raise RuntimeError('stream readers not prepared, closed or failed') from self.failure
            if not self.slots.acquire(blocking=False):
                raise RuntimeError('stream reader capacity exceeded')
            done = threading.Event()
            try:
                future = self.executor.submit(run)
            except BaseException:
                self.slots.release()
                raise
            self.jobs[future] = cancel
            # A completed future calls back synchronously here; this lock must be reentrant.
            future.add_done_callback(lambda f: self._complete(f, done))
            return _ReaderJob(done)

    def snapshot(self):
        with self.lock:
            return dict(prepared=self.prepared, closed=self.closed, workers=self.workers,
                        capacity=self.capacity, active_jobs=len(self.jobs),
                        worker_tids=sorted(self.worker_tids), setup=self.setup,
                        failure=repr(self.failure) if self.failure is not None else None)

    def close(self):
        with self.lock:
            if self.finished:
                return
            if threading.get_native_id() in self.worker_tids:
                raise RuntimeError('cannot join stream readers from their worker')
            self.closed = True
            cancel = list(self.jobs.values())
            executor = self.executor
        error = None
        for stop in cancel:
            try:
                stop()
            except BaseException as exc:
                if error is None:
                    error = exc
        # Cancel the paths, not the futures: queued jobs must release their leases
        # and deliver their terminal callback, even when they never send a request.
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        with self.lock:
            self.finished = True
            if self.jobs:
                raise RuntimeError('stream readers did not drain')
            failure = self.failure if self.failure is not None else error
        if failure is not None:
            raise RuntimeError('stream reader cleanup failed') from failure
