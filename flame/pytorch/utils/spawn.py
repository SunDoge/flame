from typing import List
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import _wrap, ProcessException, ProcessExitedException, ProcessRaisedException
import signal
import multiprocessing
import warnings


class ProcessContext:
    def __init__(self, processes: List[mp.Process], error_queues):
        self.error_queues = error_queues
        self.processes = processes
        self.sentinels = {
            process.sentinel: index
            for index, process in enumerate(processes)
        }

    def pids(self):
        return [int(process.pid) for process in self.processes]

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True

        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break

        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0

        # Assume failure. Terminate processes that are still alive.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                # process.kill()
            process.join()

        # There won't be an error on the queue if the process crashed.
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException(
                    "process %d terminated with signal %s" %
                    (error_index, name),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d" %
                    (error_index, exitcode),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode
                )

        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
        raise ProcessRaisedException(msg, error_index, failed_process.pid)



# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like Ipython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Args:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (string): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    if start_method != 'spawn':
        msg = ('This method only supports start_method=spawn (got: %s).\n'
               'To use a different start_method use:\n\t\t'
               ' torch.multiprocessing.start_process(...)' % start_method)
        warnings.warn(msg)
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
