"""Stream a training subprocess into a notebook and stop it on interruption."""

import subprocess


def run_with_progress(command, *, emit=None):
    """Return only after the child exits; propagate failures and notebook interrupts.

    Python children should use ``-u`` so their lines reach the notebook immediately.
    The pilot runs in this single child; stopping it leaves the last atomic checkpoint
    available for a subsequent resume and prevents an orphaned training writer.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    try:
        for line in process.stdout:
            if emit is None:
                print(line, end='', flush=True)
            else:
                emit(line)
        result = subprocess.CompletedProcess(command, process.wait())
        result.check_returncode()
        return result
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
