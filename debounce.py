import time, threading, functools

def debounce(wait: float, *, leading=True, trailing=False):
    """
    Decorator that drops calls that arrive within `wait`
    seconds of the previous accepted call.

    leading  – execute the *first* call in the burst
    trailing – execute the *last* call when burst ends
    """
    def decorator(fn):
        last, timer, lock = 0.0, None, threading.Lock()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal last, timer
            now = time.time()

            def call():
                nonlocal last, timer
                last = time.time()
                timer = None
                fn(*args, **kwargs)

            with lock:
                if now - last >= wait:
                    # we’re outside the quiet period
                    if leading:
                        call()
                    else:
                        last = now
                elif trailing:
                    # inside quiet period – schedule trailing call
                    if timer:
                        timer.cancel()
                    timer = threading.Timer(wait - (now - last), call)
                    timer.start()
        return wrapper
    return decorator
