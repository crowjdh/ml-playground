import signal

from utils.singleton import Singleton


class CleanupHelper(metaclass=Singleton):
    def __init__(self):
        self.cleanup_requested = False
        self.cleanup = lambda *args, **kwargs: None
        signal.signal(signal.SIGINT, self.cleanup_and_exit)
        signal.signal(signal.SIGTERM, self.cleanup_and_exit)

    def cleanup_and_exit(self, _, __):
        self.cleanup_requested = True
        print("Clean up requested. Clean up will be performed soon.")

    def try_to_cleanup(self, *args, **kwargs):
        has_cleaned_up = False
        if self.cleanup_requested:
            print("Cleaning up...")
            self.cleanup(*args, **kwargs)
            print("Cleanup done")
            has_cleaned_up = True

        return has_cleaned_up
