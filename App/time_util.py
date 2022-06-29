import time
import threading

class ElapsedTimeThread(threading.Thread):
    """"Stoppable thread that prints the time elapsed"""
    def __init__(self):
        super(ElapsedTimeThread, self).__init__()
        self._stop_event = threading.Event()
        self.time = None

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        thread_start = time.time()
        while not self.stopped():
            #print("\rElapsed Time {:.1f} seconds".format(time.time()-thread_start), end="")
            self.time = "\r{:.1f} seconds".format(time.time()-thread_start)
            #return  time_T
            #include a delay here so the thread doesn't uselessly thrash the CPU
            time.sleep(0.01)

if __name__ == "__main__":
    #start = time.time()
    thread = ElapsedTimeThread()
    thread.start()
    # do something
    #print(thread.time)
    #time.sleep(5)
    #print(thread.time)

    # something is finished so stop the thread
    thread.stop()
    thread.join()
    #print() # empty print() to output a newline
    #print("Finished in {:.1f} seconds".format(time.time()-start))