import threading
#import queue
import Queue as queue
from subprocess import call

class Runner:
    def __init__(self, resources, task_args, task_fn):
        self.resources = resources
        self.task_fn = task_fn
        self.task_args = task_args
        self.work_queue = queue.Queue()
        self.threads = []

        for targ in task_args:
            self.work_queue.put(targ)

        def worker(resource):
            while True:
                name, gpu = resource
                task_args = self.work_queue.get()
                if task_args is None:
                    return
                task_args['resource'] = resource
                cmd_string = task_fn(**task_args)
                print('Starting', resource, cmd_string)
                if name == "localhost":
                    cmd_string = cmd_string.split(";")[-1]
                    call(cmd_string.split())
                else:
                    call(['ssh', name, cmd_string])
                print('Done', resource, cmd_string)

        for r in self.resources:
            thread = threading.Thread(target=worker, args=(r, ))
            self.threads.append(thread)
            self.work_queue.put(None)
            thread.start()

def test_runner():
    resources = [
                    ('ravi@bodega.graphics.cs.cmu.edu', 0),
                    ('ravi@bodega.graphics.cs.cmu.edu', 0),
                    ('ravi@bodega.graphics.cs.cmu.edu', 0),
                    ('ravi@bodega.graphics.cs.cmu.edu', 0),
                ]

    task_args = []
    num_tasks = 10

    for i in range(num_tasks):
        args = {}
        args['task_num'] = i
        task_args.append(args)

    def task_fn(**kwargs):
        task_num = kwargs['task_num']
        cmd = 'echo %d' %(task_num)
        return cmd

    Runner(resources, task_args, task_fn)

if __name__ == "__main__":
    test_runner()
