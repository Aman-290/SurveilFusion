import queue

class ReplaceQueue(queue.Queue):
    def put(self, item, block=True, timeout=None):
        with self.mutex:
            if self.full():
                # If the queue is full, remove the oldest item
                _ = self.get_nowait()
            super().put(item, block, timeout)

# Create a ReplaceQueue with a maximum size of 1
my_replace_queue = ReplaceQueue(maxsize=1)

# Enqueue elements
my_replace_queue.put(10)
print("Enqueued 10")

# Enqueue another element (this will replace the existing element)
my_replace_queue.put(20)
print("Enqueued 20")

# Dequeue elements
item = my_replace_queue.get()
print("Dequeued item:", item)
