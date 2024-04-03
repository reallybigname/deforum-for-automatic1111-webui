import time
import threading
import queue
from modules.shared import opts, state
from .general_utils import debug_print

class IntervalPreviewDisplayer:
    def __init__(self, frames_per_cycle=1):
        self.image_queue = queue.Queue()
        self.timer = None
        self.cycle_start_time = None
        self.polling_interval = 0.5
        self.polling_interval_min = 0.01
        self.frames_per_cycle = frames_per_cycle
        self.cycle_duration = self.frames_per_cycle * self.polling_interval
        self.frame_number = -1
        self.running = True
        self.paused = False
        self.buffer_size = frames_per_cycle
        self.start_timer()

    def add_to_display_queue(self, image):
        self.frame_number += 1

        # if cadence 1, just pass image through for display with no queue
        if self.frames_per_cycle == 1:
            self.display_image(image)
        else:
            # start cycle on frame 0
            if self.frame_number == 0:
                self.display_image(image)
                self.cycle_start_time = time.time()
            else:
                # add image to queue immediately
                self.image_queue.put(image)

                # store up first cadence cycle worth of images in queue before beginning to release, to have a buffer
                is_end_frame = False if self.frame_number == 0 else (self.frame_number % self.frames_per_cycle == 0)
                if is_end_frame and self.frame_number >= self.frames_per_cycle:
                    self.cycle_duration = time.time() - self.cycle_start_time
                    self.cycle_start_time = time.time()
                    self.adjust_polling_interval()

    def adjust_polling_interval(self):
        frames = self.frames_per_cycle + self.image_queue.qsize()
        if frames > self.buffer_size:
            frames -= self.buffer_size
        self.polling_interval = max(self.cycle_duration / frames, self.polling_interval_min)

    def display_image(self, image):
        opts.live_previews_enable = True
        state.assign_current_image(image)

    def timer_callback(self):
        while self.running:
            if not self.paused and not self.image_queue.empty():
                image = self.image_queue.get()
                self.display_image(image)
                self.report_status()
            time.sleep(self.polling_interval)

    def empty_queue(self, keep_n_items):
        while self.image_queue.qsize() > keep_n_items:
            excess_image = self.image_queue.get()
            self.display_image(excess_image)

    def start_timer(self):
        self.timer = threading.Thread(target=self.timer_callback)
        self.timer.start()
        
    def stop(self):
        self.running = False  # Signal the thread to stop

    def pause(self):
        self.paused = True  # Pause the timer

    def resume(self):
        self.paused = False  # Resume the timer
 
    def report_status(self):
        console_msg = f"DISPLAYER Images in queue: {self.image_queue.qsize()} | Current cycle duration/cadence {self.cycle_duration:.2f}/{self.frames_per_cycle} = {self.polling_interval:.2f} sec polling interval"
        debug_print(console_msg)
        
    def before_generation(self):
        opts.live_previews_enable = False
        self.pause()

    def after_generation(self):
        opts.live_previews_enable = True
        self.resume()
