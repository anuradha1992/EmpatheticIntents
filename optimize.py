import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps):
        super(CustomSchedule, self).__init__()

        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = step / self.warmup_steps
        arg2 = (self.total_steps - step) / (self.total_steps - self.warmup_steps)
        return self.peak_lr * tf.math.minimum(arg1, arg2)
