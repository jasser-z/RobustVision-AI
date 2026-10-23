# Methodology

CIFAR-10 is the primary supported dataset. A future full evaluation must retain the official test set for final evaluation only and create validation data deterministically from the training set. Accuracy is correct predictions divided by evaluated examples. Accuracy drop is clean accuracy minus corrupted accuracy, expressed in percentage points when formatted as percentages. Macro metrics average class metrics with zero-division handled as zero. ECE bins maximum confidence and measures the weighted absolute gap between bin accuracy and confidence; it changes with binning.

Corruption severity is centralized in `robustvision.data.corruptions.SEVERITY`. Gaussian noise adds zero-mean normal pixel noise, blur applies Pillow Gaussian blur, brightness/contrast multiply their corresponding enhancement factor, JPEG round-trips at a quality level, and salt/pepper sets seeded pixels to 0 or 255. Tests verify seeded behavior and dimensions.

Inference must run in evaluation mode with gradients disabled. Timing must state device, warm-up policy, batch size, and repetitions. CUDA timing requires synchronization. Hardware, drivers, PyTorch versions, and nondeterministic kernels can affect exact reproducibility.
