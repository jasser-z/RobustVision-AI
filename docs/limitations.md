# Limitations

This local research application is not a production security boundary. It has no authentication, rate limiting, remote artifact access controls, or asynchronous benchmark job queue. The dashboard supports saved-checkpoint selection and Grad-CAM for inspection, but Grad-CAM is an explanation aid rather than a causal account of model behavior.

The full benchmark artifacts evaluate the official CIFAR-10 test split repeatedly under defined corruptions. They are useful descriptive measurements, but repeated inspection of this test set can bias future design decisions. The original aggregate robustness CSVs contain no logits or probabilities; separate clean and corrupted calibration prediction artifacts were therefore generated from the saved checkpoints. The ResNet result uses CIFAR normalization despite pretrained initialization; this preprocessing choice should be compared in a controlled rerun.
