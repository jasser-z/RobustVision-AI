import argparse

from torchvision.datasets import CIFAR10

from robustvision.data.cifar_archive import install_local_archive

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
a = p.parse_args()
installed = install_local_archive(a.data_dir)
if installed:
    # `download=False` performs torchvision's expected CIFAR layout/integrity check and
    # guarantees this local-archive path makes no network request.
    CIFAR10(a.data_dir, download=False)
    print(f"CIFAR-10 verified from local archive at {installed}")
else:
    # Preserve torchvision's standard HTTPS implementation and its TLS verification.
    CIFAR10(a.data_dir, download=True)
    print(f"CIFAR-10 downloaded and cached at {a.data_dir}")
