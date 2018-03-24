#!/usr/bin/env python3
# mnists
# Copyright (C) 2018 Connor Olding
# Distributed under terms of the MIT license.

__version__ = "0.2.1"

# NOTE: npz functionality is incomplete.

import array
import gzip
import hashlib
import numpy as np
import os
import os.path
import struct
import sys
from urllib.request import urlretrieve

from .hashes import hashes


home = os.path.expanduser("~")
output_directory = os.path.join(home, ".mnist")
webhost = "https://eaguru.guru/mnist/"


def lament(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def make_meta(npz, train_part="train", test_part="t10k", prefix=""):
    images_suffix = "-images-idx3-ubyte.gz"
    labels_suffix = "-labels-idx1-ubyte.gz"
    return (prefix, npz,
            train_part + images_suffix,
            train_part + labels_suffix,
            test_part + images_suffix,
            test_part + labels_suffix)


def make_emnist_meta(npz, name):
    return make_meta(npz, name + "-train", name + "-test", prefix="emnist")


metadata = dict(
    emnist_balanced=make_emnist_meta("emnist_balanced.npz", "emnist-balanced"),
    emnist_byclass=make_emnist_meta("emnist_byclass.npz", "emnist-byclass"),
    emnist_bymerge=make_emnist_meta("emnist_bymerge.npz", "emnist-bymerge"),
    emnist_digits=make_emnist_meta("emnist_digits.npz", "emnist-digits"),
    emnist_letters=make_emnist_meta("emnist_letters.npz", "emnist-letters"),
    emnist_mnist=make_emnist_meta("emnist_mnist.npz", "emnist-mnist"),
    fashion_mnist=make_meta("fashion_mnist.npz", prefix="fashion-mnist"),
    mnist=make_meta("mnist.npz", prefix="mnist"),
)


# correct the path separators for the user's system.
hashes = dict((os.path.join(*k.split("/")), v) for k, v in hashes.items())


def construct_path(name):
    return os.path.join(output_directory, name)


def make_directories():
    directories = [output_directory]
    directories += [construct_path(meta[0]) for meta in metadata.values()]
    for directory in directories:
        if not os.path.isdir(directory):
            os.mkdir(directory)


def download(name):
    url_name = "/".join(os.path.split(name))
    path = construct_path(name)
    already_exists = os.path.isfile(path)
    if not already_exists:
        url = webhost + url_name
        try:
            urlretrieve(url, path)
        except Exception:
            lament(f"Failed to download {url} to {path}")
            raise
    return already_exists


def validate(name):
    if name not in hashes.keys():
        raise Exception(f"Unknown mnist dataset: {name}")

    with open(construct_path(name), "rb") as f:
        data = f.read()

    known_hash = hashes[name]
    hash = hashlib.sha256(data).hexdigest()

    if hash != known_hash:
        raise Exception(f"""Failed to validate dataset: {name}
Hash mismatch: {hash} should be {known_hash}
Please check your local file for tampering or corruption.""")


def onehot(ind):
    unique = np.unique(ind)
    hot = np.zeros((len(ind), len(unique)), dtype=np.int8)
    offsets = np.arange(len(ind)) * len(unique)
    hot.flat[offsets + ind.flat] = 1
    return hot


def read_array(f):
    return np.array(array.array("B", f.read()), dtype=np.uint8)


def open_maybe_gzip(path):
    opener = gzip.open if path.endswith(".gz") else open
    return opener(path, "rb")


def load(images_path, labels_path):
    # load labels first so we can determine how many images there are.
    with open_maybe_gzip(labels_path) as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = read_array(f)

    with open_maybe_gzip(images_path) as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = read_array(f).reshape(len(labels), 1, rows, cols)

    return images, labels


def squeeze(labels):
    unique = np.unique(labels)
    new_values = np.arange(len(unique))
    if np.all(new_values == unique):
        return labels

    relabelled = np.zeros_like(labels)
    for i in new_values:
        relabelled[labels == unique[i]] = i
    return relabelled


def batch_flatten(x):
    return x.reshape(len(x), -1)


def prepare(dataset="mnist", return_floats=True, return_onehot=True,
            flatten=False, squeeze_labels=True, check_integrity=True):
    if dataset not in metadata.keys():
        raise Exception(f"Unknown dataset: {dataset}")

    meta = metadata[dataset]
    prefix, names = meta[0], meta[1:]
    names = [os.path.join(prefix, name) for name in names]
    npz, train_images, train_labels, test_images, test_labels = names
    images_and_labels = names[1:]

    make_directories()

    existing = [os.path.isfile(construct_path(name)) for name in names]
    npz_existing, gz_existing = existing[0], existing[1:]

    for name in images_and_labels:
        download(name)
        if check_integrity:
            validate(name)

    if npz_existing:
        with np.load(construct_path(npz)):
            train_images_data, train_labels_data, \
                test_images_data, test_labels_data = \
                f["train_images"], f["train_labels"], \
                f["test_images"], f["test_labels"]
    else:
        train_images_data, train_labels_data = load(
            construct_path(train_images), construct_path(train_labels))
        test_images_data, test_labels_data = load(
            construct_path(test_images), construct_path(test_labels))

    # correct the orientation of emnist images.
    if prefix == "emnist":
        train_images_data = train_images_data.transpose(0, 1, 3, 2)
        test_images_data = test_images_data.transpose(0, 1, 3, 2)

    if return_floats:  # TODO: better name.
        train_images_data = train_images_data.astype(np.float32) / 255
        test_images_data = test_images_data.astype(np.float32) / 255

    # emnist_letters uses labels indexed from 1 instead of the usual 0.
    # the onehot function assumes labels are contiguous from 0.
    if squeeze_labels or return_onehot:
        train_labels_data = squeeze(train_labels_data)
        test_labels_data = squeeze(test_labels_data)

    if return_onehot:
        train_labels_data = onehot(train_labels_data)
        test_labels_data = onehot(test_labels_data)

    if flatten:
        train_images_data = batch_flatten(train_images_data)
        test_images_data = batch_flatten(test_images_data)

    return train_images_data, train_labels_data, \
        test_images_data, test_labels_data

    # kanpeki!
