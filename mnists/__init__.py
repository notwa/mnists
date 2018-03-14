#!/usr/bin/env python3
# mnists
# Copyright (C) 2018 Connor Olding
# Distributed under terms of the MIT license.

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
    emnist_balanced = make_emnist_meta("emnist_balanced.npz", "emnist-balanced"),
    emnist_byclass = make_emnist_meta("emnist_byclass.npz", "emnist-byclass"),
    emnist_bymerge = make_emnist_meta("emnist_bymerge.npz", "emnist-bymerge"),
    emnist_digits = make_emnist_meta("emnist_digits.npz", "emnist-digits"),
    emnist_letters = make_emnist_meta("emnist_letters.npz", "emnist-letters"),
    emnist_mnist = make_emnist_meta("emnist_mnist.npz", "emnist-mnist"),
    fashion_mnist = make_meta("fashion_mnist.npz", prefix="fashion-mnist"),
    mnist = make_meta("mnist.npz", prefix="mnist"),
)


hash_data = """
emnist/emnist-balanced-test-images-idx3-ubyte.gz 117fc12b015e37ea98bfb2cbba3f77c3782c056cdbd219b87abfc8661d7135db
emnist/emnist-balanced-test-labels-idx1-ubyte.gz a67cac948379b67270b9a4d0a01c8ec661598accf2f2b1964340b7c83cb175cd
emnist/emnist-balanced-train-images-idx3-ubyte.gz 74295e75e4e9fa2c102147460ee7eb763653d5132d883a84fb88932bfb8d2db4
emnist/emnist-balanced-train-labels-idx1-ubyte.gz b46d41e5156a0a8aaaa4d18979fdd1588850fa8571018acffd5d93b325a7d6ed
emnist/emnist-byclass-test-images-idx3-ubyte.gz 5ec835bd5d5b1a66378a132e727fef20fe994b3cc666899befcca5ac766f50cd
emnist/emnist-byclass-test-labels-idx1-ubyte.gz 501e305eeeacb093b162592d80e8a2632040872360fa6d85cfcb2200d91ca92a
emnist/emnist-byclass-train-images-idx3-ubyte.gz 9e21987f03a6eb8b870187e785883ddad96947299587908f1a7ec2ca0f4be5f0
emnist/emnist-byclass-train-labels-idx1-ubyte.gz f7ef403d48dc18a89dae0d83d11ee69cecb6e1c12c3a961d03712e495c02a8fa
emnist/emnist-bymerge-test-images-idx3-ubyte.gz 5e5815ec1b522089134435f7c55e75a00960b58414793593ffff23efc0821683
emnist/emnist-bymerge-test-labels-idx1-ubyte.gz 8bda1516255683ba9071e141947ca0c91a911176065cdd6259b0b4d8dc5b5f35
emnist/emnist-bymerge-train-images-idx3-ubyte.gz 39bf8c68037dad40db35348895f7e4e056e8c0a005ab78869c9fbc2c96e947be
emnist/emnist-bymerge-train-labels-idx1-ubyte.gz 2c47293b9e62d6abf6ad05fbf67e40baa32adfd380b8d7b44dcc35534b64e1f1
emnist/emnist-digits-test-images-idx3-ubyte.gz 20ebe43509264f7639d37c835a3f0b009e90b29e00a099a03101abe1644a3f63
emnist/emnist-digits-test-labels-idx1-ubyte.gz bad1834c45d5988e270b99d43b13edc3fd898f658ffbc2caf2846ef27f7b05b0
emnist/emnist-digits-train-images-idx3-ubyte.gz f7c95004d14d81af89522e67d8f0c781dfb3bd544181c81bfe1eb6b41c35b726
emnist/emnist-digits-train-labels-idx1-ubyte.gz 6638a6ff5fe2eefd9cca2471995b46c82986b8e4b2a70d2f5ce8e05d7edb319e
emnist/emnist-letters-test-images-idx3-ubyte.gz 50f0e93d75b99b463e9760b8345866f22587fa2c2fba472730a5ba09c693412d
emnist/emnist-letters-test-labels-idx1-ubyte.gz 924955c31fe7fd809b7303b5c0141282955f4b8390c74cdfd5d80435e0f4b4cb
emnist/emnist-letters-train-images-idx3-ubyte.gz 6288f418a917e007bf5fddb823a2dc7a5c6a35031401b93292a539cb12bd881f
emnist/emnist-letters-train-labels-idx1-ubyte.gz 9759af91ea6bccdf07a6040fd085861f2ec88e37f7c3f5760f6feb57106fbe5c
emnist/emnist-mnist-test-images-idx3-ubyte.gz b25640fa9e356c618132f5df347c95ef5bae6d9c2596be1de40f7e0ae1c3eac6
emnist/emnist-mnist-test-labels-idx1-ubyte.gz bcc5385d38f083dba046f5f756e99b32f5ea7fad122c6243c71c2239873065ca
emnist/emnist-mnist-train-images-idx3-ubyte.gz 5f9942f441031c0f1f2e4d162059fd6e19ea808b34c328b6d16cab0ed24c78f2
emnist/emnist-mnist-train-labels-idx1-ubyte.gz 00c6d3c2342fdffb1711e0b5656cac4ce5862d2826399d383a9bac07d612e9f8
fashion-mnist/t10k-images-idx3-ubyte.gz 78dcbbfb5a27efaf1b6c1d616caca68560be8766c5bffcb2791df9273a534229
fashion-mnist/t10k-labels-idx1-ubyte.gz 42bd18137a62d5998cdeae52bf3a0676ac6b706f5cf8439b47bb5b151ae3dccf
fashion-mnist/train-images-idx3-ubyte.gz 08dc20ab1689590a0bcd66e874647b6ee8464e2d501b5a3f1f78831db19a3fdc
fashion-mnist/train-labels-idx1-ubyte.gz b0197879cbda89f3dc7b894f9fd52b858e68ea4182b6947c9d8c2b67e5f18dcc
mnist/t10k-images-idx3-ubyte.gz 8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6
mnist/t10k-labels-idx1-ubyte.gz f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6
mnist/train-images-idx3-ubyte.gz 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609
mnist/train-labels-idx1-ubyte.gz 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c
"""

hashes = dict(hash.split(" ") for hash in hash_data.strip().split("\n"))

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
        except:
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
                  flatten=False, squeeze_layers=True):
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
        train_images_data = train_images_data.transpose(0,1,3,2)
        test_images_data = test_images_data.transpose(0,1,3,2)

    if return_floats:  # TODO: better name.
        train_images_data = train_images_data.astype(np.float32) / 255
        test_images_data = test_images_data.astype(np.float32) / 255

    # emnist_letters uses labels indexed from 1 instead of the usual 0.
    # the onehot function assumes labels are contiguous from 0.
    if squeeze_layers or return_onehot:
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
