# mnists

downloads and prepares various mnist-compatible datasets.

files are downloaded to `~/.mnist`
and checked for integrity by SHA-256 hashes.

**dependencies:** python 3.6 (or later), numpy.

**install:** `pip install --upgrade --upgrade-strategy only-if-needed 'https://github.com/notwa/mnists/tarball/master#egg=mnists'`

I've added `--upgrade-strategy` to the command-line
so you don't accidentally "upgrade" numpy to
a version not compiled specifically for your system.

## usage

```python
import mnists

dataset = "emnist_balanced"
train_images, train_labels, test_images, test_labels = mnists.prepare(dataset)
```

the default images shape is (n, 1, 28, 28).
pass `flatten=True` to `mnists.prepare` to get (n, 784).

## datasets

in alphabetical order:

### [emnist][emnist]

* `emnist_balanced`
* `emnist_byclass`
* `emnist_bymerge`
* `emnist_digits`
* `emnist_letters`
* `emnist_mnist`

### [fashion-mnist][fashion-mnist]

* `fashion_mnist`

### [mnist][mnist]

* `mnist`

[emnist]: //www.nist.gov/itl/iad/image-group/emnist-dataset
[fashion-mnist]: //github.com/zalandoresearch/fashion-mnist
[mnist]: http://yann.lecun.com/exdb/mnist/
