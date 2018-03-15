# mnists

downloads and prepares various mnist-compatible datasets.

files are downloaded to `~/.mnist`
and checked for integrity by SHA-256 hashes.

### dependencies

python 3.6 (or later), numpy.

### install

`pip install --upgrade 'https://github.com/notwa/mnists/tarball/master#egg=mnists'`

I recommend adding `--upgrade-strategy only-if-needed` to the command
so that you don't accidentally "upgrade" numpy to
a version not compiled specifically for your environment.
This can happen when using e.g. [Anaconda.][anaconda]

[anaconda]: //www.anaconda.com/

## usage

```python
import mnists

dataset = "emnist_balanced"
train_images, train_labels, test_images, test_labels = mnists.prepare(dataset)
```

the default images shape is (n, 1, 28, 28) and scaled to the range [0, 1].
labels are output in [one-hot encoding.][onehot]

[onehot]: //machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/

### `prepare` arguments

pass `flatten=True` to get a flattened (n, 784) image shape.

pass `return_floats=False` to get the raw [0, 255] integer range of images.

pass `return_onehot=False` to get the raw [0, M-1] integer encoding of labels.

### why the extra dimension?

you will notice that, by default,
there is a single-dimensional entry in the shape of images:
(n, **1,** 28, 28).
this exists to obtain compatibility with programs that
expect a number of color channels in that place.
since mnist-like datasets are (as of writing) all grayscale,
there is only one color channel, and thus the size of this dimension is 1.

## datasets

in alphabetical order, using default `mnists.prepare` arguments:

[emnist]: //www.nist.gov/itl/iad/image-group/emnist-dataset
[fashion-mnist]: //github.com/zalandoresearch/fashion-mnist
[mnist]: http://yann.lecun.com/exdb/mnist/

| dataset              | train images shape   | train labels shape   | test images shape    | test labels shape    |
| :---                 | ---:                 | ---:                 | ---:                 | ---:                 |
| emnist\_balanced     | (112800, 1, 28, 28)  | (112800, 47)         | (18800, 1, 28, 28)   | (18800, 47)          |
| emnist\_byclass      | (697932, 1, 28, 28)  | (697932, 62)         | (116323, 1, 28, 28)  | (116323, 62)         |
| emnist\_bymerge      | (697932, 1, 28, 28)  | (697932, 47)         | (116323, 1, 28, 28)  | (116323, 47)         |
| emnist\_digits       | (240000, 1, 28, 28)  | (240000, 10)         | (40000, 1, 28, 28)   | (40000, 10)          |
| emnist\_letters      | (124800, 1, 28, 28)  | (124800, 26)         | (20800, 1, 28, 28)   | (20800, 26)          |
| emnist\_mnist        | (60000, 1, 28, 28)   | (60000, 10)          | (10000, 1, 28, 28)   | (10000, 10)          |
| fashion\_mnist       | (60000, 1, 28, 28)   | (60000, 10)          | (10000, 1, 28, 28)   | (10000, 10)          |
| mnist                | (60000, 1, 28, 28)   | (60000, 10)          | (10000, 1, 28, 28)   | (10000, 10)          |
