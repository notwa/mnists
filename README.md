# mnists

downloads and prepares various mnist-compatible datasets.

files are downloaded to `~/.mnist`
and checked for integrity by SHA-256 hashes.

### dependencies

python 3.6 (or later), numpy.

### install

`pip install --upgrade --upgrade-strategy only-if-needed 'https://github.com/notwa/mnists/tarball/master#egg=mnists'`

I've added `--upgrade-strategy` to the command-line
so you don't accidentally "upgrade" numpy to
a version not compiled specifically for your system.
This can happen when using e.g. [Anaconda.][anaconda]

[anaconda]: //www.anaconda.com/

## usage

```python
import mnists

dataset = "emnist_balanced"
train_images, train_labels, test_images, test_labels = mnists.prepare(dataset)
```

the default images shape is (n, 1, 28, 28).
pass `flatten=True` to `mnists.prepare` to get (n, 784).

## datasets

in alphabetical order, using default `mnists.prepare` parameters:

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
