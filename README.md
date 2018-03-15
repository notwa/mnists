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

in alphabetical order:

### [emnist][emnist]

 *  `emnist_balanced`  
    train images shape: (112800, 1, 28, 28)  
    train labels shape: (112800, 47)  
    test images shape: (18800, 1, 28, 28)  
    test labels shape: (18800, 47)  

 *  `emnist_byclass`  
    train images shape: (697932, 1, 28, 28)  
    train labels shape: (697932, 62)  
    test images shape: (116323, 1, 28, 28)  
    test labels shape: (116323, 62)  

 *  `emnist_bymerge`  
    train images shape: (697932, 1, 28, 28)  
    train labels shape: (697932, 47)  
    test images shape: (116323, 1, 28, 28)  
    test labels shape: (116323, 47)  

 *  `emnist_digits`  
    train images shape: (240000, 1, 28, 28)  
    train labels shape: (240000, 10)  
    test images shape: (40000, 1, 28, 28)  
    test labels shape: (40000, 10)  

 *  `emnist_letters`  
    train images shape: (124800, 1, 28, 28)  
    train labels shape: (124800, 26)  
    test images shape: (20800, 1, 28, 28)  
    test labels shape: (20800, 26)  

 *  `emnist_mnist`  
    train images shape: (60000, 1, 28, 28)  
    train labels shape: (60000, 10)  
    test images shape: (10000, 1, 28, 28)  
    test labels shape: (10000, 10)  

### [fashion-mnist][fashion-mnist]

 *  `fashion_mnist`  
    train images shape: (60000, 1, 28, 28)  
    train labels shape: (60000, 10)  
    test images shape: (10000, 1, 28, 28)  
    test labels shape: (10000, 10)  

### [mnist][mnist]

 *  `mnist`  
    train images shape: (60000, 1, 28, 28)  
    train labels shape: (60000, 10)  
    test images shape: (10000, 1, 28, 28)  
    test labels shape: (10000, 10)  

[emnist]: //www.nist.gov/itl/iad/image-group/emnist-dataset
[fashion-mnist]: //github.com/zalandoresearch/fashion-mnist
[mnist]: http://yann.lecun.com/exdb/mnist/
