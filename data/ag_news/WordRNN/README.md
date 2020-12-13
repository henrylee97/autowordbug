## Columns
Each csv has 4 columns, which are original input, prediction of original input, perturbed input, and prediction of perturbed input.

## Attack methods
| File             | Description |
|:-----------------|:------------|
| `insert.csv`     | `insert` strategy in paper [1]. Data are generated with testset. |
| `remove.csv`     | `remove` strategy in paper [1]. Data are generated with testset. |
| `substitute.csv` | `substitute` strategy (`flip` in their implementation) in paper [1]. Data are generated with testset. |
| `swap.csv`       | `swap` strategy in paper [1]. Data are generated with testset. |

## Data lengths
| File             | Length |
|:-----------------|-------:|
| `insert.csv`     | 2127   |
| `remove.csv`     | 1990   |
| `substitute.csv` | 2137   |
| `swap.csv`       | 1111   |

## Reference
[1]: Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers [paper](https://arxiv.org/abs/1801.04354) [github](https://github.com/QData/deepWordBug)
