# Missing Data Imputation using Optimal Transport

## Overview

This repository complements the paper [Missing Data Imputation using Optimal Transport](http://arxiv.org/abs/2002.03860) (Muzellec B., Josse J., Boyer C., Cuturi, M.):

- `experiment.py` allows to reproduce the imputation benchmark therein;
- `imputers.py` contains the classes corresponding to algorithms 1 and 3;
- `data_loaders.py` contains data loading utilities for the UCI ML repository datasets on which experiments are run;
- `utils.py` contains methods of general utility, and the implementation of MAR and MNAR missing data mechanisms in particular;
- `softimpute.py` contains the implementation of the softimpute baseline.

An example notebook is also available: `UCI_demo.ipynb`.
## References

Muzellec B., Josse J., Boyer C., Cuturi, M.: [Missing Data Imputation using Optimal Transport](http://arxiv.org/abs/2002.03860)

```
@inproceedings{muzellec2020missing,
  title={Missing Data Imputation using Optimal Transport},
  author={Muzellec, Boris and Josse, Julie and Boyer, Claire and Cuturi, Marco},
  booktitle={International Conference on Machine Learning},
  pages={7130--7140},
  year={2020},
  organization={PMLR}
}
```

## Dependencies
- Python 3+
- [PyTorch](https://pytorch.org/)
- [GeomLoss](https://www.kernel-operations.io/geomloss/)
- [POT](https://pot.readthedocs.io/en/stable/)

To use the data loading utilities in `data_loaders.py`, [wget](https://pypi.org/project/wget/) is also required.
