# This is the Github repository for CONG and CSIG

## Folder structure
./
----/cong
----/ib
----/data

- cong: contains execution code for concept matching functions. The implementation of concept-based EMD distance is included inside the similarity.py file.
- ib: contains source code for training GNN embedders based on information bottleneck
- data: contains Graph-Twitter dataset

## Twitter data:

Download the data from this link and extract to the same directory: https://drive.google.com/file/d/1IjibSWZCHuVDxppAkATcPokv6CwRiqgX/view?usp=sharing

***Note***: If you use source code from this repository please cite one of the following papers.

```
@inproceedings{bui2024toward,
  title={Toward Interpretable Graph Classification via Concept-Focused Structural Correspondence},
  author={Bui, Tien-Cuong and Li, Wen-Syan},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={20--31},
  year={2024},
  organization={Springer}
}

@inproceedings{bui2023toward,
  title={Toward interpretable graph neural networks via concept matching model},
  author={Bui, Tien-Cuong and Li, Wen-Syan},
  booktitle={2023 IEEE International Conference on Data Mining (ICDM)},
  pages={950--955},
  year={2023},
  organization={IEEE}
}
```