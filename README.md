# Textomics: A Dataset for Genomics Data Summary Generation

This repository provides resources developed within the following article:

> Mu-Chun Wang, Zixuan Liu, and Sheng Wang. 2022. Textomics: A Dataset for Genomics Data Summary Generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics 

## Abstract
Summarizing biomedical discovery from genomics data using natural languages is an essential step in biomedical research but is mostly done manually. Here, we introduce Textomics, a novel dataset of genomics data description, which contains 22,273 pairs of genomics data matrices and their summaries. Each summary is written by the researchers who generated the data and associated with a scientific paper. Based on this dataset, we study two novel tasks: generating textual summary from a genomics data matrix and vice versa. Inspired by the successful applications of k nearest neighbors in modeling genomics data, we propose a kNN-Vec2Text model to address these tasks and observe substantial improvement on our dataset. We further illustrate how Textomics can be used to advance other applications, including evaluating scientific paper embeddings and generating masked templates for scientific paper understanding. Textomics serves as the first benchmark for generating textual summaries for genomics data and we envision it will be broadly applied to other biomedical and natural language processing applications.

## Citation

If you use the resources presented in this repository, please cite:

```
@inproceedings{wang-etal-2022-textomics,
    title = "Textomics: A Dataset for Genomics Data Summary Generation",
    author = "Wang, Mu-Chun  and
      Liu, Zixuan  and
      Wang, Sheng",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.335",
    pages = "4878--4891",
    abstract = "Summarizing biomedical discovery from genomics data using natural languages is an essential step in biomedical research but is mostly done manually. Here, we introduce Textomics, a novel dataset of genomics data description, which contains 22,273 pairs of genomics data matrices and their summaries. Each summary is written by the researchers who generated the data and associated with a scientific paper. Based on this dataset, we study two novel tasks: generating textual summary from a genomics data matrix and vice versa. Inspired by the successful applications of $k$ nearest neighbors in modeling genomics data, we propose a $k$NN-Vec2Text model to address these tasks and observe substantial improvement on our dataset. We further illustrate how Textomics can be used to advance other applications, including evaluating scientific paper embeddings and generating masked templates for scientific paper understanding. Textomics serves as the first benchmark for generating textual summaries for genomics data and we envision it will be broadly applied to other biomedical and natural language processing applications.",
}
```

## Contact

Should you have any questions, please contact Mu-Chun Wang (muchunw2@illinois.edu)