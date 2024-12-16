
# DHB Danish Canon Project 📚 

<a href="https://chc.au.dk"><img src="https://github.com/centre-for-humanities-computing/intra/raw/main/images/onboarding/CHC_logo-turquoise-full-name.png" width="25%" align="right"/></a>
[![cc](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![cc](https://img.shields.io/badge/EMNLP-NLP4DH-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKAAAABwAgMAAADkn5ORAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAlQTFRFAAAA7R4k7RwkoncKaQAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAACxMAAAsTAQCanBgAAAJVSURBVFjD7Zc7ksMwDEPTpMnptmGD023DhqdcApCT2NvYnJTxTH7K09CiSAi+3b7X23VHIQLxQbAiA/lREJFAzsCIyOQLUZVcQvCjYx9CnwZLYxwuENI0zUZNQUbtmw8mJHsePzt6z5qBDVX/lcxwpXZO6a4enoFKSFZTvRoCXIuwnIJgVKAa1JqgdTSHIahUdIq6CHQTTDTSuZ+B/ZuJ8ZqwtpOz+3ZmYJaKIXLVG/evv2pwBopy0MVoN8GJM1D5SFc/3AX9zowFZqAiKT+K6mrztCHIPoCaQcrEmGyJXmPmDOwKcBmk+iHUXKW5QzBiKwgo3SFFgMJPQQ7Gz/vYfaEzEJSRxO8O5NTCELTS1TF0/ivc06AkL2sX+hFWwiFIpas8gCGNmYFSThwTLiWoGahujUN6WgBZJ0NQ+o4d+IDldAaGy3aXHpZZK3XOQGvefgvvkqwYgrQEnfE9WBKCGIJs/8MWPiQEmIIUpWNRgCdoDUElgoUqRUke7kjLtIafAn0a1Flhped/9g2wQZLLyasgj9yKzXj55JT4ydvgJSznwfZuPVQRyxttJ6jekNdBtb+NUdhplY4O9jDLpa6DaQcn/1aAK8xHk/J1HWRM0CF5gg63lS2b2eugV2LpxMr+2td3m3QN1OazIGwNZbyWMrzq8SxYKzRrNL0OuTme9/I0cRVMLIte9gnhHijXbb0WcxpUPGCVlsL1Uio2h/N8WDkN2nLRc0HGCzov5Wwq483KnQZ1mtlZOtjm5NxwhQmY6zGiKFd6GGCr5noOyKvg9/peo+sPhLv+BGIWS+UAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTMtMDgtMDRUMjE6NTc6MjkrMDg6MDAj62PfAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDEzLTA4LTA0VDIxOjU3OjI5KzA4OjAwUrbbYwAAAABJRU5ErkJggg==)](https://aclanthology.org/2024.nlp4dh-1.14.pdf)


###

This repository contains data (embeddings), plots and results for our project on the Danish canon of the Modern Breakthrough.


## Useful directions 📌

Some useful directions:
- the main folder contains the notebooks used for the analysis, `ML_experiment.py` is the main notebook for analysis, `descriptive.py` is used to generate descriptive stats and plots.
- `figures/` contains the figures generated
- `results/` contains the results of the ML
- `data/` contains saved embeddings (.json) used for the analysis

## Data & paper 📝

The dataset used is available at [huggingface](https://huggingface.co/datasets/MiMe-MeMo/Corpus-v1.1)

Please cite our previous [paper](https://aclanthology.org/2024.nlp4dh-1.14.pdf) if you use the code or the embeddings:

```
@inproceedings{feldkamp-etal-2024-canonical,
    title = "Canonical Status and Literary Influence: A Comparative Study of {D}anish Novels from the Modern Breakthrough (1870{--}1900)",
    author = "Feldkamp, Pascale  and
      Lassche, Alie  and
      Kostkan, Jan  and
      Kardos, M{\'a}rton  and
      Enevoldsen, Kenneth  and
      Baunvig, Katrine  and
      Nielbo, Kristoffer",
    editor = {H{\"a}m{\"a}l{\"a}inen, Mika  and
      {\"O}hman, Emily  and
      Miyagawa, So  and
      Alnajjar, Khalid  and
      Bizzoni, Yuri},
    booktitle = "Proceedings of the 4th International Conference on Natural Language Processing for Digital Humanities",
    month = nov,
    year = "2024",
    address = "Miami, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.nlp4dh-1.14",
    pages = "140--155"
}
```