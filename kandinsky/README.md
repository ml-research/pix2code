# Kandinsky Pattern Generator

This is a repository to generate Kandinsky Patterns based on the [Kandinsky Patterns project](https://github.com/human-centered-ai-lab/app-kandinsky-pattern-generator) for the paper:

### [Neuro-Symbolic Forward Reasoning](https://arxiv.org/abs/2110.09383) , preprint [GitHub](https://github.com/ml-research/nsfr)
#### [Hikaru Shindo](https://www.hikarushindo.com/), [Devendra Singh Dhami](https://sites.google.com/view/devendradhami), and [Kristian Kersting](https://ml-research.github.io/people/kkersting/index.html).

A **Kandinsky Pattern** is defined as a set of Kandinsky Figures following a "Model of Truth", i.e. for each  Kandinsky Figure, we can tell if it belongs to the Kandinsky Pattern and why this is the case.

A **Kandinsky Figure** consists of at least one (1 ... n) objects within a square with the following conditions:

* each objects has a
  - type: *circle*, *square*,  *triangle*, etc.
  - colour: *red*, *blue*,  *yellow*, etc.
  - specific size and position
* Objects are non overlapping
* Objects are completely within the square, i.e. they have a maximal size
* Objects are recognizable, i.e. they have a minimal size

We refer [the original repository](https://github.com/human-centered-ai-lab/app-kandinsky-pattern-generator) for further details. 


# Generate Patterns
The figures can be generated as follows:
```
python generate_patterns.py twopairs &
python generate_patterns.py threepairs &
python generate_patterns.py closeby &
python generate_patterns.py online-pair &
python generate_patterns.py red-triangle &
python generate_patterns.py closeby_pretrain &
python generate_patterns.py online_pretrain &
```

Parameters, e.g., the size of the objects, and the number of images to be generated can be specified in `generate_patterns.py`.

# Generate Pattern-free Figures
The pattern-free figures to train perception models can be generated as follows:
```
python src/kpgen_random.py
```
Parameters, e.g., the number of figures to be generated and the size of the figures can be specified in `src/kpgen_random.py`.
The code will produce the set of pattern-free kandinsky figures and json files that contain annotations for each object.

## License

[GPLv3+](https://choosealicense.com/licenses/gpl-3.0/)
