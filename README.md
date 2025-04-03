# Pix2Code

This is the official repository for the paper [Pix2Code: Learning to Compose Neural Visual Concepts as Programs](https://arxiv.org/abs/2402.08280) which contains code for Pix2Code as well as dataset links for the introduced data sets. Pix2Code is a neuro-symbolic framework for generalizable, inspectable and revisable visual concept learning. By utilizing both neural and program synthesis components, Pix2Code integrates the power of neural object representations with the generalizability and readability of program representations.

<img src="figures/overview.png" alt="Pix2Code overview" height="400">

## Created Datasets
### RelKP

In our work we created the dataset RelKP which consists of 200 Kandinsky Patterns that are based on relational clauses. The code for the generation is based on the [Kandinsky Pattern Generator](https://github.com/ml-research/kandinsky_generator) of Shindo et al. 

The created patterns have varying complex concepts based on the number of objects, concept types, number of relations and number of pairs. There are two types of concepts, those where the relations refer to all objects in the image and those where the relations refer to only a pair of objects in the image. The relations include object concepts like "same shape" and "one object is a red triangle".

The patterns were createde using the script in `kandinsky/src/generate_tasks.py` which stores the data in folders for each task, i.e., `/relkp/<support | query>/<clause-type>/<taskname>/<true | false>`. Each folder gets an instances.json file with meta information. Folder structure:

```
relkp 
| - support 
| -- 2_no_pairs 
| --- 2_same_color 
| ---- true 
| ----- 000000.png 
| ----- 000000.json 
| ----- ...
| ----- instances.json 
| ---- false 
| ----- 000000.png 
| ----- ... 
| ---- ... 
| - query
. 
. 
```
The RelKP dataset can be downloaded [here](https://hessenbox.tu-darmstadt.de/getlink/fi2AvLVx6AyCY1btCkauxeMr/relkp.zip) for the image folder structure and [here](https://hessenbox.tu-darmstadt.de/getlink/fiHBwrsZk1X4geWZoR59iz8T/rel_kp_curi_format.zip) for the CURI-like folder structure. 

<img src="figures/kandinsky.jpg"  height="400">

### AllCubes-N and AllMetalOneGray-N
The data sets [AllCubes-N](https://hessenbox.tu-darmstadt.de/getlink/fiF7bBW5EbbLHJqJ8NMcWzZS/all-cubes-X.zip) and [AllMetalOneGray-N](https://hessenbox.tu-darmstadt.de/getlink/fiW5Va9ACWcG6n1oEJrwRTqd/all-metal-one-gray-X.zip) are based on the [CURI](https://github.com/facebookresearch/productive_concept_learning) data set of Vendantam et al. 
The aim of the data sets is to test for entity level generalization of the concepts `all objects are cubes` and `all objects are metal and one is gray`. For this 200 new test examples of each concept were created with an increased number of concepts, i.e. 5, 8 and 10. 

<img src="figures/all-cubes.jpg"  height="200"> <img src="figures/all-metal-one-gray.jpg"  height="200">

### CURI-Hans
A small subset of the CURI data set where one test task has been confounded, download [here](https://hessenbox.tu-darmstadt.de/getlink/fiDMY5wXvQaC3FCkC1gotdaF/confounded-clevr.zip). The test task is the concept `There exists a cube and all objects are metal` and the confounder `cyan` is added to the support set of this task.

## Setup
For pix2seq, you can use the provided `pix2seq/Dockerfile` and `pix2seq/docker-compose.yml` file to setup a Docker container. 
To use DreamCoder, setup the Docker container from `dreamcoder/Dockerfile`. 

## Object extraction and task formulation
### Pix2Seq
For the implementation of Pix2Seq we use the code of the pytorch implementation with pretrained model of [Pretrained-Pix2Seq](https://github.com/gaopengcuhk/Pretrained-Pix2Seq). For Pix2Code, Pix2Seq is trained on data sets of random Kandinsky Patterns and random CLEVR images, both of size 2000. To train pix2seq, use following command:
```
sh train.sh --model pix2seq --coco_path <DATA_DIR> --output_dir <RESULT_DIR> 
```

We provide here the fine-tuned checkpoints for [Kandinsky](https://hessenbox.tu-darmstadt.de/getlink/fiEvAqt1FEn1GXkdYo93C8tK/pix2seq_checkpoint_kandinsky.pth) and [CLEVR](https://hessenbox.tu-darmstadt.de/getlink/fiU3fe6ao2KRm1Q891168Gym/pix2seq_checkpoint_best_clevr.pth).

### Convert KP to DreamCoder tasks
1. Download RelKP and store it in `data/kandinsky`
2. Process Kandinsky images with the pix2seq model. Use file `pix2seq/use_model_kandinksy.py`. 
3. The pix2seq output needs to be processed into DreamCoder task format. This is done in `pix2seq/convert_to_dreamcoder.py`.

Whole execution trace:
```bash
python pix2seq/use_model_kandinsky.py --coco_path <path_to_data_set> --output_dir <path_to_model_results>
python pix2seq/convert_to_dreamcoder.py --input_path <path_to_model_results> --output_path <path_to_target_folder> --domain "kandinsky"
```

If the usage of Pix2Seq is supposed to be skipped and the annotations of the Kandinsky Patterns are supposed to be used for the DreamCoder tasks (i.e. schema representations), this can be done by using `kandinsky/src/pix2seq_shortcut.py`:

```bash
python kandinsky/src/pix2seq_shortcut.py
```

### Convert CURI to DreamCoder tasks
1. Download CURI as explained [here](https://github.com/facebookresearch/productive_concept_learning?tab=readme-ov-file#download-the-curi-dataset) and store it in `curi/curi_release`
2. Filter CURI data set with `curi/filter_meta_dataset.py`
3. Convert CURI to task format based on schema representations by using `curi/data_processing/create_curi_tasks.py`
4. Use pix2seq on CURI images (unordered) `use_model_clevr.py` and `create_curi_images_tasks.py`
5. Convert CURI to task format based on image representations retrieved with pix2seq by using `curi/data_processing/create_curi_images_tasks.py`

Whole execution trace:
```bash
python curi/data_processing/filter_meta_data.py
python curi/data_processing/create_curi_tasks.py
python pix2seq/use_model_clevr.py --coco_path "curi/curi_release/images/" --output_dir "data/curi_release_model_results/" --resume <path-to-pix2seq-checkpoint>
python pix2seq/convert_to_dreamcoder.py --input_path "data/curi_release_model_results" --output_path "data/curi_release_dc_inputs" --domain "clevr"
pyhton curi/data_processing/create_curi_images_tasks.py --target_folder "curi_image_dc_test_tasks" --mode "test"
```


## Program Synthesis with DreamCoder
The code for using DreamCoder to synthesize programs is based on the official [DreamCoder](https://github.com/ellisk42/ec) repository of Ellis et al. and adapted for our use case.
To set up the code you can use `dreamcoder/Dockerfile`. 

To run experiments on the RelKP dataset proceed as follows, run
```bash
python bin/relations.py
```
The experiments are implemented in `dreamcoder/dreamcoder/domains/relations/main.py`. In `relations.py` it can be specified which experiment is started and which parameters are used. 
The evaluation is performed via `python bin/relations.py` as well, there the flag `eval` needs to be set to true in the method call.


For the CURI dataset, run
```bash
python bin/clevr.py
```
The experiments are implemented in `dreamcoder/dreamcoder/domains/clevr/main.py` and the parameters can be specified in `clevr.py` as well. 


## Citation 
If you find the code of this repository helpful, consider citing us.

```
@article{wust2024pix2code,
  title={Pix2Code: Learning to Compose Neural Visual Concepts as Programs},
  author={W{\"u}st, Antonia and Stammer, Wolfgang and Delfosse, Quentin and Dhami, Devendra Singh and Kersting, Kristian},
  journal={arXiv preprint arXiv:2402.08280},
  year={2024}
}
```
