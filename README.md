# alt-vs-spyn

Implementing alternative variable splitting methods for Sum-Product
Network (SPN) structure learning as presented in:

_N. Di Mauro, F. Esposito, F.G. Ventola, A. Vergari_  
**Alternative Variable Splitting Methods to Learn Sum-Product Networks**  
in proceedings of AIxIA 2017.

Methods are embedded in `LearnSPN-b`, an SPN structure learner implemented in ["_spyn_"](https://github.com/arranger1044/spyn/) and presented in:

_A. Vergari, N. Di Mauro, and F. Esposito_   
**Simplifying, Regularizing and Strengthening Sum-Product Network Structure Learning**  
in proceedings of ECML-PKDD 2015.

## requirements
_alt-vs-spyn_ requires [numpy](http://www.numpy.org/) (min. version 1.12.1),
[scikit-learn](http://scikit-learn.org/stable/) (min. version 0.18.1),
[scipy](http://www.scipy.org/) (min. version 0.15.1), and [numba](http://numba.pydata.org/) (min. version 0.23.1).

## usage
Several datasets are provided in the `data/` folder.

In order to overcome the github file size limit, the training set of the  _EUR-Lex_ dataset has been split into 3 parts.
Concatenate these 3 parts into one single file before using it. For example, with `cat`:

    cat data/eurlex.ts.data.part1of3 data/eurlex.ts.data.part2of3 data/eurlex.ts.data.part3of3 > data/eurlex.ts.data


To run the algorithms and their grid search, check the `learnspn.py` script in the `bin/` folder.  

To get an overview of the possible parameters use `-h`:

    -h, --help            show this help message and exit
    -k [N_ROW_CLUSTERS], --n-row-clusters [N_ROW_CLUSTERS]
                          Number of clusters to split rows into (for DPGMM it is
                          the max num of clusters)
    -c [CLUSTER_METHOD], --cluster-method [CLUSTER_METHOD]
                          Cluster method to apply on rows ["GMM"|"DPGMM"|"HOEM"]
    -f [FEATURE_SPLIT_METHOD], --features-split-method [FEATURE_SPLIT_METHOD]
                          Feature splitting method to apply on columns ["GVS"|"RGVS"|"EBVS"|"WRGVS"|"RSBVS"]
    -e [ENTROPY_THRESHOLD], --entropy-threshold [ENTROPY_THRESHOLD]
                          The entropy threshold for entropy based feature splitting (only for EBVS)
    -j [PERCENTAGE_FEATURES], --percentage-features [PERCENTAGE_FEATURES]
                          Percentage of number of features taken at random in a features split (only for RGVS, WRGVS).
                          In any case, it takes at least 2 features at random (even if set to 0).
                          With RGVS and WRGVS, if not specified or set to -1.0, it takes SQRT #features at random.
    -l [PERCENTAGE_INSTANCES], --percentage-instances [PERCENTAGE_INSTANCES]
                          Percentage of number of instances taken at random in a features split (only RSBVS).
                          In any case, it takes at least 2 instances at random (even if set to 0).
                          With RSBVS if not specified it takes 50% of instances at random.
    --seed [SEED]         Seed for the random generator
    -o [OUTPUT], --output [OUTPUT]
                          Output dir path
    -g G_FACTOR [G_FACTOR ...], --g-factor G_FACTOR [G_FACTOR ...]
                          The "p-value like" for G-Test on columns
    -i [N_ITERS], --n-iters [N_ITERS]
                          Number of iterates for the row clustering algo
    -r [N_RESTARTS], --n-restarts [N_RESTARTS]
                          Number of restarts for the row clustering algo (only
                          for GMM)
    -p CLUSTER_PENALTY [CLUSTER_PENALTY ...], --cluster-penalty CLUSTER_PENALTY [CLUSTER_PENALTY ...]
                          Penalty for the cluster number (i.e. alpha in DPGMM
                          and rho in HOEM, not used in GMM)
    -s [SKLEARN_ARGS], --sklearn-args [SKLEARN_ARGS]
                          Additional sklearn parameters in the for of a list
                          "[name1=val1,..,namek=valk]"
    -m MIN_INST_SLICE [MIN_INST_SLICE ...], --min-inst-slice MIN_INST_SLICE [MIN_INST_SLICE ...]
                          Min number of instances in a slice to split by cols
    -a ALPHA [ALPHA ...], --alpha ALPHA [ALPHA ...]
                          Smoothing factor for leaf probability estimation
    --clt-leaves          Whether to use Chow-Liu trees as leaves
    --kde-leaves          Whether to use kernel density estimations as leaves
    --save-model          Whether to store the model file as a pickle file
    --gzip                Whether to compress the model pickle file
    --suffix              Dataset output suffix
    --feature-scheme      Path to feature scheme file
    --cv                  Folds for cross validation for model selection
    --y-only              Whether to load only the Y from the model pickle file
    -v [VERBOSE], --verbose [VERBOSE]
                          Verbosity level
    --adaptive-entropy    Whether to use adaptive entropy threshold with EBVS (EBVS-AE)

To run a grid search you can do (it uses `GVS` as variable splitting method when not specified with `-f` parameter):

    ipython -- bin/learnspn.py data/nltcs --data-ext ts.data valid.data test.data -k 2 -c GMM -g 5 10 15 20 -m 10 50 100 500 -a 0.1 0.2 1.0 2.0 -o output/learnspn_alt_vs

To use `RGVS` you can run:

    ipython -- bin/learnspn.py data/nltcs --data-ext ts.data valid.data test.data -k 2 -c GMM -f RGVS -g 5 10 15 20 -m 10 50 100 500 -a 0.1 0.2 1.0 2.0 -o output/learnspn_alt_vs

For instance, to take the 30% of variables with `RGVS` you can run:

    ipython -- bin/learnspn.py data/nltcs --data-ext ts.data valid.data test.data -k 2 -c GMM -f RGVS -j 0.3 -g 5 10 15 20 -m 10 50 100 500 -a 0.1 0.2 1.0 2.0 -o output/learnspn_alt_vs

To use `WRGVS` you can run:

    ipython -- bin/learnspn.py data/nltcs --data-ext ts.data valid.data test.data -k 2 -c GMM -f WRGVS -g 5 10 15 20 -m 10 50 100 500 -a 0.1 0.2 1.0 2.0 -o output/learnspn_alt_vs

For instance, to run a grid search taking the 20%, 30% and 45% of variables with `WRGVS` you can run:

    ipython -- bin/learnspn.py data/nltcs --data-ext ts.data valid.data test.data -k 2 -c GMM -f WRGVS -j 0.2 0.3 0.45 -g 5 10 15 20 -m 10 50 100 500 -a 0.1 0.2 1.0 2.0 -o output/learnspn_alt_vs

To use `EBVS` you can run (for `EBVS-AE` just add the `--adaptive-entropy` parameter):

    ipython -- bin/learnspn.py data/nltcs --data-ext ts.data valid.data test.data -k 2 -c GMM -f EBVS -e 0.05 0.1 0.3 0.5 -m 10 50 100 500 -a 0.1 0.2 1.0 2.0 -o output/learnspn_alt_vs

To use `RSBVS`, for example taking the 30% and 40% of instances when splitting variables, you can run:

    ipython -- bin/learnspn.py data/nltcs --data-ext ts.data valid.data test.data -k 2 -c GMM -f RSBVS -l 0.3 0.4 -g 5 10 15 20 -m 10 50 100 500 -a 0.1 0.2 1.0 2.0 -o output/learnspn_alt_vs


## docker
To try _alt-vs-spyn_ quickly you can pull and run a ready-to-go _alt-vs-spyn_ docker image (with numpy 1.12.1, scikit-learn 0.18.2, scipy 0.19.1, numba 0.24.0, llvmlite 0.9.0, llvm 3.7, python 3.5.2) through the following commands.

Pull the docker image:

    docker pull ventola/alt-vs-spyn

Run the container using the pulled image:

    docker run -i -t -d ventola/alt-vs-spyn:latest /bin/bash

For instance, to run a grid search you can execute the following command into a running container (pay attention, use absolute file pathnames):
    
    docker exec -it <your_running_docker_id> ipython -- /alt-vs-spyn/bin/learnspn.py /alt-vs-spyn/data/nltcs --data-ext ts.data valid.data test.data -k 2 -c GMM -g 5 10 15 20 -m 10 50 100 500 -a 0.1 0.2 1 2 -o output/learnspn_alt_vs

Or you may want to `docker attach` to the running _alt-vs-spyn_ container and run commands as described in the previous section.


Alternatively, you can build and run a docker image from scratch starting from the `Dockerfile` stored in this repository.

Note: this docker image takes inspiration from other docker image projects such as 
[biipy](https://github.com/cggh/biipy),
[dl-docker](https://github.com/floydhub/dl-docker),
[deepo](https://github.com/ufoym/deepo),
[docker-ipython](https://github.com/mingfang/docker-ipython), 
[rocm-testing](https://github.com/numba/rocm_testing_dockers).
