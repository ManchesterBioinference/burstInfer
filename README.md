# burstInfer
burstInfer is a package designed for inferring single-cell transcriptional parameters (kon, koff,  Pol II loading rate etc.) from MS2-MCP time series data, using an object-oriented Hidden Markov Model-based approach. Building upon earlier work by Lammers et al. (https://www.pnas.org/content/117/2/836), our model allows for scalable inference of transcriptional parameters for longer genes than the original model. Additionally, the model can be used to infer single-cell transcriptional parameters, allowing for modelling and visualisation of spatial gradients of transcriptional activity at single-cell resolution.

Examples included in the package outline how to process both synthetic and real Drosophila melanogaster data, train the model using Expectation Maximization and extract single-cell parameters.

## Installing burstInfer
1. Clone burstInfer repository:
```
git clone https://github.com/ManchesterBioinference/burstInfer.git
```

2. Install:
```
cd burstInfer
pip install -r requirements.txt
python setup.py install
```

See requirements.txt for a list of required packages. library_versions_used.txt contains the package versions used while completing the model. There seems to be an issue with the current version of Numpy, so it's recommended to instead run: 

```
cd burstInfer
pip install -r library_versions_used.txt
python setup.py install
```

## Included Examples
Three examples are included in the package - training the model using two synthetic genes and one experimental Drosophila dataset from Hoppe et al., Developmental Cell, 2020. These are included in the example_datasets folder in the main burstInfer folder.

Please see each of these folders for an explanation of what the examples do and which files to run.

| Folder Name  | Description   |
|---|---|
|synthetic_short_gene_example   | Generate synthetic MS2 fluorescence traces using promoter sequences created using a Markov Process. These synthetic traces have been created while specifying a 'short' window size (5), making it possible to run the model on a laptop. Train the model, get the inferred parameters and use these to generate single cell parameters. Training takes less than 5 minutes. |
|synthetic_long_gene_example   | As above, but with synthetic data generated using a longer window size (13).  |
|hoppe_et_al_ush_real_data_example   | Train the model using experimental MS2 data for the Drosophila gene ush, similar to that presented in Hoppe et al.  |

Also in the example_datasets folder are scripts to reproduce two of the figures in the paper (model running time comparison and parameter convergence).

## Included Library Files
These are the core library files containing classes and functions used to train the model and generate parameters. Each of the example folders includes a main file which typically imports and processes MS2 data and then creates an instance of the HMM class, which uses some of these utility functions.

| File Name  | Description   |
|---|---|
|calcObservationLikelihood   | Calculate HMM observation likelihood|
|calculate_single_cell_transition_rates   | Given posterior transition probabilities, convert to transition rates |
|compute_dynamic_F   | Get current compound state dynamically|
|exact_forward_backward | Exact version of forward backward (experimental)|
|export_em_parameters | Export inferred parameters as dataframe|
|forward_backward | Train model using forward-backward algorithm|
|get_adjusted | Get number of 1's and 0's in current compound state|
|get_posterior | Get posterior probabilities from trained model|
|get_single_cell_emission | Calculate single cell emission probabilities|
|HMM | HMM class, contains functions to run EM etc.|
|HMMnumba | More efficient version of HMM (experimental)|
|initialise_parameters | Initialise HMM parameters|
|log_sum_exp | Calculate log sum exp efficiently |
|logsumexp_numba | (Potentially) more efficient version of above|
|ms2_loading_coeff | Take fluorescence ramp-up during probe transit into account (similar to original model)|
|probe_adjustment | Run above|
|process_raw_data | Process MS2 data|
|v_log_solve | Calculate HMM emission parameter (similar to original model)|
