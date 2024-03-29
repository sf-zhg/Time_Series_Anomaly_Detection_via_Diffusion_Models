# Anomaly Detection via Denoising Diffusion Models Probabilistic Models

# 1. Download this Repository:
```
git clone https://github.com/sf-zhg/1D_Diffusion_Ano.git
cd APPLIED_DL
```
# 2. Create a virtual environment:
create a virtual environment for to run this repository on and install dependencies. 
```
python -m venv working_environment
source working_environment/bin/activate
pip install -r requirements.txt
```
Alternatively, one can use conda to create a virtual environment:
```
conda create -n working_environment python=3.9
source activate working_environment
conda install --file requirements.txt
```
Note that ```source activate``` does not work on WindowsOS and has to be substituted by ```conda activate```

# 3. Generate synthetic time series data:

After installing the dependencies, one needs to generate a synthetic sine wave dataset. For that one can run:
```
python data/synth_data/synth_sine_series.py --amp=[0,5] /
 --freq=[0,5] /
 --phi=[0,5] /
 --samp_rate=10 /
 --time_lower_lim=0 /
 --time_upper_lim=128 /
 --path='../../raw_data_csv' /
 --name_train='sine_wave_train_data.csv' /
 --name_test='sine_wave_test_data.csv' /
 --cardinality=1024
```
The test and train datasets are then saved in a new folder raw_data_csv with the specified file names. For different data specifications, the user can simply adjust the arguments of the sine wave generator. Further details are provided with the help function:
```
python synth_sine_series.py -h
```

# 4. Train and evaluate the model:

Now one can train and evaluate the anomaly detection and diffusion model simultaneously with the run_experiment script.
The run_experiment.py script provides an easy sample on how to use the repository in its entirety. Simple run:
```
python run_experiment/run_experiment.py 
```
Again, use the help function for different training and inference specification.
```
python run_experiment/run_experiment.py -h
```

# 5. Run on Notebook:
The run_experiment folder also contains a Jupyter Notebook which can be run on Colab. it shows an example on how to run the diffusion. Remember to create an environment first via:
```
!pip install --user virtualenv
!virtualenv /content/drive/MyDrive/colab_env
```


