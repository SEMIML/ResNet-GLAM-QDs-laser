# Project Overview

This project aims to optimize and analyze the processed RHEED data using machine learning methods. It includes codes, datasets, and programs tailored for deployment on specific equipment. Below is a detailed description of the folder structure and contents.


## Environment Setup

To run the project, you need to set up the Python environment with specific dependencies, especially for using CUDA-enabled PyTorch with NVIDIA GPUs. Follow the steps below to install the environment and necessary packages.

### Step 1: Update NVIDIA GPU Driver
Ensure your NVIDIA GPU driver is updated to the latest version. This is required to support CUDA 11.8.

### Step 2: Install CUDA 11.8
Download and install CUDA 11.8 from the NVIDIA website. This version is compatible with the PyTorch version we will install later.

### Step 3: Install Conda
Download and install Conda from the official **[Anaconda](https://www.anaconda.com/)** website. Conda will help in managing the Python environment and packages.

### Step 4: Create a Python Environment
Create a new Python environment with Python version 3.9 using Conda:
```bash
conda create -n myenv python=3.9
conda activate myenv
```
### Step 5: Install PyTorch with CUDA 11.8 Support
Install PyTorch and torchvision that support CUDA 11.8:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
### Step 6: Install Additional Python Packages

After setting up the environment, you need to install the additional Python packages required for this project. You can install them using `conda` or `pip` as shown below:

# Install common dependencies
```bash
conda install pandas numpy matplotlib scikit-learn tqdm seaborn opencv ipython
```
# Install packages via pip
```bash
pip install onnxruntime pillow torch torchvision scikit-image
```
# List of Required Packages
The following Python packages are required for the project:
```bash
"torch", "pandas", "numpy", "torchvision", "PIL", "sklearn", "matplotlib", "tqdm", "seaborn", "IPython", "opencv-python", "onnxruntime"
```
These steps will set up the environment necessary for running the project's code and models.


## QDs laser code.zip File

This file contains the scripts and model files related to the data preprocessing, format modification, and model training for this research project. Each script is designed to perform specific tasks, and they work together to streamline the overall process:

* **01**: This script sequentially calls the scripts **02**, **03**, and **04** to perform data preprocessing, format modification, and model training. It randomly divides the dataset into training and validation sets according to a specified ratio to ensure that different data are used for training and validation. Finally, it deploys the model by calling the pre-trained ONNX file we provided.
* **02**: This script is responsible for data preprocessing, including image augmentation, concatenation, and other related operations.
* **03**: This script is used to train the final model with fixed parameters. It is designed to handle the final training phase with optimal configurations.
* **04**: This script uses the pre-trained model, which has been converted to ONNX format, from the **ONNX file** folder. It performs online inference on small-batch datasets stored in the folder and generates output results.
* **05**: This folder stores the final model used in this research, converted to ONNX format for cross-platform inference.

Each script plays a crucial role in the research workflow, from data preparation to model deployment.

## QDs laser data.zip File

This file contains small-batch datasets related to the two models used in this research:

* Data is organized by model name and label name for convenient loading and use.* 

## How to Run the Example Code 

To run the **01** script, which handles data preprocessing, dataset splitting, model training, and model deployment, follow these steps. In the **01** script, each "Section" is separated by "*". 

1. **Prepare the Data**:
   * Before using the data, unzip the **QDs laser data** package and store it in a **Data**folder without changing the file format.
   * Define the path of this folder as `****` and, following the format in **01**, replace instances of `****` in the script with this path.

2. **Check the First Section**:
   * Ensure the `script_name` is set to the **02** script in the same file path.
   * Verify each pair of `input_paths` and `output_paths`, used for preprocessing the data and storing the preprocessed data, respectively.

3. **Check the Second Section**:
   * Set the `base_directory` to the path of the processed NPY files from the previous step.

4. **Check the Third Section**:
   * Review each set of parameters in `configs`, where each set corresponds to a training model.
   * Set different paths for `train_path`, `val_path`, `num_classes`, `csv_path`, and `weights_path_template`:
     * `train_path`: Path for the model training data.
     * `val_path`: Path for the model validation data.
     * `num_classes`: Number of classes the model should classify.
     * `csv_path`: Path to store the results generated during model training.
     * `weights_path_template`: Path to store the weight information generated during model training.
   * Ensure `script_path` points to the **03** script.

5. **Check the Fourth Section**:
   * Verify `model_paths` and `folder_paths` for the models and the data used for inference.
   * Ensure `script_path` points to the **04** script.
   
6. **Running the Code**:
   * Open the Command Line (Windows)：Press `Win + R`, type cmd, and press `Enter`.
   * Change Command Line Directory to the Code Location. If the code is stored in `D:\QDs laser code`, run: `cd /d D:\QDs laser code`.
   * Run `01 Main Loop.py`： python "01 Main Loop.py"



## Labview program QDs laser.zip File

This file contains the Labview program designed for the Riber 32P system:

* Temperature reading and control are implemented using the universal Eurotherm serial communication protocol.
* The shutter controller code is written based on the system manual.
* The camera interface uses USB 3.0 for data acquisition.
* The program is deployed in this research and supports real-time RHEED data processing and feedback control.
* Before running the program, ensure the following folders are manually created and correctly set:
  * **Real time storage Excel Folder**: For real-time data output.
  * **Image save Folder**: For storing RHEED image data.
  * **ONNX File**: Path for checking and calling ONNX files

## Xlsx file

* **Data_S1**: Contains the raw data used for plotting the figures in the article.

## Notes

1. Ensure that all required environments and dependencies are properly installed for running the codes and performing ONNX model inference.
2. The Labview program is exclusively designed for the Riber 32P system and requires appropriate configuration to match the specific system environment.
3. Verify all paths and dependencies before running the codes to prevent errors during execution.

## Contact Information

For further information or questions, please contact the research team.
