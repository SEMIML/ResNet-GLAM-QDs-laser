"""***************************************************************************************************************"""
"""***************************************************************************************************************"""
"""***************************************************************************************************************"""
import subprocess

# Define the Python file name to run
script_name = "02 Data preprocessing to convert PNG format to NPY format.py"

# Define different input and output paths
input_paths = ["****/Data/Shutter model", 
               "****/Data/Temperature model"
              ]
output_paths = ["****/Data/NPY/Shutter model", 
                "****/Data/NPY/Temperature model"
               ]

# Traverse each pair of input and output paths and run the script
for input_path, output_path in zip(input_paths, output_paths):
    # Construct the command to run
    command = [
        "python",
        script_name,
        input_path,
        output_path
    ]

    # Run the command and capture the output
    result = subprocess.run(command, text=True, capture_output=True)

    # Print output results
    print(f"Running with input: {input_path} and output: {output_path}")
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


"""***************************************************************************************************************"""
"""***************************************************************************************************************"""
"""***************************************************************************************************************"""
import os
import shutil
import random

def delete_csv_files(base_dir):
    print(f"Start checking and deleting all. csv files：{base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                source_path = os.path.join(root, file)
                print(f"Delete file {source_path}")
                os.remove(source_path)

def split_and_move_files(base_dir, ratio=0.8):
    for root, dirs, files in os.walk(base_dir):
        if 'select_width_size' in root and not any(sub in root for sub in ['train', 'valid']):
            print(f"\ncurrent directory：{root}")
            sub_dirs = [os.path.join(root, d) for d in dirs]
            random.shuffle(sub_dirs)
            
            split_index = int(len(sub_dirs) * ratio)
            train_dirs = sub_dirs[:split_index]
            valid_dirs = sub_dirs[split_index:]

            train_dir = root.replace('select_width_size', 'train select_width_size')
            valid_dir = root.replace('select_width_size', 'valid select_width_size')

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)

            for dir_path in train_dirs:
                shutil.move(dir_path, train_dir)
            
            for dir_path in valid_dirs:
                shutil.move(dir_path, valid_dir)
            
            print(f"Delete original directory：{root}")
            shutil.rmtree(root)

    move_to_upper_directory(base_dir)

def move_to_upper_directory(base_dir):
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for file in files:
            if file.endswith('.npy'):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(os.path.dirname(root), file)
                print(f"move file from {source_path} to {destination_path}")
                shutil.move(source_path, destination_path)

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                print(f"remove empty directories：{dir_path}")
                os.rmdir(dir_path)

def rename_folders(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'select_width_size' in dir_name:
                old_dir_path = os.path.join(root, dir_name)
                new_dir_name = dir_name.split(' select_width_size')[0]
                new_dir_path = os.path.join(root, new_dir_name)
                
                if not os.path.exists(new_dir_path):
                    print(f"Change folder name: {old_dir_path} -> {new_dir_path}")
                    os.rename(old_dir_path, new_dir_path)

def split_model_folders(base_dir):
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        train_path = model_path + " train"
        valid_path = model_path + " valid"
        
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)
        
        for root, dirs, files in os.walk(model_path):
            for dir_name in dirs:
                if 'train' in dir_name:
                    shutil.move(os.path.join(root, dir_name), train_path)
                elif 'valid' in dir_name:
                    shutil.move(os.path.join(root, dir_name), valid_path)
        
        print(f"Delete original directory：{model_path}")
        shutil.rmtree(model_path)

def remove_train_valid_suffix(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if 'model' in root:
            for dir_name in dirs:
                if ' train' in dir_name or ' valid' in dir_name:
                    old_dir_path = os.path.join(root, dir_name)
                    new_dir_name = dir_name.replace(' train', '').replace(' valid', '')
                    new_dir_path = os.path.join(root, new_dir_name)

                    if not os.path.exists(new_dir_path):
                        print(f"Change folder name: {old_dir_path} -> {new_dir_path}")
                        os.rename(old_dir_path, new_dir_path)

base_directory = '****/Data/NPY'

delete_csv_files(base_directory)
split_and_move_files(base_directory)
split_model_folders(base_directory)
rename_folders(base_directory)
remove_train_valid_suffix(base_directory) 




"""***************************************************************************************************************"""
"""***************************************************************************************************************"""
"""***************************************************************************************************************"""

import subprocess

# Parameter configuration
configs = [    
    {
        "train_path": "****/Data/NPY/Shutter model train",
        "val_path": "****/Data/NPY/Shutter model valid",
        "num_classes": 2,
        "csv_path": "./Shutter_Model_seperately_Acc_and_Loss.csv",
        "weights_path_template": "./weights_Shutter_Model_seperately_/"
    },
    {
        "train_path": "****/Data/NPY/Temperature model train",
        "val_path": "****/Data/NPY/Temperature model valid",
        "num_classes": 3,
        "csv_path": "./Temperature_Model_seperately_Acc_and_Loss.csv",
        "weights_path_template": "./weights_Temperature_Model_seperately/"
    }
]

# Script file path
script_path = "03 (Shutter model and Temperature model train separate valid) ResNet+GLAM 0.25-3 Temperature test select_width_size 24 width_length 128.py"

# Run each configuration
for config in configs:
    command = [
        "python", script_path,
        "--train_path", config["train_path"],
        "--val_path", config["val_path"],
        "--num_classes", str(config["num_classes"]),
        "--csv_path", config["csv_path"],
        "--weights_path_template", config["weights_path_template"],
        "--batch_size", "128",
        "--epochs", "500",
        "--lr", "0.01",
        "--device", "cuda:0"
    ]
    
    print(f"Running command: {' '.join(command)}")
    
    # Use Popen to read and display output line by line
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Real time output standard output and error output
    for stdout_line in process.stdout:
        print(stdout_line, end='')  # Output standard output

    for stderr_line in process.stderr:
        print(stderr_line, end='')  # Output error output
    
    process.wait()  # Waiting for the child process to end


"""***************************************************************************************************************"""
"""***************************************************************************************************************"""
"""***************************************************************************************************************"""
import subprocess
import sys

# Define different model paths and folder paths
model_paths = [
    "****/ResNet_GLAM_Shutter.onnx",
    "****/ResNet_GLAM_Shutter.onnx",
    "****/ResNet_GLAM_Shutter.onnx",
    "****/ResNet_GLAM_Shutter.onnx",
    
    "****/ResNet_GLAM_Temperature.onnx",
    "****/ResNet_GLAM_Temperature.onnx",
    "****/ResNet_GLAM_Temperature.onnx",
    "****/ResNet_GLAM_Temperature.onnx",
    "****/ResNet_GLAM_Temperature.onnx",
    "****/ResNet_GLAM_Temperature.onnx"
]

folder_paths = [
    "****/Data/Shutter model/No/1",
    "****/Data/Shutter model/No/2",
    "****/Data/Shutter model/Yes/1",
    "****/Data/Shutter model/Yes/2",
    
    "****/Data/Temperature model/High/1",
    "****/Data/Temperature model/High/2",
    "****/Data/Temperature model/Low/1",
    "****/Data/Temperature model/Low/2",
    "****/Data/Temperature model/Suitable/1",
    "****/Data/Temperature model/Suitable/2",
]

# Ensure that the target script path is correct
script_path = "04 (Without pytorch) Read small batches of data and model inference.py"

# Traverse the list of model paths and folder paths
for onnx_model_path, folder_path in zip(model_paths, folder_paths):
    # Run the target script and pass the model path and folder path as parameters
    process = subprocess.run(
        [sys.executable, script_path, onnx_model_path, folder_path],
        capture_output=True,
        text=True
    )

    # Standard output and error output of the output script
    print(process.stdout)
    print(process.stderr)



"""***************************************************************************************************************"""
"""***************************************************************************************************************"""
"""***************************************************************************************************************"""
