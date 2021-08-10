# ML PIPELINE
Demo repo to showcase data version control using DVC.

## Prerequisites
- Install DVC on your machine.  
- Ensure AWS CLI is setup on your machine correctly


# Setup Local Environment
Run  ```dvc pull``` to pull all the data required to run ML pipeline

## How to save changes remotely

## How To run experiments
Make changes to the pramams.py file.
Run ```dvc exp run -n <name of experiment>```
To create a local branch of the experiment run... ```dvc exp branch <name of experiment>```
To commit your experiment to git and dvc run... ```dvc experiments branch <name of experiment> <name of git branch>```

To compare your local environment to an experiment branch run... ```dvc exp diff <name of experiment>```

## How to view all experiments committed remotely and on your local machine
Run ```dvc exp show```

## How to run a stage without committing
Use ```dvc repro <name of stage> --no-commit```

## How to check the status of of a change

## How to add a pipeline to DVC



# Setup Remote AWS Environment
# Using AWS instance to run Pipeline


