# ML PIPELINE
Demo repo to showcase data version control using DVC.  In this example we will be using Amazon S3 as the data repository.

## Prerequisites
- Install DVC on your machine.  
- Ensure AWS CLI is setup on your machine correctly


# Setup Local Environment
1.  Run  ```dvc pull``` to pull all the data required to run ML pipeline
2.  Create a .env file based off of the .env.template
3.  Fill the according values to the .env file

## How to visualized current pipeline
To visualize the diagram in the cli run... ```dvc dag```.
To obtain the dot format of the graph run... ```dvc dag --dot | pbcopy```.  This will copy the dot structure so you can 
paste it in an online dot visualizer (https://edotor.net)

## How to save changes remotely
Run ```dvc push``` to push the data changes to dvc.  Use git as you would in any other repo to save changes remotely.

## How To run experiments
1. Make changes to the pramams.py file.
2.  Run ```dvc exp run -n <name of experiment>```

To create a local branch of the experiment run... ```dvc exp branch <name of experiment>```

To commit your experiment to local git and dvc run... ```dvc experiments branch <name of experiment> <name of git branch>```

To compare your local environment to an experiment branch run... ```dvc exp diff <name of experiment>```

## How to view all experiments committed remotely and on your local machine
Run ```dvc exp show```

## How to run a stage without committing
Use ```dvc repro <name of stage> --no-commit```

## How to check the status of of a change

## How to add a stage to DVC



# Setup Remote AWS Environment
# Using AWS instance to run Pipeline


