#!/bin/bash

brew update
brew install ansible watchman

ansible-playbook -i ansible/local_inventory ansible/update_local_environment.yml

watchman-make -p 'ai/**/*.py' 'shared/**/*.py' 'ai/Dockerfile' '.env' './**/*.sh' --run 'ansible-playbook -i ansible/local_inventory ansible/update_local_environment.yml'