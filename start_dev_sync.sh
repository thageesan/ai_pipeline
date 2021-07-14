#!/bin/bash

brew update
brew install ansible watchman

ansible-playbook -i ansible/local_inventory ansible/update_local_environment.yml

watchman-make -p ai/**/* --run "ansible-playbook -i ansible/local_inventory ansible/update_local_environment.yml"