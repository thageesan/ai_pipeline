#!/bin/bash

brew update
brew install ansible watchman

ansible-playbook -i ansible/local_inventory ansible/data/start_remote_sync.yml