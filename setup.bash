
#!/bin/bash

if [[ ! -f "$(pwd)/setup.bash" ]]
then
  echo "please launch from the xadapt_ctrl folder!"
  exit
fi

PROJECT_PATH=$(pwd)
echo "project path: $PROJECT_PATH"

echo "Setting up the project..."
git submodule update --init --recursive

cd py3dmath
sudo -H pip install -e .

cd $PROJECT_PATH
conda env create -f environment.yml
conda activate xadap

python simulate.py 
echo "Have a safe flight!"
