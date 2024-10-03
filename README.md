## Setup
```
conda env list
conda activate conda310diss
python -m pip install --upgrade pip
conda config --append channels conda-forge
conda config --append channels nvidia
conda install -c conda-forge cudatoolkit
conda install -c conda-forge cudnn
nvcc --version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print('Is CUDA enabled?', torch.cuda.is_available())"
pip install diffusers
pip install transformers scipy ftfy
pip install scikit-learn
pip install tqdm
pip install pandasd
pip install matplotlib
pip install seaborn
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=conda310diss
pip install ultralytics
```




