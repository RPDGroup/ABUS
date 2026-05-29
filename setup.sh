conda create -n abus_env python=3.7 -y
conda activate abus_env

pip install -r requirements.txt
pip3 install tqdm 
pip3 install einops 
pip3 install keras-applications==1.0.8
pip3 install opencv-python==4.5.5.62
pip3 install pandas
pip3 install torch==1.10.1
pip3 install biopython --upgrade
pip3 install plotly 
pip3 install torchsummary 
pip3 install torchsummaryX 
pip3 install scipy 
pip install scikit-learn  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install matplotlib 
pip3 install seaborn 
pip3 install ml_collections 
pip3 install kaleido 
pip3 install -U scikit-learn
pip3 install pdb2sql
pip3 install ipython
pip3 install networkx
pip3 install tensorboard
pip3 install jupyter


wget https://github.com/PyMesh/PyMesh/releases/download/v0.3/pymesh2-0.3-cp37-cp37m-linux_x86_64.whl
pip install pymesh2-0.3-cp37-cp37m-linux_x86_64.whl

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

conda install bioconda::msms
conda install bioconda::reduce
conda install -c ostrokach dssp