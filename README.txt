hyperopt: Distributed Asynchronous Hyper-parameter Optimization
===============================================================

Hyperopt is a Python library for serial and parallel optimization over awkward
search spaces, which may include real-valued, discrete, and conditional
dimensions.

Official project git repository:
http://github.com/hyperopt/hyperopt

Documentation:
http://hyperopt.github.io/hyperopt

Announcements mailing list:
https://groups.google.com/forum/#!forum/hyperopt-announce

Thanks
------
This work was supported in part by the National Science Foundation (IIS-0963668),
and by the Banting Postdoctoral Fellowship program.



#####################################################################
# jesse's notes:
# to install on aws:
yes | sudo yum groupinstall "Development Tools"
yes | sudo yum install emacs
yes | sudo yum install xorg-x11-xauth.x86_64 xorg-x11-server-utils.x86_64 dbus-x11.x86_64

# here you have to exit the instance, then ssh back in

mkdir software
mkdir projects
cd software
mkdir dpp_sampler
cd dpp_sampler
scp -r jessedd@pinot.cs.washington.edu:/homes/gws/jessedd/projects/dpp_sampler/DPP_Sampler ./
cd DPP_Sampler/for_redistribution
./MyAppInstaller_web.install

# this opens up a gui
# install DPP_Sampler to /home/ec2-user/software/DPP_Sampler
# install matlab runtime to /home/ec2-user/software/matlab
# put this in ~/.bashrc (the "/usr/lib64:/lib64" are to help tensorflow import /usr/lib64/libstdc++.so.6 and /lib64/libgcc_s.so.1 correctly, i.e. not the ones installed with matlab)

export LD_LIBRARY_PATH="/usr/lib64:/lib64:/home/ec2-user/software/matlab/v901/runtime/glnxa64:/home/ec2-user/software/matlab/v901/bin/glnxa64:/home/ec2-user/software/matlab/v901/sys/os/glnxa64:/home/ec2-user/software/matlab/v901/sys/opengl/lib/glnxa64"
export PYTHONPATH="/home/ec2-user/software/matlab/v901/extern/engines/python/dist"



# install anaconda:
cd ~/software
wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
bash Anaconda2-4.3.1-Linux-x86_64.sh

# install into /home/ec2-user/software/anaconda2

rm Anaconda2-4.3.1-Linux-x86_64.sh

# exit the instance, then ssh back in

cd projects 

git clone https://github.com/Noahs-ARK/ARKcat.git
git clone https://github.com/dodgejesse/hyperopt.git

yes | conda create --name arkcat python
source activate arkcat

cd hyperopt
pip install -e .

cd /home/ec2-user/software/DPP_Sampler/application
sudo python setup.py install

pip install scipy
pip install pymongo
pip install networkx
pip install pandas


############# if we want to also install arkcat

pip install scikit-learn
pip install xgboost
pip install tensorflow
