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
jesse's notes:
to install on aws:
sudo yum groupinstall "Development Tools"
sudo yum install emacs
sudo yum install xorg-x11-xauth.x86_64 xorg-x11-server-utils.x86_64 dbus-x11.x86_64

# here you have to exit the instance, then ssh back in

mkdir software
mkdir projects
cd software
mkdir dpp_sampler
cd dpp_sampler
scp -r jessedd@pinot.cs.washington.edu:/homes/gws/jessedd/projects/dpp_sampler/ ./
cd DPP_Sampler/for_redistribution
./MyAppInstaller_web.install

# this opens up a gui
# install DPP_Sampler to /home/ec2-user/software/DPP_Sampler
# install matlab runtime to /home/ec2-user/software/matlab
# put this in ~/.bashrc:

export LD_LIBRARY_PATH="/home/ec2-user/software/matlab/v901/runtime/glnxa64:/home/ec2-user/software/matlab/v901/bin/glnxa64:/home/ec2-user/software/matlab/v901/sys/os/glnxa64:/home/ec2-user/software/matlab/v901/sys/opengl/lib/glnxa64
export PYTHONPATH="/home/ec2-user/software/matlab/v901/extern/engines/python/dist"
cd /home/ec2-user/software/DPP_Sampler/application
sudo python setup.py install

# exit the instance, then ssh back in

sudo pip install numpy

# HERE IT WORKS

install anaconda, now it doesn't work.
i think it's because anaconda uses an older version of numpy. if we create a conda environment
