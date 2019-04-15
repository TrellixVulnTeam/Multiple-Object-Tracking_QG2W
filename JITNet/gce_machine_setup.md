## Google Compute Engine Machine Setup

If Tensorflow is the main framework being used, GCE provides a Deep Learning Virtual Machine based on Debian, optimized to run on GCE machines.

Follow the instructions here, under **Launching a TensorFlow Deep Learning VM Instance from the Cloud Marketplace**: https://cloud.google.com/deep-learning-vm/docs/tensorflow_start_instance

Change vCPUs to 8, memory to 72GB (this is a default machine type that's price optimized, with the max available RAM). Choose to install NVIDIA drivers. Set region to us-west1-b.

Once deployed, wait for all the installations to finish (there should be a page showing install progress). The deployment creates a new
Compute Engine VM instance.

The VM can be accessed through SSH or gcloud command line. (ex. gcloud compute ssh stevenzc3@tensorflow-1-vm). 

The VM has tensorflow 1.10 preinstalled, as well as many python ML libraries. However, these are installed in Python 3.5. As is, it can run JITNet timing code without any installs.

The machine has a static external IP and allows http and https traffic. There is also a VPC firewall rule added to allow tcp:7000 traffic from external sources, which can be used for Jupyter notebook.

## Disk Setup

Each VM has its own boot disk, but we can use a data disk to share between multiple instances. However, in order for the disk to be used
at the same time by all machines, they need to mount it read only.

Followed this guide: https://cloud.google.com/compute/docs/disks/add-persistent-disk to create a persistent disk, format it, and mount it. The disk is currently mounted on tensorflow-1-vm as RW.

## Software Setup

The machine tensorflow-1-vm currently has the following installed:

* ffmpeg with https (ssl) support, for downloading youtube streams
* youtube-dl
* CUDA 9.2
* Anaconda Python 3.6 as default python and pip
* Tensorflow 1.10.1 compiled from source for the machine and CUDA version
* PyTorch 0.4.1
* Jupyter Notebook

All the Python packages are installed in the Anaconda default Python (no conda env or virtualenv used).

Google Deep Learning VMs come with prebuilt whl binaries for Tensorflow, found in `/opt/deeplearning/binaries/tensorflow`.

## Jupyter Notebook

To use Jupyter Notebook, launch the notebook with `jupyter notebook --ip=0.0.0.0 --port=7000 --no-browser`.
Take the URL with token, and replace the machine name/localhost with the static IP address of the machine.

## Notes

The VM I created is called tensorflow-1-vm. The VM GPU, CPU, RAM can be changed at any time while the instance is stopped. The only changes
I made to the VM were to add the ssh key to my GitHub and clone JITNet repo in ~/git.

The disk I created is called tensorflow-disk. It will currently auto-mount to tensorflow-1-vm at /mnt/disks/tensorflow-disk.
