stereo_reconstruction
=====================

in this code we take the images and the odometry to find the best suitable way we can concatenate the point clouds

for running this code you need PCL 1.6 or greater and the latest OpenCV version

download the repository into a local folder

In the current directory 

cmake . \n
make 

running the code 
./3d_points_2_clouds_using_dat_file (location of folder containing left images) (location of folder containing right images) (location of odometry file)
