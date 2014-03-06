#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <opencv/cv.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <fnmatch.h>
#include <fstream>
#include <iterator>
 #include <sstream>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <ctime>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
//#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/thread/thread.hpp>
#include <pcl/features/normal_3d.h>

#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/surface/convex_hull.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>




#include <pcl/common/time.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/ModelCoefficients.h>
//#include <pcl/segmentation/organized_multi_plane_segmentation.h>
//#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/console/parse.h>
//#include <pcl/geometry/polygon_operations.h>






#define CUSTOM_REPROJECT
const char *lreg, *rreg;
typedef double Matrix[4][3];
typedef double Matrix1[3];
using namespace cv;
pcl::PCDWriter pcdWriter;
pcl::PLYWriter writer;
int isLeft(struct dirent const *entry)
{
//	printf("hello %s \n",entry->d_name);
  return !fnmatch(lreg, entry->d_name,0);
}
 double string_to_double( const std::string& s )
 {
   std::istringstream i(s);
   double x;
   if (!(i >> x))
     return 0;
   return x;
 } 

int isRight(struct dirent const *entry)
{
 // printf("hello 12%s \n",entry->d_name);
  return !fnmatch(rreg, entry->d_name,0);
}
boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}
boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer1 (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  viewer->addCoordinateSystem ( 1.0 );
  viewer->initCameraParameters ();
  return (viewer);
}
boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis (
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud and normals-----
  // --------------------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb,"sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 10, 10, "normals");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

int main(int argc, char** argv)
{
	const char* algorithm_opt = "--algorithm=";
    const char* img1_filename = 0;
    const char* img2_filename = 0;
    const char* disparity_filename = 0;
    const char* q_xml=0;

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
    int alg = STEREO_SGBM;
    int SADWindowSize = 0, numberOfDisparities = 0;
    bool no_display = false;
    float scale = 1.f;

    StereoBM bm;
    StereoSGBM sgbm;
    StereoVar var;
 if(argc < 4)
    {
        return 0;
    }
    for( int i = 1; i < argc-1; i++ )
    {
        if( argv[i][0] != '-' )
        {
            if( !img1_filename )
                img1_filename = argv[i];
            else
                img2_filename = argv[i];
        }
        else if( strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0 )
        {
            char* _alg = argv[i] + strlen(algorithm_opt);
            alg = strcmp(_alg, "bm") == 0 ? STEREO_BM :
                  strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
                  strcmp(_alg, "hh") == 0 ? STEREO_HH :
                  strcmp(_alg, "var") == 0 ? STEREO_VAR : -1;
            if( alg < 0 )
            {
                printf("Command-line parameter error: Unknown stereo algorithm\n\n");
                return -1;
            }
        }
        else
        {
            printf("Command-line parameter error: unknown option %s\n", argv[i]);
            return -1;
        }
    }
    q_xml=argv[3];

    if( !img1_filename || !img2_filename )
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }
  cv::FileStorage fs(q_xml, cv::FileStorage::READ);
  cv::Mat Q;
  
  fs["Q"] >> Q;
  
  //If size of Q is not 4x4 exit
  if (Q.cols != 4 || Q.rows != 4)
  {
    std::cerr << "ERROR: Could not read matrix Q (doesn't exist or size is not 4x4)" << std::endl;
    return 1;
  }
	#ifdef CUSTOM_REPROJECT
  //Get the interesting parameters from Q
  double Q03, Q13, Q23, Q32, Q33;
  Q03 = Q.at<double>(0,3);
  Q13 = Q.at<double>(1,3);
  Q23 = Q.at<double>(2,3);
  Q32 = Q.at<double>(3,2);
  Q33 = Q.at<double>(3,3);
  
  std::cout << "Q(0,3) = "<< Q03 <<"; Q(1,3) = "<< Q13 <<"; Q(2,3) = "<< Q23 <<"; Q(3,2) = "<< Q32 <<"; Q(3,3) = "<< Q33 <<";" << std::endl;
#endif  
  std::cout << "Read matrix in file " << argv[3] << std::endl;

  //Show the values inside Q (for debug purposes)
  /*
  for (int y = 0; y < Q.rows; y++)
  {
    const double* Qy = Q.ptr<double>(y);
    for (int x = 0; x < Q.cols; x++)
    {
      std::cout << "Q(" << x << "," << y << ") = " << Qy[x] << std::endl;
    }
  }
  */
	IplImage *limg, *rimg, *img_col;
    struct dirent **limgs, **rimgs;
	int nlimgs,nrimgs=0;

/**
   * Preparing a list of left and right images.
   * If lett and right images are in seperate folder then there will be two args: <path to left images folder> <path to right images folder>
   */
   cout<<argc<<endl;
	if(argc < 3)
	{
		return -1;
	}
	
	
	    lreg = "*";
		rreg = "*";
	nlimgs = scandir(argv[1], &limgs, isLeft, versionsort)-2;
	nrimgs = scandir(argv[2], &rimgs, isRight, versionsort)-2;
	
	printf("left=%d, right=%d\n", nlimgs, nrimgs);
	if(nlimgs <0 || nrimgs <0)
	  {
		if(argc==3 && nrimgs<0)
		  perror(argv[2]);
		else
		  perror(argv[1]);
		return 0;
	  }
	if(nlimgs != nrimgs)
	  {
		printf("Number of left images and right images is not equal: left=%d, right%d.\n", nlimgs, nrimgs);
		return 0;
	  }
	std::cout << "Creating Point Cloud..." <<std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
     pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
     pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr2 (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr custom_coloured (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloured (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr custom_coloured2 (new pcl::PointCloud<pcl::PointXYZRGB>);


	//for(int i=0;i<nlimgs;i
	for(int num=10;num<60;num+=3)
     {
    cout<<"image number "<<num<<endl;
    //fn<<argv[2]<<"/"<<rimgs[i+2]->d_name;
    //fl<<argv[1]<<"/"<<limgs[i+2]->d_name;
    //printf("fn=%s\n", fn.str().c_str());
    //printf("fl=%s\n", fl.str().c_str());
    //limg = cvLoadImage(fn.str().c_str(), CV_LOAD_IMAGE_COLOR);
	//rimg = cvLoadImage(fl.str().c_str(), CV_LOAD_IMAGE_COLOR);
    //Mat img1 = limg;
    //Mat img2 = rimg;
  
   	std::stringstream fn,fl;
    int color_mode = alg == STEREO_BM ? 0 : -1;
    fn<<argv[2]<<"/"<<rimgs[num+2]->d_name;
    fl<<argv[1]<<"/"<<limgs[num+2]->d_name;
    printf("fn=%s\n", fn.str().c_str());
    printf("fl=%s\n", fl.str().c_str());
    Mat img1 = imread(fl.str().c_str(), color_mode);
    Mat img2 = imread(fn.str().c_str(), color_mode);

   if( scale != 1.f )
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q1;

   numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    bm.state->roi1 = roi1;
    bm.state->roi2 = roi2;
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
    bm.state->minDisparity = 0;
    bm.state->numberOfDisparities = numberOfDisparities;
    bm.state->textureThreshold = 10;
    bm.state->uniquenessRatio = 15;
    bm.state->speckleWindowSize = 100;
    bm.state->speckleRange = 32;
    bm.state->disp12MaxDiff = 1;

    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

    int cn = img1.channels();

    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = numberOfDisparities;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = bm.state->speckleWindowSize;
    sgbm.speckleRange = bm.state->speckleRange;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = alg == STEREO_HH;

    var.levels = 3;                                 // ignored with USE_AUTO_PARAMS
    var.pyrScale = 0.5;                             // ignored with USE_AUTO_PARAMS
    var.nIt = 25;
    var.minDisp = -numberOfDisparities;
    var.maxDisp = 0;
    var.poly_n = 3;
    var.poly_sigma = 0.0;
    var.fi = 15.0f;
    var.lambda = 0.03f;
    var.penalization = var.PENALIZATION_TICHONOV;   // ignored with USE_AUTO_PARAMS
    var.cycle = var.CYCLE_V;                        // ignored with USE_AUTO_PARAMS
    var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;

    Mat disp, disp8 , disp1,disp9;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    int64 t2 = getTickCount();
    if( alg == STEREO_BM )
   {     bm(img1, img2, disp);
        bm(img2, img1, disp1);
    }else if( alg == STEREO_VAR ) {
        var(img1, img2, disp);
         var(img1, img2, disp1);
    }
    else if( alg == STEREO_SGBM || alg == STEREO_HH )
 {       sgbm(img1, img2, disp);
                sgbm(img1, img2, disp1);
  }  t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if( alg != STEREO_VAR )
       { disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
                disp.convertTo(disp9, CV_8U, 255/(numberOfDisparities*16.));
   } else
    {    disp.convertTo(disp8, CV_8U);
                disp.convertTo(disp9, CV_8U);
}
 //if( !no_display )
    //{
        //namedWindow("left", 1);
        //imshow("left", img1);
        //namedWindow("right", 1);
        //imshow("right", img2);
        //namedWindow("disparity", 0);
        //imshow("disparity", disp8);
        //namedWindow("disparity1", 0);
        //imshow("disparity1", disp9);

        //printf("press any key to continue...");
        //fflush(stdout);
        //waitKey();
        //printf("\n");
    //}

  cv::Mat img_rgb = cv::imread(fn.str().c_str(), CV_LOAD_IMAGE_COLOR);
  //Load disparity image
  Mat img_disparity = disp8;
    //Show both images (for debug purposes)
  //cv::namedWindow("rgb-image");
  //cv::namedWindow("disparity-image");
  //imshow("rbg-image", img_rgb);
  //cv::imshow("disparity-image", img_disparity);
  //std::cout << "Press a key to continue..." << std::endl;
  //cv::waitKey(0);
  //cv::destroyWindow("rgb-image");
  //cv::destroyWindow("disparity-image");
  //Create point cloud and fill it
     double Mx[11];
  uchar pr, pg, pb;

  double px, py, pz,dx,dy,dz;
  double x,y,z,p,t1,r;

#ifdef PITCH
 std::ifstream f("NewCollege_3_11_2008__VO.dat");
 std::istream_iterator<std::string> beg(f), end;

  const int SIZE = distance(beg, end) ;
  std::string words[SIZE]; // C array? to hold our words we read in.
  std::string str; // Temp string to
 std::ifstream fin("NewCollege_3_11_2008__VO.dat");
  for (int j = 0; (fin >> str) && (j < SIZE); ++j) // Will read up to eof() and stop at every
  {                                                // whitespace it hits. (like spaces!)
    words[j] = str;
   } // We now also need to stop before we hit the no define array space.
  fin.close();
 string name = words[1+(15*num)];

 double x1 = string_to_double (words[9+(15*num)]);
 double y1 = string_to_double (words[10+(15*num)]);
 double z1 = string_to_double (words[11+(15*num)]);
 double roll = string_to_double (words[12+(15*num)]);
 double pitch = string_to_double (words[13+(15*num)]);
 double yaw= string_to_double (words[14+(15*num)]);
    double Sx = sinf(pitch);
    double Sy = sinf(-yaw);
    double Sz = sinf(roll);
    double Cx = cosf(pitch);
    double Cy = cosf(-yaw);
    double Cz = cosf(roll);
        Mx[0]=Cy*Cz+Sx*Sy*Sz;
        Mx[1]=Cz*Sy*Sx-Cy*Sz;
        Mx[2]=Cx*Sy;
        Mx[3]=Cx*Sz;
        Mx[4]=Cx*Cz;
        Mx[5]=-Sx;
        Mx[6]=Cy*Sx*Sz-Cz*Sy;
        Mx[7]=Sy*Sz+Cy*Cz*Sx;
        Mx[8]=Cx*Cy;

 
        
        Mx[9]=x1;
        Mx[10]=y1;
        Mx[11]=z1;
#else
std::ifstream f("translation.txt");
 std::istream_iterator<std::string> beg(f), end;

  const int SIZE = distance(beg, end) ;
  std::string words[SIZE]; // C array? to hold our words we read in.
  std::string str; // Temp string to
 std::ifstream fin("translation.txt");
  for (int j = 0; (fin >> str) && (j < SIZE); ++j) // Will read up to eof() and stop at every
  {                                                // whitespace it hits. (like spaces!)
    words[j] = str;
  } // We now also need to stop before we hit the no define array space.
  fin.close();

std::ifstream f_r("rotationn.txt");
 std::istream_iterator<std::string> beg_r(f_r), end_r;

  const int SIZE_R = distance(beg_r, end_r) ;
  std::string words_r[SIZE_R]; // C array? to hold our words we read in.
  std::string str_r; // Temp string to
 std::ifstream fin_r("rotationn.txt");
  for (int j = 0; (fin_r >> str_r) && (j < SIZE_R); ++j) // Will read up to eof() and stop at every
  {                                                // whitespace it hits. (like spaces!)
    words_r[j] = str_r;

  } // We now also need to stop before we hit the no define array space.
  fin_r.close(); 
  
   Mx[0]=string_to_double (words_r[(9*num)]);
   Mx[1]=string_to_double (words_r[(9*num)+1]);
   Mx[2]=string_to_double (words_r[(9*num)+2]);
   Mx[3]=string_to_double (words_r[(9*num)+3]);
   Mx[4]=string_to_double (words_r[(9*num)+4]);
   Mx[5]=string_to_double (words_r[(9*num)+5]);
   Mx[6]=string_to_double (words_r[(9*num)+6]);
   Mx[7]=string_to_double (words_r[(9*num)+7]);
   Mx[8]=string_to_double (words_r[(9*num)+8]);
   Mx[9]=string_to_double (words[(3*num)]);
   Mx[10]=string_to_double (words[(3*num)+1]);
   Mx[11]=string_to_double (words[(3*num)+2]);


#endif     
     cout<<Mx[0]<<" "<<Mx[1]<<" "<<Mx[2]<<" \n"<<Mx[3]<<" "<<Mx[4]<<" "<<Mx[5]<<" \n"<<Mx[6]<<" "<<Mx[7]<<" "<<Mx[8]<<" \n"<<Mx[9]<<" "<<Mx[10]<<" "<<Mx[11]<<endl;   
  int num_of_points=0;
  for (int z = 0; z < img_rgb.rows; z++)
  { 
    uchar* rgb_ptr = img_rgb.ptr<uchar>(z);
#ifdef CUSTOM_REPROJECT
    uchar* disp_ptr = img_disparity.ptr<uchar>(z);
#else
#endif
    for (int j = 0; j < img_rgb.cols; j++)
    {
		
      //Get 3D coordinates
#ifdef CUSTOM_REPROJECT
      uchar d = disp_ptr[j];
      if ( d == 0) continue; //Discard bad pixels
      double pw = -1.0 * static_cast<double>(d) * Q32 + Q33; 
      px = static_cast<double>(j) + Q03;
      py = static_cast<double>(z) + Q13;
      pz = Q23;
      px = px/pw;
      py = py/pw;
      pz = pz/pw;



#else
   #endif
   //Get RGB info
      pb = rgb_ptr[3*j];
      pg = rgb_ptr[3*j+1];
      pr = rgb_ptr[3*j+2];

    dx=Mx[0]*px+Mx[1]*py+Mx[2]*pz+Mx[9];
    dy=Mx[3]*px+Mx[4]*py+Mx[5]*pz+Mx[10];
    dz=Mx[6]*px+Mx[7]*py+Mx[8]*pz+Mx[11];
      num_of_points++;
      //cout<<"values : "<<dx<<"  "<<dy<<"  "<<dz;
      //Insert info into point cloud structure
      pcl::PointXYZ point1;
      pcl::PointXYZRGB point;
      
      point.x = dx;point1.x=dx;
      point.y = dy;point1.y=dy;
      point.z = dz;point1.z=dz;
      uint32_t rgb = (static_cast<uint32_t>(pr) << 16 |
              static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
      point.rgb = *reinterpret_cast<float*>(&rgb);
      point_cloud_ptr->points.push_back (point);
      coloured->points.push_back (point);
     if(num==0)
	  {
		        point_cloud_ptr2->points.push_back (point);
		       }

    }
  }
 

pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
pcl::SACSegmentation<pcl::PointXYZRGB> seg;
seg.setOptimizeCoefficients(true);
seg.setModelType(pcl::SACMODEL_PLANE);
seg.setMethodType(pcl::SAC_RANSAC);
seg.setDistanceThreshold(1.5);


  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZRGB>), cloud_p (new pcl::PointCloud<pcl::PointXYZRGB>), cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);
 // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  int i = 0, nr_points = (int) coloured->points.size ();
  // While 30% of the original cloud is still there
  while (coloured->points.size () > 0)//.05 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (coloured);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the inliers
    extract.setInputCloud (coloured);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_p);
	int red,green,blue;

	for(size_t n = 0; n < cloud_p->points.size (); ++n)
{
	if(i<3)
	{
	red=0,green=0,blue=0;
	}
	if(i%3==0)
	{blue=255;
		}
		if((i%3-1)==0)
	{red=255;
		}
		if((i%3-2)==0)
	{		green=255;
		}
	cloud_p->points[n].r=red;
	cloud_p->points[n].b=green;
	cloud_p->points[n].g=blue;	}

	for(size_t f = 0; f < cloud_p->points.size (); ++f)
{    pcl::PointXYZRGB point10;
      
      point10.x = cloud_p->points[f].x;
      point10.y = cloud_p->points[f].y;
      point10.z = cloud_p->points[f].z;
      point10.rgb = cloud_p->points[f].rgb;
      custom_coloured->points.push_back (point10);
	if(num==0)
	{
		     // custom_coloured2->points.push_back (point10);

		}
	}
    //std::cerr << "PointCloud representing the planar component: " << cloud_p->points.size () << " data points." << std::endl;
 
   // writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_p, false);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    coloured.swap (cloud_f);
    
     	
    i++;
  }





 //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer5;
  //viewer5 = createVisualizer1( point_cloud_ptr );
 //while ( !viewer5->wasStopped())
  //{
    //viewer5->spinOnce(100);
    //boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  //}	
    std::stringstream ss;
    ss << "table_scene_lms400_plane_" << i << ".ply";












	
	std::cerr << "PointCloud before filtering: " << point_cloud_ptr->width * point_cloud_ptr->height 
       << " data points (" << pcl::getFieldsList (*point_cloud_ptr) << ").";

  // Create the filtering object
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud (point_cloud_ptr);
  sor.setLeafSize (5,5,5);
  sor.filter (*point_cloud_ptr1);
  
  
   std::cerr << "PointCloud after filtering: " << point_cloud_ptr1->width * point_cloud_ptr1->height 
       << " data points (" << pcl::getFieldsList (*point_cloud_ptr1) << ").";
  point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
  point_cloud_ptr->height = 1;
   custom_coloured->width = (int) custom_coloured->points.size();
  custom_coloured->height = 1;
  cout<<"num_of_points "<<num_of_points<<endl;
  cout<<"num_of_points of coloured "<<custom_coloured->width<<endl;


  
  
  //NDT
  
  
  //// Loading first scan of room.
  //pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ>);
//target_cloud=cloud_filtered;
  //// Loading second scan of room from new perspective.
  //pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  //input_cloud=point_cloud_ptr1;
  //// Filtering input scan to roughly 10% of original size to increase speed of registration.
  //pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  //pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
  //approximate_voxel_filter.setLeafSize (0.2, 0.2, 0.2);
  //approximate_voxel_filter.setInputCloud (input_cloud);
  //approximate_voxel_filter.filter (*filtered_cloud);
  //std::cout << "Filtered cloud contains " << filtered_cloud->size ()
            //<< " data points from room_scan2.pcd" << std::endl;

  //// Initializing Normal Distributions Transform (NDT).
  //pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

  //// Setting scale dependent NDT parameters
  //// Setting minimum transformation difference for termination condition.
  //ndt.setTransformationEpsilon (0.01);
  //// Setting maximum step size for More-Thuente line search.
  //ndt.setStepSize (0.1);
  ////Setting Resolution of NDT grid structure (VoxelGridCovariance).
  //ndt.setResolution (1.0);

  //// Setting max number of registration iterations.
  //ndt.setMaximumIterations (35);

  //// Setting point cloud to be aligned.
  //ndt.setInputSource (filtered_cloud);
  //// Setting point cloud to be aligned to.
  //ndt.setInputTarget (target_cloud);

  //// Set initial alignment estimate found using robot odometry.
  //Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
  //Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
  //Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

  //// Calculating required rigid transform to align the input cloud to the target cloud.
  //pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  //ndt.align (*output_cloud, init_guess);


  //std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged ()
            //<< " score: " << ndt.getFitnessScore () << std::endl;

  //// Transforming unfiltered, input cloud using found transform.
  //pcl::transformPointCloud (*input_cloud, *output_cloud, ndt.getFinalTransformation ());



//for (size_t j = 0; j < point_cloud_ptr1->points.size (); ++j)
  //{
  //pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  //kdtree.setInputCloud (cloud_filtered);

  //pcl::PointXYZ searchPoint;
  //searchPoint.x = point_cloud_ptr1->points[j].x;
  //searchPoint.y = point_cloud_ptr1->points[j].y;
  //searchPoint.z = point_cloud_ptr1->points[j].z;

   
  //std::vector<int> pointIdxRadiusSearch;
  //std::vector<float> pointRadiusSquaredDistance;

  //float radius = 1000;
  
   //std::cout << "Neighbors within radius search at (" << searchPoint.x 
            //<< " " << searchPoint.y 
            //<< " " << searchPoint.z
            //<< ") with radius=" << radius << std::endl;


  //if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
  //{
    //for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
      //std::cout << "    "  <<   cloud_filtered->points[ pointIdxRadiusSearch[i] ].x 
                //<< " " << cloud_filtered->points[ pointIdxRadiusSearch[i] ].y 
                //<< " " << cloud_filtered->points[ pointIdxRadiusSearch[i] ].z 
                //<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
  
  //}
 //}
 pcl::PointCloud<pcl::PointXYZRGB>::Ptr test (new pcl::PointCloud<pcl::PointXYZRGB>);

 int radius;
 int check;
 int point_to_remove;
  for (size_t j = 0; j < point_cloud_ptr1->points.size (); ++j)
  {
	    pcl::PointXYZ searchPoint;
  searchPoint.x = point_cloud_ptr1->points[j].x;
  searchPoint.y = point_cloud_ptr1->points[j].y;
  searchPoint.z = point_cloud_ptr1->points[j].z;
	  if(num==0)
	  {}
	  else{
		    srand (time (NULL));

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  // Generate pointcloud data

  cloud->width = (int) point_cloud_ptr1->points.size();
  cloud->height = 1;
  cloud->points.resize (cloud->width * cloud->height);

  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    cloud->points[i].x = point_cloud_ptr1->points[i].x;
    cloud->points[i].y = point_cloud_ptr1->points[i].y;
    cloud->points[i].z = point_cloud_ptr1->points[i].z;
  }
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  kdtree.setInputCloud (cloud);


  
int K = 1;


  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);

  //std::cout << "K nearest neighbor search at (" << searchPoint.x 
            //<< " " << searchPoint.y 
            //<< " " << searchPoint.z
            //<< ") with K=" << K << std::endl;

 if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
  {
    for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
    {
      //std::cout << "    "  <<   cloud->points[ pointIdxNKNSearch[i] ].x 
                //<< " " << cloud->points[ pointIdxNKNSearch[i] ].y 
                //<< " " << cloud->points[ pointIdxNKNSearch[i] ].z 
                //<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
               radius= pointNKNSquaredDistance[i];
               point_to_remove=pointIdxNKNSearch[i];
               
}
  }
   
  //std::vector<int> pointIdxRadiusSearch;
  //std::vector<float> pointRadiusSquaredDistance;

  //float radius = 2;
  
   //std::cout << "Neighbors within radius search at (" << searchPoint.x 
            //<< " " << searchPoint.y 
            //<< " " << searchPoint.z
            //<< ") with radius=" << radius << std::endl;


  //if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
  //{
    //for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
      //std::cout << "    "  <<   point_cloud_ptr2->points[ pointIdxRadiusSearch[i] ].x 
                //<< " " << point_cloud_ptr2->points[ pointIdxRadiusSearch[i] ].y 
                //<< " " << point_cloud_ptr2->points[ pointIdxRadiusSearch[i] ].z 
                //<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
  
  //}
  }
      //pcl::PointXYZ point2;
      //point2.x=point_cloud_ptr1->points[j].x;
      //point2.y=point_cloud_ptr1->points[j].y;
      //point2.z=point_cloud_ptr1->points[j].z;
      int num1;
      if(radius<=4)
      {
	num1++;

    cloud_filtered->points.push_back (searchPoint);

	for(size_t l = 0; l < point_cloud_ptr2->points.size (); ++l)
	{
	  pcl::PointXYZRGB point3;
      point3.x=point_cloud_ptr2->points[l].x;
      point3.y=point_cloud_ptr2->points[l].y;
      point3.z=point_cloud_ptr2->points[l].z;
      //point3.rgb=point_cloud_ptr->points[l].rgb;
     
		if( (cloud_filtered->points[point_to_remove].x-2)<point3.x && point3.x<(cloud_filtered->points[point_to_remove].x+2))
	   {if( (cloud_filtered->points[point_to_remove].y-2)<point3.y && point3.y<(cloud_filtered->points[point_to_remove].y+2))
	   {if( (cloud_filtered->points[point_to_remove].z-2)<point3.z && point3.z<(cloud_filtered->points[point_to_remove].z+2))
			{//std::cout<<cloud_filtered->points[point_to_remove].x-1<<std::endl;
			//test->points.push_back (point3);
						//std::cout<<point_cloud_ptr2->points[l]<<std::endl;
			point_cloud_ptr2->points[l].x=0;
			point_cloud_ptr2->points[l].y=1;
			point_cloud_ptr2->points[l].z=0;
			//std::cout<<point_cloud_ptr2->points[l]<<std::endl;
			
			check++;
			}
			}
			}
			
		}       
	    
	for(size_t l = 0; l < custom_coloured2->points.size (); ++l)
	{
	  pcl::PointXYZRGB point3;
      point3.x=custom_coloured2->points[l].x;
      point3.y=custom_coloured2->points[l].y;
      point3.z=custom_coloured2->points[l].z;
      //point3.rgb=point_cloud_ptr->points[l].rgb;
     
		if( (cloud_filtered->points[point_to_remove].x-2)<point3.x && point3.x<(cloud_filtered->points[point_to_remove].x+2))
	   {if( (cloud_filtered->points[point_to_remove].y-2)<point3.y && point3.y<(cloud_filtered->points[point_to_remove].y+2))
	   {if( (cloud_filtered->points[point_to_remove].z-2)<point3.z && point3.z<(cloud_filtered->points[point_to_remove].z+2))
			{
			custom_coloured2->points[l].x=0;
			custom_coloured2->points[l].y=1;
			custom_coloured2->points[l].z=0;
			//custom_coloured2->points.push_back (point4);
			check++;
			}
			}
			}
			
		}       
	  
	  }
	  if(num==0)
	  {
		            cloud_filtered->points.push_back (searchPoint);
  }
  //std::cout<<"2222222222222222222222-" <<check<<std::endl; 

  }
  //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer12;
  //viewer12 = createVisualizer1(  custom_coloured2);
    
  //while ( !viewer12->wasStopped())
  //{
    //viewer12->spinOnce(100);
    //boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  //}	


    for (size_t m = 0; m < point_cloud_ptr->points.size (); ++m)
{
  pcl::PointXYZRGB addtoworldframe;
  addtoworldframe.x = point_cloud_ptr->points[m].x;
  addtoworldframe.y = point_cloud_ptr->points[m].y;
  addtoworldframe.z = point_cloud_ptr->points[m].z;	
  addtoworldframe.rgb=point_cloud_ptr->points[m].rgb;
  
    point_cloud_ptr2->points.push_back(addtoworldframe);
  
  pcl::PointXYZRGB addtoworldframe_custom;
  addtoworldframe_custom.x = custom_coloured->points[m].x;
  addtoworldframe_custom.y = custom_coloured->points[m].y;
  addtoworldframe_custom.z = custom_coloured->points[m].z;	
  addtoworldframe_custom.rgb=custom_coloured->points[m].rgb;
  custom_coloured2->points.push_back(addtoworldframe_custom);
	}
  


  
  
  
  
  
  
  
  
  point_cloud_ptr->points.clear();
    custom_coloured->points.clear();
  point_cloud_ptr1->points.clear();
coloured->points.clear();
  t2 = getTickCount() - t2;
  printf("Time elapsed for a loop: %fms\n", t2*1000/getTickFrequency());
 
	}
	 cloud_filtered->width = (int) cloud_filtered->points.size();
  cloud_filtered->height = 1;  
	 point_cloud_ptr2->width = (int) point_cloud_ptr2->points.size();
  point_cloud_ptr2->height = 1; 
  custom_coloured2->width = (int) custom_coloured2->points.size();
  custom_coloured2->height = 1;  
   //Create visualizer
   bool binary = true;

  pcdWriter.write("test_pcd.pcd",*cloud_filtered);
  pcdWriter.write("test_pcd2.pcd",*point_cloud_ptr2);
  //sensor_msgs::PointCloud2 cloud1;
  //loadPCDFile ("test_pcd2.pcd", cloud1);
  //pcl::io::loadPCDFile("test_pcd2.pcd", *cloud1);
  writer.write<pcl::PointXYZ> ("test_pcd.ply", *cloud_filtered, binary);
  //Main loop
   pcdWriter.write<pcl::PointXYZRGB> ("complete_3d.pcd", *point_cloud_ptr2, false);
     writer.write<pcl::PointXYZRGB> ("complete_3d.ply", *point_cloud_ptr2, false);
  writer.write<pcl::PointXYZRGB> ("custom_colored.ply", *custom_coloured2, false);

//pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
  //pcl::search::Search<pcl::PointXYZRGB>::Ptr tree1 = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);
  //pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  //normal_estimator.setSearchMethod (tree1);
  //normal_estimator.setInputCloud (point_cloud_ptr2);
  //normal_estimator.setKSearch (20);
  //normal_estimator.compute (*normals);
	//for(size_t t = 0; t < normals->points.size (); ++t)
//{
	//cout<<normals->points[t].normal_x<<endl;
	//}







boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  viewer = createVisualizer1(  custom_coloured2);
    
  while ( !viewer->wasStopped())
  {
    viewer->spinOnce(100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }	
 //while ( !viewer3->wasStopped())
  //{
    //viewer3->spinOnce(100);
    //boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  //}	
  return 0;  
}
