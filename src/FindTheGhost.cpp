#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

int main() {
	Mat train = imread("../ghost.png"), train_g;
	cvtColor(train, train_g, CV_BGR2GRAY);
	threshold(train_g,train_g,230,255,THRESH_BINARY);

	vector<KeyPoint> train_kp;
	Mat train_desc;

	//feature point stuff
	SurfFeatureDetector featureDetector(100);
	featureDetector.detect(train_g, train_kp);
	SurfDescriptorExtractor featureExtractor;
	featureExtractor.compute(train_g, train_kp, train_desc);

	// matching stuff
	FlannBasedMatcher matcher;
	vector<Mat> train_desc_collection(1, train_desc);
	matcher.add(train_desc_collection);
	matcher.train();

	Mat test = imread("../snaptcha.png"), test_g;
	//I need to split this into 9 images.
	int h_div = test.rows / 3, w_div = test.cols / 3;
	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			cout << i << ", " << j;
			Mat subimg = test(Rect(i*w_div,j*h_div,w_div,h_div));
			cvtColor(subimg, test_g, CV_BGR2GRAY);
			threshold(test_g,test_g,230,255,THRESH_BINARY);
			//feature stuff again
			vector<KeyPoint> test_kp;
			Mat test_desc;
			featureDetector.detect(test_g, test_kp);
			featureExtractor.compute(test_g, test_kp, test_desc);

			// get neighbors
			vector<vector<DMatch> > matches;
			matcher.knnMatch(test_desc, matches, 2);

			// filter for good matches
			vector<DMatch> good_matches;
			vector<Point2f> obj, scene;
			for(int i = 0; i < matches.size(); i++) {
				if(matches[i][0].distance < 0.4 * matches[i][1].distance) {
					good_matches.push_back(matches[i][0]);
					obj.push_back( train_kp[ matches[i][0].trainIdx ].pt );
					scene.push_back(test_kp[ matches[i][0].queryIdx ].pt );
				}
			}

			vector<Point2f> unique;
			for(int i = 0; i < obj.size(); i++) {
				if(std::find(unique.begin(), unique.end(), obj[i])==unique.end())
					unique.push_back(obj[i]);
			}
			float uniqueness = float(unique.size()) / float(obj.size());

			//Detect whether there is a ghost or not
			if(good_matches.size() > 10)
				cout << ": A ghost was found" << endl;
			else if(good_matches.size() < 2 || uniqueness < 0.5f)
				cout << ": No ghost found" << endl;
			else
				cout << ": A ghost was found" << endl;


			Mat img_show;
			drawMatches(subimg, test_kp, train, train_kp, good_matches, img_show);
			imshow("Matches", img_show);
			cvWaitKey();
		}
	}
	return 0;
}
