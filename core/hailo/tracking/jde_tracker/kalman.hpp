/*
  A simple Kalman filter for tracking bounding objects in image space.

  The 4-dimensional state space: x, Vy, y, Vy

  contains the bounding box center position (x, y) and  their respective velocities.

  Object motion follows a constant velocity model. The bounding box location
  (x, y) is taken as direct observation of the state space (linear observation model).

 */

#pragma once
#include <opencv2/opencv.hpp>

namespace ocv_kalman
{
class kalman
{
private:
	bool _initialized = false;

	cv::KalmanFilter _k;
	cv::Mat _state;
	cv::Mat _measurement;
	int _stateSize ; // Constant Velocity Model: x,vx,y,vy ,Acceleration Model : x,vx,ax,y,vy,ay
	int _measurementSize = 2; //zx,zy
	int _controlSize = 0;
	int _type = CV_32F;
        int _mode = 1;
public:
	kalman()
	{
	}
			
	~kalman()
	{
	}
	
	void Init(int iMode,cv::Rect2f bbox,float stateCov_x,float stateCov_Vx,float stateCov_Ax,float measureCov_zx)
	{
		if (!_initialized)
		{
			cv::Point2f icentre;
			icentre.x = bbox.x + bbox.width / 2.0;
			icentre.y = bbox.y + bbox.height / 2.0;
			if (icentre.x != 0 || icentre.y != 0) //this ensures valid input
			{
                                _mode=iMode;
				cv::Mat A;
				cv::Mat H;
				cv::Mat P;
				cv::Mat Q;
				cv::Mat R;

           			switch(_mode)
				{
				    case 1:
					_stateSize = 4;
					_measurementSize = 2;

					//state vector
					A = cv::Mat::zeros(cv::Size(_stateSize, _stateSize), _type);
					A.at<float>(0)  = 1.0f;
					A.at<float>(1)  = 1.0f;
					A.at<float>(5)  = 1.0f;
					A.at<float>(10) = 1.0f;
					A.at<float>(11) = 1.0f;
					A.at<float>(15) = 1.0f;

					//measurement vector
					H = cv::Mat::zeros(cv::Size(_stateSize, _measurementSize), _type);
					H.at<float>(0) = 1.0f;
					H.at<float>(6) = 1.0f;

					//state uncertinity matrix
					P = cv::Mat::zeros(cv::Size(_stateSize, _stateSize), _type);
					P.at<float>(0)  = 10e5f;
					P.at<float>(5)  = 10e5f;
					P.at<float>(10) = 10e5f;
					P.at<float>(15) = 10e5f;

					//state covariance matrix -Note: covariance coefficents are same for x,y and also same for Vx and Vy
					Q = cv::Mat::zeros(cv::Size(_stateSize, _stateSize), _type);
					Q.at<float>(0)  = stateCov_x;  //25.0f
					Q.at<float>(5)  = stateCov_Vx; //10.0f
					Q.at<float>(10) = stateCov_x;  //25.0f 
					Q.at<float>(15) = stateCov_Vx; //10.0f

					//measurement covariance matrix -Note: covariance coefficents are same for zx,zy
					R = cv::Mat::zeros(cv::Size(_measurementSize, _measurementSize), _type);
					R.at<float>(0) = measureCov_zx; //25000.0f
					R.at<float>(3) = measureCov_zx; //25000.0f

                                      	break;
                                    case 2:
					_stateSize = 6;
					_measurementSize = 2;

					A = Mat::zeros(Size(_stateSize, _stateSize), _type);
					A.at<float>(0)  = 1.0f;
					A.at<float>(1)  = 1.0f;
					A.at<float>(2)  = 0.5f;
					A.at<float>(7)  = 1.0f;
					A.at<float>(8)  = 1.0f;
					A.at<float>(14) = 1.0f;
					A.at<float>(21) = 1.0f;
					A.at<float>(22) = 1.0f;
					A.at<float>(23) = 0.5f;
					A.at<float>(28) = 1.0f;
					A.at<float>(29) = 1.0f;
					A.at<float>(35) = 1.0f;

					H = Mat::zeros(Size(_stateSize, _measurementSize), _type);
					H.at<float>(0) = 1.0f;
					H.at<float>(9) = 1.0f;

					P = Mat::zeros(Size(_stateSize, _stateSize), _type);
					P.at<float>(0)  = 10e5f;
					P.at<float>(7)  = 10e5f;
					P.at<float>(14) = 10e5f;
					P.at<float>(21) = 10e5f;
					P.at<float>(28) = 10e5f;
					P.at<float>(35) = 10e5f;

					Q = Mat::zeros(Size(_stateSize, _stateSize), _type);
					Q.at<float>(0)  = stateCov_x;  //25.0f;
					Q.at<float>(7)  = stateCov_Vx; //10.0f;
					Q.at<float>(14) = stateCov_Ax; //1.0f;
					Q.at<float>(21) = stateCov_x;  //25.0f;
					Q.at<float>(28) = stateCov_Vx; //10.0f;
					Q.at<float>(35) = stateCov_Ax; //1.0f;

					R = Mat::zeros(Size(_measurementSize, _measurementSize), _type);
					R.at<float>(0) =  measureCov_zx; //25000.0f;
					R.at<float>(3) =  measureCov_zx; //25000.0f;
                                        break;
                                    default:
				        break;
				}

				_k.init(_stateSize, _measurementSize, _controlSize, _type);

				_state = cv::Mat::zeros(_stateSize, 1, _type);
				_measurement = cv::Mat::zeros(_measurementSize, 1, _type);

				A.copyTo(_k.transitionMatrix);
				H.copyTo(_k.measurementMatrix);
				P.copyTo(_k.errorCovPre);
				Q.copyTo(_k.processNoiseCov);
				R.copyTo(_k.measurementNoiseCov);
		
				//initialize state and measurement matrix
				_measurement.at<float>(0) = icentre.x;
				_measurement.at<float>(1) = icentre.y;

				switch (_mode)
				{
				   case 1:
					_state.at<float>(0) = _measurement.at<float>(0);
					_state.at<float>(1) = 0;
					_state.at<float>(2) = _measurement.at<float>(1);
					_state.at<float>(3) = 0;
					break;
				   case 2:
					_state.at<float>(0) = _measurement.at<float>(0);
					_state.at<float>(1) = 0;
					_state.at<float>(2) = 0;
					_state.at<float>(3) = _measurement.at<float>(1);
					_state.at<float>(4) = 0;
					_state.at<float>(5) = 0;
					break;
				   default:
					break;
				}

				_k.statePost = _state;
				_initialized = true;
			}
			else
			{
				printf("ERROR:kalman : invalid input \n");
			}
		}
		else
		{
			printf("ERROR:kalman: re-initlization not possible \n");
		}

	}

	void Predict(cv::Rect2f& bbox, bool predict)
	{
		if (_initialized)
		{
			cv::Point2f icentre, ocentre;
			icentre.x = bbox.x + bbox.width / 2.0;
			icentre.y = bbox.y + bbox.height / 2.0;

			try
			{

				/*Note: case 1: measueremnt available. case 2: measuremnt not available*/
				_state = _k.predict();

				if (predict == false) //Note:  measuremnt is available
				{
					_measurement.at<float>(0) = icentre.x;
					_measurement.at<float>(1) = icentre.y;

					_k.correct(_measurement);
				}
				else
				{
					_measurement.at<float>(0) = icentre.x;
					_measurement.at<float>(1) = icentre.y;
				}
				switch (_mode)
				{
				    case 1:
					ocentre.x = _state.at<float>(0);
					ocentre.y = _state.at<float>(2);
					break;
				    case 2:
					ocentre.x = _state.at<float>(0);
					ocentre.y = _state.at<float>(3);
					break;
				    default:
					break;
				}

				bbox.x = ocentre.x - bbox.width / 2.0;
				bbox.y = ocentre.y - bbox.height / 2.0;
			}
			catch (const std::exception& e)
			{
				std::cout << "Exception: kalman " << std::endl;
				return;
			}
		}
		else
		{
			std::cout << "ERROR uninitialized" << std::endl;
		}
	}
};
}
