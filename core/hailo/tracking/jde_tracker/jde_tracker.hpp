/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/*
  Class header for Joint Detection and Embedding (JDE) model with Kalman filtering to track object instances.
*/

#pragma once

// General cpp includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Tappas includes
#include "hailo_objects.hpp"
#include "kalman.hpp"
#include "kalman_filter.hpp"
#include "lapjv.hpp"
#include "strack.hpp"
#include "tracker_macros.hpp"

//shm include
#include<stdio.h>
#include<sys/ipc.h>
#include<sys/shm.h>
#include<sys/types.h>
#include<errno.h>
#include<stdlib.h>

#define DEFAULT_KALMAN_DISTANCE (0.7f)
#define DEFAULT_IOU_THRESHOLD (0.8f)
#define DEFAULT_INIT_IOU_THRESHOLD (0.9f)
#define DEFAULT_KEEP_FRAMES (2)
#define DEFAULT_KEEP_PAST_METADATA (true)
#define DEFAULT_STD_WEIGHT_POSITION (0.01)
#define DEFAULT_STD_WEIGHT_POSITION_BOX (0.00000001)
#define DEFAULT_STD_WEIGHT_VELOCITY (0.001)
#define DEFAULT_STD_WEIGHT_VELOCITY_BOX (0.00000001)
#define DEFAULT_DEBUG (false)

__BEGIN_DECLS


typedef enum
{
   PERSONS = 0,
   VEHICLES = 1
}MVIGS_ObjectDetectionClassType;

struct MVIGS_ObjectTrackingResultsType
{
   int cX;
   int cY;
   int width;
   int height;
   MVIGS_ObjectDetectionClassType classtype;
   int trackID;
   TrackState trackState;
};

#define SHM_KEY 0x1234
struct shmseg {
   MVIGS_ObjectTrackingResultsType _sot_track;
   int _selectedTarget=-1;
   bool _bValidTrack=false;
   MVIGS_ObjectTrackingResultsType _tracks[MAX_NUM_TRACKS];
   unsigned int _numTracks=0;
   float _custom_lambda=0.98;
   float _fakeThreshold=0.90;
   float _byte_track_thresh=-1;
   unsigned int _model_input_size_x=1280; //1280
   unsigned int _model_input_size_y=768; //768
   int _detect_counter=0;
   float _iou_scale_factor1 = 0.5;
   float _iou_scale_factor2 = 1.0;
   bool _iou_scale1_enable  = false;
   bool _iou_scale2_enable  = false;
   int _sot_method = 1;
   float _predictable_region = 0.9;
   float _stateCov_x;
   float _stateCov_Vx;
   float _stateCov_Ax;
   float _measureCov_zx;
   int _kalman_mode;
};
class JDETracker
{
    //******************************************************************
    // CLASS MEMBERS
    //******************************************************************
private:
    float m_kalman_dist_thr;   // threshold used for kalman tracker, bigger is looser
    float m_iou_thr;           // threshold used for iou tracker, bigger is looser
    float m_init_iou_thr;      // threshold used for iou tracker for new detections, bigger is looser
    int m_keep_tracked_frames; // number of frames to keep tracking w/o detection
    int m_keep_new_frames;     // number of frames to keep new detections w/o detection
    int m_keep_lost_frames;    // number of frames to keep lost detections w/o detection
    int m_keep_predict_frames;  // number of frames to keep predicting w/o detection
    bool m_keep_past_metadata; // keep past metadata for new detections
    int m_frame_id{0};         // the current frame id
    bool m_debug;              // debug flag to ebable output new and lost tracks
    int m_sot_counter{0};

    std::vector<STrack> m_tracked_stracks;                 // Currently tracked STracks
    std::vector<STrack> m_lost_stracks;                    // Currently lost STracks
    std::vector<STrack> m_new_stracks;                     // Currently new STracks
    KalmanFilter m_kalman_filter;                          // Kalman Filter
    std::vector<hailo_object_t> m_hailo_objects_blacklist; // Objects that will never be kept track of

    int m_shmid;		//shared memory id
    struct shmseg *m_shmp;	//shared memory data

    //******************************************************************
    // CLASS RESOURCE MANAGEMENT
    //******************************************************************
public:
    // Default Constructor
    JDETracker(float kalman_dist = DEFAULT_KALMAN_DISTANCE, float iou_thr = DEFAULT_IOU_THRESHOLD,
               float init_iou_thr = DEFAULT_INIT_IOU_THRESHOLD, int keep_tracked = DEFAULT_KEEP_FRAMES,
               int keep_new = DEFAULT_KEEP_FRAMES, int keep_lost = DEFAULT_KEEP_FRAMES,int keep_predict =  DEFAULT_KEEP_FRAMES,
               bool keep_past_metadata = DEFAULT_KEEP_PAST_METADATA, float std_weight_position = DEFAULT_STD_WEIGHT_POSITION,
               float std_weight_position_box = DEFAULT_STD_WEIGHT_POSITION_BOX, float std_weight_velocity = DEFAULT_STD_WEIGHT_VELOCITY,
               float std_weight_velocity_box = DEFAULT_STD_WEIGHT_VELOCITY_BOX, bool debug = DEFAULT_DEBUG,
               std::vector<hailo_object_t> hailo_objects_blacklist_vec = {HAILO_LANDMARKS, HAILO_DEPTH_MASK, HAILO_CLASS_MASK}) : m_kalman_dist_thr(kalman_dist), m_iou_thr(iou_thr), m_init_iou_thr(init_iou_thr),
                                                                                                                                  m_keep_tracked_frames(keep_tracked), m_keep_new_frames(keep_new), m_keep_lost_frames(keep_lost),m_keep_predict_frames(keep_predict),
                                                                                                                                  m_keep_past_metadata(keep_past_metadata), m_debug(debug), m_hailo_objects_blacklist(hailo_objects_blacklist_vec)
    {
        m_kalman_filter = KalmanFilter(std_weight_position, std_weight_position_box, std_weight_velocity, std_weight_velocity_box);

	//Shared memory
        m_shmid = shmget(SHM_KEY, sizeof(struct shmseg), 0644|IPC_CREAT);
        if (m_shmid == -1) 
	{
            perror("JDETracker | Shared memory create error\n");
        }

        m_shmp = (shmseg*)shmat(m_shmid, NULL, 0);
        
	if (m_shmp == (void *) -1)
        {
            perror("JDETracker | Shared memory attach error\n");
        }
	else
	{
	    printf("JDETracker Shared memory attach success\n");
	}
	printf("HailoTracker: tracked,lost,new,predicted:%d,%d,%d,%d\n",m_keep_tracked_frames,m_keep_lost_frames,m_keep_new_frames,m_keep_predict_frames);
	printf("HailoTracker: init_iou_thr,iou_thr,fakeThreshold:%f,%f,%f\n", m_init_iou_thr,m_iou_thr,m_shmp->_fakeThreshold);
	printf("HailoTracker:  _iou_scale_factor1, _iou_scale_factor2,_iou_scale1_enable,_iou_scale2_enable :%f,%f,%d,%d\n", m_shmp-> _iou_scale_factor1,m_shmp-> _iou_scale_factor2,(int)m_shmp->_iou_scale1_enable,(int)m_shmp->_iou_scale2_enable);
	printf("HailoTracker: predictable region: %f\n",m_shmp->_predictable_region);
	printf("HailoTracker: state covariance,measurement covaraiance coefficients: %d,%f,%f,%f,%f\n",m_shmp->_kalman_mode,m_shmp->_stateCov_x,m_shmp->_stateCov_Vx,m_shmp->_stateCov_Ax,m_shmp->_measureCov_zx);
	printf("HailoTracker: pos,box_pos,vel,box,vel: %f,%f,%f.%f\n",std_weight_position, std_weight_position_box, std_weight_velocity, std_weight_velocity_box);

    }

    // Destructor
    //~JDETracker() = default;
    ~JDETracker()
    {
	//detach shared memory
        if (shmdt(m_shmp) == -1)
        {
            perror("JDETracker | Shared memory detach error\n");
        }
    }

    //******************************************************************
    // CLASS MEMBER ACCESS
    //******************************************************************
public:
    // Setters for members accessible at element-property level
    void set_kalman_distance(float new_distance) { m_kalman_dist_thr = new_distance; }
    void set_iou_threshold(float new_iou_thr) { m_iou_thr = new_iou_thr; }
    void set_init_iou_threshold(float new_init_iou_thr) { m_init_iou_thr = new_init_iou_thr; }
    void set_keep_tracked_frames(int new_keep_tracked) { m_keep_tracked_frames = new_keep_tracked; }
    void set_keep_new_frames(int new_keep_new) { m_keep_new_frames = new_keep_new; }
    void set_keep_lost_frames(int new_keep_lost) { m_keep_lost_frames = new_keep_lost; }
    void set_keep_predict_frames(int new_keep_predict) { m_keep_predict_frames = new_keep_predict; }
    void set_keep_past_metadata(bool new_keep_past_metadata) { m_keep_past_metadata = new_keep_past_metadata; }

    void set_std_weight_position(float std_weight_position) { m_kalman_filter.set_std_weight_position(std_weight_position); }
    void set_std_weight_position_box(float std_weight_position_box) { m_kalman_filter.set_std_weight_position_box(std_weight_position_box); }
    void set_std_weight_velocity(float std_weight_velocity) { m_kalman_filter.set_std_weight_velocity(std_weight_velocity); }
    void set_std_weight_velocity_box(float std_weight_velocity_box) { m_kalman_filter.set_std_weight_velocity_box(std_weight_velocity_box); }
    void set_debug(bool debug) { m_debug = debug; }
    void set_hailo_objects_blacklist(std::vector<hailo_object_t> hailo_objects_blacklist) { m_hailo_objects_blacklist = hailo_objects_blacklist; }

    // Getters for members accessible at element-property level
    float get_kalman_distance() { return m_kalman_dist_thr; }
    float get_iou_threshold() { return m_iou_thr; }
    float get_init_iou_threshold() { return m_init_iou_thr; }
    int get_keep_tracked_frames() { return m_keep_tracked_frames; }
    int get_keep_new_frames() { return m_keep_new_frames; }
    int get_keep_lost_frames() { return m_keep_lost_frames; }
    int get_keep_predict_frames() { return m_keep_predict_frames; }
    bool get_keep_past_metadata() { return m_keep_past_metadata; }
    float get_std_weight_position() { return m_kalman_filter.get_std_weight_position(); }
    float get_std_weight_position_box() { return m_kalman_filter.get_std_weight_position_box(); }
    float get_std_weight_velocity() { return m_kalman_filter.get_std_weight_velocity(); }
    float get_std_weight_velocity_box() { return m_kalman_filter.get_std_weight_velocity_box(); }
    bool get_debug() { return m_debug; }
    std::vector<hailo_object_t> get_hailo_objects_blacklist() { return m_hailo_objects_blacklist; }

    //******************************************************************
    // TRACKING FUNCTIONS
    //******************************************************************
    /******************** PUBLIC FUNCTIONS ****************************/
public:
    static std::vector<STrack> hailo_detections_to_stracks(std::vector<HailoDetectionPtr> &inputs, int frame_id, std::vector<hailo_object_t> hailo_objects_blacklist);
    static std::vector<HailoDetectionPtr> stracks_to_hailo_detections(std::vector<STrack> &stracks, bool debug);
    STrack *get_detection_with_id(int track_id);
    std::vector<STrack> get_tracked_stracks();
    std::vector<STrack> update(std::vector<HailoDetectionPtr> &inputs, bool report_unconfirmed, bool report_lost);

    /******************** PRIVATE FUNCTIONS ****************************/
private:
    void update_trackmode(std::vector<STrack> &stracksa,std::vector<STrack> &stracksb,std::vector<STrack> &stracksc);
    void update_unmatches(std::vector<STrack *> strack_pool, std::vector<STrack> &tracked_stracks, std::vector<STrack> &lost_stracks, std::vector<STrack> &new_stracks);
    void update_matches(std::vector<std::pair<int, int>> matches, std::vector<STrack *> tracked_stracks, std::vector<STrack> &detections, std::vector<STrack> &activated_stracks);
    void linear_assignment(std::vector<std::vector<float>> &cost_matrix, int cost_matrix_rows, int cost_matrix_cols, float thresh, std::vector<std::pair<int, int>> &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);

    std::vector<std::vector<float>> iou_distance(std::vector<STrack *> &atracks, std::vector<STrack> &btracks);
    std::vector<std::vector<float>> iou_distance_custom(std::vector<STrack *> &atracks, std::vector<STrack> &btracks,float scale);
    std::vector<std::vector<float>> iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);
    std::vector<std::vector<float>> ioma_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);


    std::vector<STrack *> joint_strack_pointers(std::vector<STrack *> &tlista, std::vector<STrack *> &tlistb);
    std::vector<STrack *> joint_strack_pointers(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
    std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
    std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
    void remove_duplicate_stracks(std::vector<STrack> &stracksa, std::vector<STrack> &stracksb);
    void remove_duplicate_stracks_custom(std::vector<STrack> &stracksa,float iou_thresh);
    void remove_duplicate_detections_custom(std::vector<STrack> &set_a, std::vector<STrack> &set_b, float iou_thresh);
    void remove_duplicates_within_set(std::vector<STrack> &detections, float ioma_thresh);

    void embedding_distance(std::vector<STrack *> &tracks, std::vector<STrack> &detections, std::vector<std::vector<float>> &cost_matrix);
    void fuse_motion(std::vector<std::vector<float>> &cost_matrix, std::vector<STrack *> &tracks, std::vector<STrack> &detections, float lambda_);
    void fuse_motion_custom(std::vector<std::vector<float>> &cost_matrix, std::vector<STrack *> &tracks, std::vector<STrack> &detections);

};
__END_DECLS

// Class Definitions
// This class is split across multiple files to isolate definitions
// and maintain readability. This can be done in a header-only fashion
// by keeping the class/function declarations in this root file
// while the defintions are spread amongs the following includes.
#include "jde_tracker_lapjv.hpp"
#include "jde_tracker_ious.hpp"
#include "jde_tracker_embedding.hpp"
#include "jde_tracker_strack_management.hpp"
#include "jde_tracker_converters.hpp"
#include "jde_tracker_update.hpp"
