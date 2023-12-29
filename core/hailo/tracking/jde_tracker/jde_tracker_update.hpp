/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
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
#include "strack.hpp"
#include "tracker_macros.hpp"

/**
 * @brief Keep specific indices from an input vector of stracks.
 *
 * @param stracks  -  std::vector<STrack>
 *        The stracks to keep from.
 *
 * @param indices  -  std::vector<int>
 *        The indices to keep.
 */
inline void keep_indices(std::vector<STrack> &stracks, const std::vector<int> &indices)
{
    std::vector<STrack> stracks_swap;
    stracks_swap.reserve(indices.size());
    for (uint i = 0; i < indices.size(); i++)
    {
        if (indices[i] < (int)stracks.size())
            stracks_swap.emplace_back(stracks[indices[i]]);
    }
    stracks = stracks_swap;
}

/**
 * @brief Keep specific indices from an input vector of stracks.
 *
 * @param stracks  -  std::vector<STrack *>
 *        The stracks (by pointer) to keep from.
 *
 * @param indices  -  std::vector<int>
 *        The indices to keep.
 */
inline void keep_indices(std::vector<STrack *> &stracks, const std::vector<int> &indices)
{
    std::vector<STrack *> stracks_swap;
    stracks_swap.reserve(indices.size());
    for (uint i = 0; i < indices.size(); i++)
    {
        if (indices[i] < (int)stracks.size())
            stracks_swap.emplace_back(stracks[indices[i]]);
    }
    stracks = stracks_swap;
}

/**
 * @brief Update the matched tracklets, activating and
 *        reactivating as necessary.
 *
 * @param matches  -  std::vector<std::pair<int,int>>
 *        Pairs of matches, generated by linear assignment.
 *
 * @param tracked_stracks  -  std::vector<STrack *>
 *        The tracked stracks (by pointer).
 *
 * @param detections  -  std::vector<STrack>
 *        The detected objects.
 *
 * @param activated_stracks  - std::vector<STrack>
 *        The currently active stracks. All matched stracks
 *        will be added here.
 */
inline void JDETracker::update_matches(std::vector<std::pair<int, int>> matches,
                                       std::vector<STrack *> tracked_stracks,
                                       std::vector<STrack> &detections,
                                       std::vector<STrack> &activated_stracks)
{
    for (uint i = 0; i < matches.size(); i++)
    {
        if ((tracked_stracks.size() == 0) || (detections.size() == 0))
            continue;
        STrack *track = tracked_stracks[matches[i].first];
        STrack *det = &detections[matches[i].second];
        switch (track->get_state())
        {
        case TrackState::Tracked: // The tracklet was already tracked, so update
            track->update(*det, this->m_frame_id, this->m_keep_past_metadata);
            break;
        case TrackState::Lost: // The tracklet was lost but found, so re-activate
            track->re_activate(*det, this->m_frame_id, false, this->m_keep_past_metadata);
            break;
        case TrackState::New: // The tracklet is brand new, so activate
            track->activate(&this->m_kalman_filter, this->m_frame_id);
            break;
        }
        activated_stracks.push_back(*track);
    }
}

/**
 * @brief Update the tracklets that did not get matched.
 *        The state of the tracklets is updated here as needed.
 *        If a tracklet has overstayed its time in a state (keep_time),
 *        then it may be moved to another state.
 *        Example: If a tracked object has been unmatched for more than
 *                 m_keep_tracked_frames, then it will be marked lost
 *                 and moved to the list of lost_stracks
 *
 * @param strack_pool  -  std::vector<STrack *>
 *        The pool of unmatched stracks.
 *
 * @param tracked_stracks  -  std::vector<STrack>
 *        The list of tracked stracks.
 *
 * @param lost_stracks  -  std::vector<STrack>
 *        The list of lost stracks.
 *
 * @param new_stracks  -  std::vector<STrack>
 *        The list of new stracks.
 *
 */
inline void JDETracker::update_unmatches(std::vector<STrack *> strack_pool,
                                         std::vector<STrack> &tracked_stracks,
                                         std::vector<STrack> &lost_stracks,
                                         std::vector<STrack> &new_stracks)
{
    for (uint i = 0; i < strack_pool.size(); i++)
    {
        STrack *track = strack_pool[i];
        switch (track->get_state())
        {
        case TrackState::Tracked:
            if (this->m_frame_id - track->end_frame() < this->m_keep_tracked_frames)
            {
		if (this->m_frame_id - track->end_frame() < this->m_keep_predict_frames)
		{
			track->m_kalman.Predict(track->m_kalman_rect, true);//prediction
			track->m_kalman.Predict(track->m_kalman_rect, true);//prediction
			track->m_tlwh[0] = track->m_kalman_rect.x;
			track->m_tlwh[1] = track->m_kalman_rect.y;
			//printf("tracked : predicted:%d\n",m_keep_predict_frames);
			//printf("tracked : predicted:%f,%f,%f,%f\n",track->m_tlwh[0],track->m_tlwh[1],track->m_tlwh[2],track->m_tlwh[3]);

		}
                tracked_stracks.push_back(*track); // Not over threshold, so still tracked
            }
            else
            {
                track->mark_lost();
                lost_stracks.push_back(*track); // Over keep threshold, now lost
            }
            break;
        case TrackState::Lost:
            if (this->m_frame_id - track->end_frame() < this->m_keep_lost_frames)
            {
		if (this->m_frame_id - track->end_frame() < this->m_keep_predict_frames)
		{
			track->m_kalman.Predict(track->m_kalman_rect, true);//prediction
			track->m_kalman.Predict(track->m_kalman_rect, true);//prediction
			track->m_tlwh[0] = track->m_kalman_rect.x;
			track->m_tlwh[1] = track->m_kalman_rect.y;

			//printf("lost: predicted:%d\n",m_keep_predict_frames);
			//printf("lost: predicted:%f,%f,%f,%f\n",track->m_tlwh[0],track->m_tlwh[1],track->m_tlwh[2],track->m_tlwh[3]);

		}

                lost_stracks.push_back(*track); // Not over threshold, so still lost
            }
            else
            {
                track->mark_removed(); // Over keep threshold, now removed
            }
            break;
        case TrackState::New:
            if (this->m_frame_id - track->end_frame() < this->m_keep_new_frames)
            {
                new_stracks.push_back(*track); // Not over threshold, so still new
            }
            else
            {
                track->mark_removed(); // Over keep threshold, now removed
            }
            break;
        }
    }
}
/**
 * @brief update tracker datasbase based on the tracking mode. 
 *        if mot is enabled by the user,search the selected track id 
 *        in active tracks.if not found search in lost tracks.if the selected 
 *	  track id is available,it is considered as a valid sot.
 *        take place:
*/
inline void JDETracker::update_trackmode(std::vector<STrack> &stracksa,std::vector<STrack> &stracksb,std::vector<STrack> &stracksc)
{
	if(m_shmp->_selectedTarget!=-1)//if not mot
	{
		//search in tracked tracks
		for (uint i = 0; i < stracksa.size(); i++)
    		{
        		if(stracksa[i].m_track_id==m_shmp->_selectedTarget)
			{
				STrack sot_track=stracksa[i];

                                std::vector<float> xyah= sot_track.to_xyah();
       				m_shmp->_sot_track.cX        = xyah[0]*m_shmp->_model_input_size_x;
        			m_shmp->_sot_track.cY        = xyah[1]*m_shmp->_model_input_size_y;
				m_shmp->_sot_track.width     = xyah[2]*xyah[3]*m_shmp->_model_input_size_x; //a*h
				m_shmp->_sot_track.height    = xyah[3]*m_shmp->_model_input_size_y; //h

        			m_shmp->_sot_track.trackID   = sot_track.m_track_id;

				m_shmp->_sot_track.classtype = (MVIGS_ObjectDetectionClassType) sot_track.m_class_id;

        			m_shmp->_bValidTrack=true;

				if(m_shmp->_sot_method==2)
				{
            			    stracksa.clear();
				    stracksb.clear();
				    stracksc.clear();
				    stracksa.push_back(sot_track);
				}
				return;
			}
		}
		
		//search in lost tracks
		for (uint i = 0; i < stracksb.size(); i++)
    		{
        		if(stracksb[i].m_track_id==m_shmp->_selectedTarget)
			{
				STrack sot_track=stracksb[i];

                                std::vector<float> xyah= sot_track.to_xyah();
				m_shmp->_sot_track.cX        = xyah[0]*m_shmp->_model_input_size_x;
        			m_shmp->_sot_track.cY        = xyah[1]*m_shmp->_model_input_size_y;
				m_shmp->_sot_track.width     = xyah[2]*xyah[3]*m_shmp->_model_input_size_x; //a*h
				m_shmp->_sot_track.height    = xyah[3]*m_shmp->_model_input_size_y; //h
        			m_shmp->_sot_track.trackID   = sot_track.m_track_id;
				m_shmp->_sot_track.classtype = (MVIGS_ObjectDetectionClassType) sot_track.m_class_id;
        			m_shmp->_bValidTrack=true;

				if(m_shmp->_sot_method==2)
				{
            			    stracksa.clear();
				    stracksb.clear();
				    stracksc.clear();
				    stracksb.push_back(sot_track);
				}
				return;
			}
		}

		m_shmp->_bValidTrack=false; //Note:: so track is lost permanently -couldnt find the sot either in active or in lost tracks

		if(m_shmp->_sot_method==2)
		{
            	   stracksa.clear();
		   stracksb.clear();
		   stracksc.clear();
		}
	}
	else
	{
        	m_shmp->_bValidTrack=false;
		if(m_shmp->_reticle_track_enable)
		{
        		int nearest_track_id=-1;
        		int min_distance=10000;//set to high value
                        int current_distance=10000;
                        int reticle_rect_xc=m_shmp->_reticle_rect_x+m_shmp->_reticle_rect_w/2;
 			int reticle_rect_yc=m_shmp->_reticle_rect_y+m_shmp->_reticle_rect_h/2;
                        int detection_box_xc=-1;
			int detection_box_yc=-1;

        		for (uint i = 0; i < stracksa.size(); i++)
    			{
				std::vector<float> xyah= stracksa[i].to_xyah();
                                detection_box_xc = xyah[0]*m_shmp->_model_input_size_x;
				detection_box_yc = xyah[1]*m_shmp->_model_input_size_y;
         
                		current_distance=std::min(std::abs(reticle_rect_xc-detection_box_xc), std::abs(reticle_rect_yc-detection_box_yc)); //min of xd,yd
                		if(current_distance<min_distance)
                		{
                     			min_distance=current_distance;
                     			nearest_track_id=stracksa[i].m_track_id;
                		}
			}
			for (uint i = 0; i < stracksb.size(); i++)
    			{
				std::vector<float> xyah= stracksb[i].to_xyah();
                                detection_box_xc = xyah[0]*m_shmp->_model_input_size_x;
				detection_box_yc = xyah[1]*m_shmp->_model_input_size_y;

                		current_distance=std::min(std::abs(reticle_rect_xc-detection_box_xc), std::abs(reticle_rect_yc-detection_box_yc)); //min of xd,yd
                		if(current_distance<min_distance)
                		{
                    			min_distance=current_distance;
                    			nearest_track_id=stracksb[i].m_track_id;
                		}
			}
        		if(min_distance<m_shmp->_reticle_rect_w/2 || min_distance<m_shmp->_reticle_rect_h/2)
        		{
				m_shmp->_selectedTarget=nearest_track_id;
				m_shmp->_reticle_track_enable=false;
        		}
		}
	}
	return;
}

/**
 * @brief The main logic unit and access point of the tracker system.
 *        On each new frame of the pipeline, this function should be called
 *        and given the new detections in the frame. The following steps
 *        take place:
 *        Step 1: New detections are converted into stracks. Then a kalman filter predicts
 *                the possible motion of the current tracked objects.
 *        Step 2: Matches are made between tracked objects and detections
 *                based on gating (Mahalanobis) distances.
 *                Matched stracks are updated/activated/re-activated as needed.
 *        Step 3: Matches are made between the leftover tracked and detected objects
 *                based on iou distances.
 *                Matched stracks are updated/activated/re-activated as needed.
 *        Step 4: Matches are made between leftover detections and unconfirmed (new)
 *                objects from the previous frames. Matches are based on iou
 *                distances, but with a lower threshold.
 *                Matched stracks are updated/activated/re-activated as needed.
 *        Step 5: Any leftover new detections are added as new objects to track.
 *        Step 6: Tracker database is updated.
 *        Step 7: Outputs are chosen from the tracked objects.
 *
 * @param inputs  -  std::vector<HailoDetectionPtr>
 *        The new detections from this frame.
 *
 * @param report_unconfirmed  -  bool
 *        If true, then output unconfirmed stracks as well.
 *
 * @return std::vector<STrack>
 *         The currently tracked (and unconfirmed if report_unconfirmed) objects.
 */
inline std::vector<STrack> JDETracker::update(std::vector<HailoDetectionPtr> &inputs, bool report_unconfirmed = false, bool report_lost = false)
{
    this->m_frame_id++;
    std::vector<STrack> detections;        // New detections in this update
    std::vector<STrack> activated_stracks; // Currently active stracks
    std::vector<STrack> lost_stracks;      // Currently lost stracks
    std::vector<STrack> new_stracks;       // Currently new stracks

    std::vector<STrack *> strack_pool; // A pool of tracked/lost stracks to find matches for

    std::vector<std::vector<float>> distances; // A distance cost matrix for linear assignment
    std::vector<std::pair<int, int>> matches;  // Pairs of matches between sets of stracks
    std::vector<int> unmatched_tracked;        // Unmatched tracked stracks
    std::vector<int> unmatched_detections;     // Unmatched new detections

    //******************************************************************
    // Step 1: Prepare tracks for new detections
    //******************************************************************
    detections = JDETracker::hailo_detections_to_stracks(inputs, this->m_frame_id, this->m_hailo_objects_blacklist); // Convert the new detections into STracks

    strack_pool = joint_strack_pointers(this->m_tracked_stracks, this->m_lost_stracks); // Pool together the tracked and lost stracks
    STrack::multi_predict(strack_pool, this->m_kalman_filter);                          // Run Kalman Filter prediction step

    //******************************************************************
    // Step 2: First association, tracked with embedding
    //******************************************************************
    // Calculate the distances between the tracked/lost stracks and the newly detected inputs
    //embedding_distance(strack_pool, detections, distances); // Calculate the distances
    //fuse_motion(distances, strack_pool, detections);        // Create the cost matrix


    //******************************************************************
    // Step 3.1: First association, tracked with IOU
    //******************************************************************


    //calculate the iou distance of what's left
    distances = iou_distance(strack_pool, detections);

    fuse_motion_custom(distances, strack_pool, detections);

    // Use linear assignment to find matches
    linear_assignment(distances, strack_pool.size(), detections.size(), this->m_iou_thr, matches, unmatched_tracked, unmatched_detections);

    // Update the matches
    update_matches(matches, strack_pool, detections, activated_stracks);

    // Use the unmatched_detections indices to get a vector of just the unmatched new detections
    keep_indices(detections, unmatched_detections);

    // Use the unmatched_tracked indices to get a vector of only unmatched, previously tracked, but-not-yet-lost stracks
    keep_indices(strack_pool, unmatched_tracked);

    //******************************************************************
    // Step 3.2: Second association, leftover tracked with extended IOU
    //******************************************************************

    if(m_shmp->_iou_scale1_enable==true)
    {
    // calculate the iou distance of what's left
    distances = iou_distance_custom(strack_pool, detections, m_shmp->_iou_scale_factor1);
    fuse_motion_custom(distances, strack_pool, detections);


    // Recalculate the linear assignment, this time use the iou threshold
    linear_assignment(distances, strack_pool.size(), detections.size(), this->m_iou_thr, matches, unmatched_tracked, unmatched_detections);
	
    // Update the matches
    update_matches(matches, strack_pool, detections, activated_stracks);

    // Break down the strack_pool to just the remaining unmatched stracks
    keep_indices(strack_pool, unmatched_tracked);

    //Use the unmatched_detections indices to get a vector of just the unmatched new detections again
    keep_indices(detections, unmatched_detections);
    }

    //******************************************************************
    // Step 3.3: Second association, leftover tracked with extended IOU
    //******************************************************************
    
    if(m_shmp->_iou_scale2_enable==true)
    {
    //calculate the iou distance of what's left
    distances = iou_distance_custom(strack_pool, detections, m_shmp->_iou_scale_factor2);
    fuse_motion_custom(distances, strack_pool, detections);


    // Recalculate the linear assignment, this time use the iou threshold
    linear_assignment(distances, strack_pool.size(), detections.size(), this->m_iou_thr, matches, unmatched_tracked, unmatched_detections);

    // Update the matches
    update_matches(matches, strack_pool, detections, activated_stracks);

    // Break down the strack_pool to just the remaining unmatched stracks
    keep_indices(strack_pool, unmatched_tracked);

    //Use the unmatched_detections indices to get a vector of just the unmatched new detections again
    keep_indices(detections, unmatched_detections);

    }

    // Update the state of the remaining unmatched stracks
    update_unmatches(strack_pool, activated_stracks, lost_stracks, new_stracks);


    //******************************************************************
    // Step 4: Remove duplicate new detections before consider them as actual
    //******************************************************************

    remove_duplicate_detections_custom(activated_stracks,detections,m_shmp->_fakeThreshold);


    //******************************************************************
    // Step 5: Third association, uncomfirmed with weaker IOU
    //******************************************************************
    // Deal with the unconfirmed stracks, these are usually stracks with only one beginning frame
    // Use the unmatched_detections indices to get a vector of just the unmatched new detections again

    std::vector<STrack> blank;
    std::vector<STrack *> unconfirmed_pool = joint_strack_pointers(this->m_new_stracks, blank); // Prepare a pool of unconfirmed stracks

    // Recalculate the iou distance, this time between unconfirmed stracks and the remaining detections
    distances = iou_distance(unconfirmed_pool, detections);

    fuse_motion_custom(distances,unconfirmed_pool, detections);


    // Recalculate the linear assignment, this time with the lower m_init_iou_thr threshold
    linear_assignment(distances, unconfirmed_pool.size(), detections.size(), this->m_init_iou_thr, matches, unmatched_tracked, unmatched_detections);

    // Update the matches
    update_matches(matches, unconfirmed_pool, detections, activated_stracks);

    // Break down the strack_pool to just the remaining unmatched stracks
    keep_indices(unconfirmed_pool, unmatched_tracked);

    // Update the state of the remaining unmatched stracks
    update_unmatches(unconfirmed_pool, activated_stracks, lost_stracks, new_stracks);


    //******************************************************************
    // Step 6: Init new stracks
    //******************************************************************
    // At this point, any leftover unmatched new detections are considered new object instances for tracking
    for (uint i = 0; i < unmatched_detections.size(); i++)
        new_stracks.emplace_back(detections[unmatched_detections[i]]);

    //******************************************************************
    // Step 6: remove fake stracks :: temp
    //******************************************************************
    //remove_duplicate_stracks_custom( activated_stracks,m_shmp->_fakeThreshold);

    //******************************************************************
    // Step 7: update tracking mode
    //******************************************************************
    update_trackmode(activated_stracks,lost_stracks,new_stracks);


    //******************************************************************
    // Step 8: Update Database
    //******************************************************************
    // Update the tracker database members with the results of this update
    this->m_tracked_stracks = activated_stracks;
    this->m_lost_stracks    = lost_stracks;
    this->m_new_stracks     = new_stracks;

    //******************************************************************
    // Step 9: Set the output stracks
    //******************************************************************

    //update no of active tracks
    m_shmp->_numTracks=this->m_tracked_stracks.size();
    
    //pack all active tracks to output
    std::vector<STrack> output_stracks;
    output_stracks.reserve(this->m_tracked_stracks.size());
    for (uint i = 0; i < this->m_tracked_stracks.size(); i++)
        output_stracks.emplace_back(this->m_tracked_stracks[i]);

    //save ouput stracks to shm
    for (uint i = 0; i < output_stracks.size(); i++)
    {   if(i<MAX_NUM_TRACKS)
	{
		STrack temp_track=output_stracks[i];
        	std::vector<float> xyah= temp_track.to_xyah();
		m_shmp->_tracks[i].cX        = xyah[0]*m_shmp->_model_input_size_x;//cx
        	m_shmp->_tracks[i].cY        = xyah[1]*m_shmp->_model_input_size_y;//cy
		m_shmp->_tracks[i].width     = xyah[2]*xyah[3]*m_shmp->_model_input_size_x; //a*h
		m_shmp->_tracks[i].height    = xyah[3]*m_shmp->_model_input_size_y; //h
        	m_shmp->_tracks[i].trackID   = temp_track.m_track_id;
		m_shmp->_tracks[i].classtype = (MVIGS_ObjectDetectionClassType) temp_track.m_class_id;
	}		
     }

    return output_stracks;
}
