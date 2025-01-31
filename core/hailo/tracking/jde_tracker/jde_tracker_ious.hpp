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
#include "strack.hpp"
#include "tracker_macros.hpp"


/**
 * @brief Calculate the ious between two sets of bounding boxes.
 *        Iou is calculated and filled into a dense graph.
 * 
 * @param atlbrs  -  std::vector<std::vector<float>>
 *        A vector of bounding boxes <xmin,ymin,xmax,ymax>
 *
 * @param btlbrs  -  std::vector<std::vector<float>>
 *        A vector of bounding boxes <xmin,ymin,xmax,ymax>
 *
 * @return std::vector<std::vector<float>> 
 *         A dense graph of of ious of shape atlbrs.size() x btlbrs.size()
 *         For interpreting distances - 0 is far, 1 is close
 */
inline std::vector<std::vector<float>> ious(std::vector<std::vector<float>> &atlbrs, std::vector<std::vector<float>> &btlbrs)
{
    // The iou graph will be of shape atlbrs.size() x btlbrs.size()
    std::vector<std::vector<float>> ious( atlbrs.size() , std::vector<float> (btlbrs.size()));

    // If there are no box, then return
    if (atlbrs.size() * btlbrs.size() == 0)
        return ious;

    //Calculate the ious between each possible pair of boxes from set A and set B
    for (uint k = 0; k < btlbrs.size(); k++)
    {
        float box_area = (btlbrs[k][2] - btlbrs[k][0]) * (btlbrs[k][3] - btlbrs[k][1]);
        for (uint n = 0; n < atlbrs.size(); n++)
        {
            float iw = std::min(atlbrs[n][2], btlbrs[k][2]) - std::max(atlbrs[n][0], btlbrs[k][0]);
            if (iw > 0.0f)
            {
                float ih = std::min(atlbrs[n][3], btlbrs[k][3]) - std::max(atlbrs[n][1], btlbrs[k][1]);
                if (ih > 0.0f)
                {
                    float ua = (atlbrs[n][2] - atlbrs[n][0]) * (atlbrs[n][3] - atlbrs[n][1]) + box_area - iw * ih;
                    ious[n][k] = iw * ih / ua;
                }
                else
                {
                    ious[n][k] = 0.0;
                }
            }
            else
            {
                ious[n][k] = 0.0;
            }
        }
    }

    return ious;
}
/**
 * @brief Calculate the ious between two sets of bounding boxes.
 *        Iou is calculated and filled into a dense graph.
 * 
 * @param atlbrs  -  std::vector<std::vector<float>>
 *        A vector of bounding boxes <xmin,ymin,xmax,ymax>
 *
 * @param btlbrs  -  std::vector<std::vector<float>>
 *        A vector of bounding boxes <xmin,ymin,xmax,ymax>
 *
 * @return std::vector<std::vector<float>> 
 *         A dense graph of of ious of shape atlbrs.size() x btlbrs.size()
 *         For interpreting distances - 0 is far, 1 is close
 */
inline std::vector<std::vector<float>> ious_custom(std::vector<std::vector<float>> &atlbrs, std::vector<std::vector<float>> &btlbrs, float scale)
{
    // The iou graph will be of shape atlbrs.size() x btlbrs.size()
    std::vector<std::vector<float>> ious(atlbrs.size(), std::vector<float>(btlbrs.size()));

    // If there are no boxes, then return
    if (atlbrs.size() * btlbrs.size() == 0)
        return ious;

    // Calculate IoU for each pair of bounding boxes
    for (int k = 0; k < btlbrs.size(); k++)
    {
        // Calculate extended normalized bounding box coordinates for box k
        float t_w = btlbrs[k][2] - btlbrs[k][0];
        float t_h = btlbrs[k][3] - btlbrs[k][1];
        float btlbr_0 = btlbrs[k][0] - t_w * scale;
        float btlbr_1 = btlbrs[k][1] - t_h * scale;
        float btlbr_2 = btlbrs[k][2] + t_w * scale;
        float btlbr_3 = btlbrs[k][3] + t_h * scale;

        // Iterate over the bounding boxes in atlbrs
        for (int n = 0; n < atlbrs.size(); n++)
        {
            // Calculate extended normalized bounding box coordinates for box n
            float d_w = atlbrs[n][2] - atlbrs[n][0];
            float d_h = atlbrs[n][3] - atlbrs[n][1];
            float atlbr_0 = atlbrs[n][0] - d_w * scale;
            float atlbr_1 = atlbrs[n][1] - d_h * scale;
            float atlbr_2 = atlbrs[n][2] + d_w * scale;
            float atlbr_3 = atlbrs[n][3] + d_h * scale;

            // Calculate intersection over union (IoU) in normalized coordinates
            float iw = std::min(atlbr_2, btlbr_2) - std::max(atlbr_0, btlbr_0);
            if (iw > 0)
            {
                float ih = std::min(atlbr_3, btlbr_3) - std::max(atlbr_1, btlbr_1);
                if (ih > 0)
                {
                    float ua = (atlbr_2 - atlbr_0) * (atlbr_3 - atlbr_1) + (btlbr_2 - btlbr_0) * (btlbr_3 - btlbr_1) - iw * ih;
                    ious[n][k] = iw * ih / ua;
                }
                else
                {
                    ious[n][k] = 0.0;
                }
            }
            else
            {
                ious[n][k] = 0.0;
            }
        }
    }

    return ious;
}


/**
 * @brief Calculates the iou distances (1 - iou) between two sets of STracks
 *        Distances are returned as a dense graph.
 * 
 * @param atracks  -  std::vector<STrack *>
 *        A set of STracks (by pointer)
 *
 * @param btracks   -  std::vector<STrack>
 *        A set of STracks
 *
 * @param dist_rows   -  int &
 *        int & to fill with the # of rows of the distance graph
 *
 * @param dist_cols   -  int &
 *        int & to fill with the # of columns of the distance graph
 *
 * @return std::vector<std::vector<float>> 
 *         A dense graph of iou distances (1 - iou), of shape atracks.size() x btracks.size()
 *         For interpreting distances - 1 is far, 0 is close
 */
inline std::vector<std::vector<float>> JDETracker::iou_distance(std::vector<STrack *> &atracks, std::vector<STrack> &btracks)
{
    if ( (atracks.size() == 0) | (btracks.size() == 0) )
    {
        std::vector<std::vector<float>> cost_matrix;
        return cost_matrix;
    }

    // Prepare a set of bounding boxes from each of the two sets of STracks
    std::vector<std::vector<float>> atlbrs( atracks.size() , std::vector<float> (4));
    std::vector<std::vector<float>> btlbrs( btracks.size() , std::vector<float> (4));
    for (uint i = 0; i < atracks.size(); i++)
    {
        atlbrs[i] = atracks[i]->tlbr();
    }
    for (uint i = 0; i < btracks.size(); i++)
    {
        btlbrs[i] = btracks[i].tlbr();
    }

    // Get a dense graph of the ious between all pairs of boxes fromt he two sets
    std::vector<std::vector<float>> _ious = ious(atlbrs, btlbrs);

    // The cost matrix will have the same shape as the ious graph
    std::vector<std::vector<float>> cost_matrix( atracks.size() , std::vector<float> (btracks.size()));
    //The cost matrix = 1 - ious
    for (uint i = 0; i < _ious.size(); i++)
    {
        for (uint j = 0; j < _ious[i].size(); j++)
        {
            cost_matrix[i][j] = 1 - _ious[i][j];
        }
    }

    return cost_matrix;
}
/**
 * @brief Calculates the iou distances (1 - iou) between two sets of STracks
 *        Distances are returned as a dense graph.
 * 
 * @param atracks  -  std::vector<STrack *>
 *        A set of STracks (by pointer)
 *
 * @param btracks   -  std::vector<STrack>
 *        A set of STracks
 * @param scaleFactor   -  float
 *        resize factor in terms of size of rectangles
 *
 * @param dist_rows   -  int &
 *        int & to fill with the # of rows of the distance graph
 *
 * @param dist_cols   -  int &
 *        int & to fill with the # of columns of the distance graph
 *
 * @return std::vector<std::vector<float>> 
 *         A dense graph of iou distances (1 - iou), of shape atracks.size() x btracks.size()
 *         For interpreting distances - 1 is far, 0 is close
 */
inline std::vector<std::vector<float>> JDETracker::iou_distance_custom(std::vector<STrack *> &atracks, std::vector<STrack> &btracks,float scale)
{
    if ( (atracks.size() == 0) | (btracks.size() == 0) )
    {
        std::vector<std::vector<float>> cost_matrix;
        return cost_matrix;
    }

    // Prepare a set of bounding boxes from each of the two sets of STracks
    std::vector<std::vector<float>> atlbrs( atracks.size() , std::vector<float> (4));
    std::vector<std::vector<float>> btlbrs( btracks.size() , std::vector<float> (4));
    for (uint i = 0; i < atracks.size(); i++)
    {
        atlbrs[i] = atracks[i]->tlbr();
    }
    for (uint i = 0; i < btracks.size(); i++)
    {
        btlbrs[i] = btracks[i].tlbr();
    }

    // Get a dense graph of the ious between all pairs of boxes fromt he two sets
    std::vector<std::vector<float>> _ious = ious_custom(atlbrs, btlbrs,scale);

    // The cost matrix will have the same shape as the ious graph
    std::vector<std::vector<float>> cost_matrix( atracks.size() , std::vector<float> (btracks.size()));
    //The cost matrix = 1 - ious
    for (uint i = 0; i < _ious.size(); i++)
    {
        for (uint j = 0; j < _ious[i].size(); j++)
        {
            cost_matrix[i][j] = 1 - _ious[i][j];
        }
    }

    return cost_matrix;
}


/**
 * @brief Calculates the iou distances (1 - iou) between two sets of STracks
 *        Distances are returned as a dense graph.
 * 
 * @param atracks  -  std::vector<STrack *>
 *        A set of STracks
 *
 * @param btracks  -  std::vector<STrack>
 *        A set of STracks
 *
 * @return std::vector<std::vector<float>> 
 *         A dense graph of iou distances (1 - iou), of shape atracks.size() x btracks.size()
 *         For interpreting distances - 1 is far, 0 is close
 */
inline std::vector<std::vector<float>> JDETracker::iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks)
{
    if ( (atracks.size() == 0) | (btracks.size() == 0) )
    {
        std::vector<std::vector<float>> cost_matrix;
        return cost_matrix;
    }

    // Prepare a set of bounding boxes from each of the two sets of STracks
    std::vector<std::vector<float>> atlbrs( atracks.size() , std::vector<float> (4));
    std::vector<std::vector<float>> btlbrs( btracks.size() , std::vector<float> (4));
    for (uint i = 0; i < atracks.size(); i++)
    {
        atlbrs[i] = atracks[i].tlbr();
    }
    for (uint i = 0; i < btracks.size(); i++)
    {
        btlbrs[i] = btracks[i].tlbr();
    }

    // Get a dense graph of the ious between all pairs of boxes fromt he two sets
    std::vector<std::vector<float>> _ious = ious(atlbrs, btlbrs);
    // The cost matrix will have the same shape as the ious graph
    std::vector<std::vector<float>> cost_matrix( atracks.size() , std::vector<float> (btracks.size()));
    //The cost matrix = 1 - ious
    for (uint i = 0; i < _ious.size(); i++)
    {
        for (uint j = 0; j < _ious[i].size(); j++)
        {
            cost_matrix[i][j] = 1 - _ious[i][j];
        }
    }

    return cost_matrix;
}

/**
 * @brief Calculate the intersection over minimum area between two sets of bounding boxes.
 *        Ioma is calculated and filled into a dense graph.
 * 
 * @param atlbrs  -  std::vector<std::vector<float>>
 *        A vector of bounding boxes <xmin,ymin,xmax,ymax>
 *
 * @param btlbrs  -  std::vector<std::vector<float>>
 *        A vector of bounding boxes <xmin,ymin,xmax,ymax>
 *
 * @return std::vector<std::vector<float>> 
 *         A dense graph of of iomas of shape atlbrs.size() x btlbrs.size()
 *         For interpreting distances - 0 is far, 1 is close
 */

inline std::vector<std::vector<float>> iomas(std::vector<std::vector<float>> &atlbrs, std::vector<std::vector<float>> &btlbrs)
{
    // The ioma graph will be of shape atlbrs.size() x btlbrs.size()
    std::vector<std::vector<float>> iomas(atlbrs.size(), std::vector<float>(btlbrs.size()));

    // If there are no boxes, then return
    if (atlbrs.size() * btlbrs.size() == 0)
        return iomas;

    // Calculate the iomas between each possible pair of boxes from set A and set B
    for (uint k = 0; k < btlbrs.size(); k++)
    {
        float box_area = (btlbrs[k][2] - btlbrs[k][0]) * (btlbrs[k][3] - btlbrs[k][1]);
        for (uint n = 0; n < atlbrs.size(); n++)
        {
            float iw = std::min(atlbrs[n][2], btlbrs[k][2]) - std::max(atlbrs[n][0], btlbrs[k][0]);
            if (iw > 0.0f)
            {
                float ih = std::min(atlbrs[n][3], btlbrs[k][3]) - std::max(atlbrs[n][1], btlbrs[k][1]);
                if (ih > 0.0f)
                {
                    // Calculate the Intersection over Minimum Area (IoMA)
                    float min_area = std::min((atlbrs[n][2] - atlbrs[n][0]) * (atlbrs[n][3] - atlbrs[n][1]), box_area);
                    iomas[n][k] = iw * ih / min_area;
                }
                else
                {
                    iomas[n][k] = 0.0;
                }
            }
            else
            {
                iomas[n][k] = 0.0;
            }
        }
    }

    return iomas;
}


/**
 * @brief Calculates the ioma distances (1 - iou) between two sets of STracks
 *        Distances are returned as a dense graph.
 * 
 * @param atracks  -  std::vector<STrack>
 *        A set of STracks
 *
 * @param btracks  -  std::vector<STrack>
 *        A set of STracks
 *
 * @return std::vector<std::vector<float>> 
 *         A dense graph of ioma distances (1 - ioma), of shape atracks.size() x btracks.size()
 *         For interpreting distances - 1 is far, 0 is close
 */

inline std::vector<std::vector<float>> JDETracker::ioma_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks)
{
    if ( (atracks.size() == 0) | (btracks.size() == 0) )
    {
        std::vector<std::vector<float>> cost_matrix;
        return cost_matrix;
    }

    // Prepare a set of bounding boxes from each of the two sets of STracks
    std::vector<std::vector<float>> atlbrs( atracks.size() , std::vector<float> (4));
    std::vector<std::vector<float>> btlbrs( btracks.size() , std::vector<float> (4));
    for (uint i = 0; i < atracks.size(); i++)
    {
        atlbrs[i] = atracks[i].tlbr();
    }
    for (uint i = 0; i < btracks.size(); i++)
    {
        btlbrs[i] = btracks[i].tlbr();
    }

    // Get a dense graph of the ious between all pairs of boxes fromt he two sets
    std::vector<std::vector<float>> _iomas = iomas(atlbrs, btlbrs);

    // The cost matrix will have the same shape as the _iomas graph
    std::vector<std::vector<float>> cost_matrix( atracks.size() , std::vector<float> (btracks.size()));
    for (uint i = 0; i < _iomas.size(); i++)
    {
        for (uint j = 0; j < _iomas[i].size(); j++)
        {
            cost_matrix[i][j] = 1 - _iomas[i][j];
        }
    }

    return cost_matrix;
}
