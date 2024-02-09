/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <gst/gst.h>
#include <gst/video/video.h>
#include <opencv2/opencv.hpp>
#include "preprocess/gstoeplpreprocess.hpp"
#include "common/image.hpp"
#include "gst_hailo_meta.hpp"

#include <map>
#include <vector>
#include "hailo_common.hpp"

//shared mem include
#include<stdio.h>
#include<sys/ipc.h>
#include<sys/shm.h>
#include<sys/types.h>
#include<errno.h>
#include<stdlib.h>

#define STREAM_CHANNELS 3  //input camera stream num of channels
#define STREAM_WIDTH 1920  //input camera stream width
#define STREAM_HEIGHT 1080 //input camera stream height
#define PR_SHM_KEY 0x1221

//shared memory
struct pr_shmseg {
    unsigned char bDynamicContrast=1;
    unsigned char bResize=0;
    unsigned char bCustomGamma=1;
    float gamma_scale=2.0;
    int zoomcrop_x=320; //(1920-1280)/2
    int zoomcrop_y=180; //(1080-720)/2
    int zoomcrop_width=1280;
    int zoomcrop_height=720;
    int dest_image_width=1280; //by keeping the aspect ratio of input
    int dest_image_height=720; //by keeping the aspect ratio of input
    //unsigned int preprocess_count=0;
};
static int pr_shmid=-1;		        //shared memory id
static struct pr_shmseg *pr_shmp=nullptr;	//shared memory data

static struct pr_shmseg pr_shm;

//intermediate Mat image
static cv::Mat destMat;
static unsigned char* destBuff;


//gamma and contrast 
static double currentMultiplier = 1.0f;
static double currentGammaValue = 1.0f;
static double newGammaValue = 1.0f;
static double newMultiplier = 1.0f;
static int maxIntensity = 255;
static cv::Scalar scalarAvg_contrast, scalarSdv_contrast, scalarAvg_gamma;

static cv::Mat pGamma8BitImage;
static unsigned char GammaTable8Bit[256][3];

GST_DEBUG_CATEGORY_STATIC(gst_oeplpreprocess_debug_category);
#define GST_CAT_DEFAULT gst_oeplpreprocess_debug_category

/* prototypes */

static void gst_oeplpreprocess_set_property(GObject *object,
                                          guint property_id, const GValue *value, GParamSpec *pspec);
static void gst_oeplpreprocess_get_property(GObject *object,
                                          guint property_id, GValue *value, GParamSpec *pspec);
static void gst_oeplpreprocess_dispose(GObject *object);
static void gst_oeplpreprocess_finalize(GObject *object);

static gboolean gst_oeplpreprocess_start(GstBaseTransform *trans);
static gboolean gst_oeplpreprocess_stop(GstBaseTransform *trans);
static GstFlowReturn gst_oeplpreprocess_transform_ip(GstBaseTransform *trans,
                                                   GstBuffer *buffer);

/* class initialization */

G_DEFINE_TYPE_WITH_CODE(GstOeplPreprocess, gst_oeplpreprocess, GST_TYPE_BASE_TRANSFORM,
                        GST_DEBUG_CATEGORY_INIT(gst_oeplpreprocess_debug_category, "oeplpreprocess", 0,
                                                "debug category for oeplpreprocess element"));


static void
gst_oeplpreprocess_class_init(GstOeplPreprocessClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstBaseTransformClass *base_transform_class =
        GST_BASE_TRANSFORM_CLASS(klass);

    const char *description = "Preprocessing Before VideoAnalytics."
                              "\n\t\t\t   "
                              "processes incoming frames.";
    /* Setting up pads and setting metadata should be moved to
       base_class_init if you intend to subclass this class. */
    gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
                                       gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS,
                                                            gst_caps_from_string(GST_VIDEO_CAPS_MAKE("{ RGB, YUY2 }"))));
    gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
                                       gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
                                                            gst_caps_from_string(GST_VIDEO_CAPS_MAKE("{ RGB, YUY2 }"))));

    gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),
                                          "oeplpreprocess - preprocess element",
                                          "Hailo/Tools",
                                          description,
                                          "hailo.ai <contact@hailo.ai>");

 
    //Shared memory
    pr_shmid = shmget(PR_SHM_KEY, sizeof(struct pr_shmseg), 0644|IPC_CREAT);
    if (pr_shmid == -1)
    {
         perror("oepl preprocess | Shared memory create error\n");
         return ;
    }
    pr_shmp = (pr_shmseg*)shmat(pr_shmid, NULL, 0);
    if(pr_shmp == (void *) -1) 
    {
         perror("oepl preprocess | Shared memory attach error\n");
         return ;
    }
    //copy shared memory data locally
    memcy(&pr_shm,pr_shmp,size(struct pr_shmseg));    

    //dest mat
    destBuff=new unsigned char[pr_shm.dest_image_width*pr_shm.dest_image_height*STREAM_CHANNELS];
    destMat=cv::Mat(cv::Size(pr_shm.dest_image_width,pr_shm.dest_image_height),CV_8UC3,destBuff);

    //Gamma Correction LUT Image
    pGamma8BitImage = cv::Mat(cv::Size(256, 1), CV_8UC3, GammaTable8Bit);
    for (int pixelIndx = 0; pixelIndx < 256; pixelIndx++)
    {
	GammaTable8Bit[pixelIndx][0]   = pixelIndx;
	GammaTable8Bit[pixelIndx][1]   = pixelIndx;
        GammaTable8Bit[pixelIndx][2]   = pixelIndx;
    }

    //preprocess count
    //pr_shmp->preprocess_count=0;
        
    gobject_class->dispose = gst_oeplpreprocess_dispose;
    gobject_class->finalize = gst_oeplpreprocess_finalize;
    base_transform_class->start = GST_DEBUG_FUNCPTR(gst_oeplpreprocess_start);
    base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_oeplpreprocess_stop);
    base_transform_class->transform_ip =
        GST_DEBUG_FUNCPTR(gst_oeplpreprocess_transform_ip);
    printf("init preprocess zoom-scale\n");
}

static void
gst_oeplpreprocess_init(GstOeplPreprocess *oeplpreprocess)
{
	g_print ("Initializing Live Preprocessing\n\n");
}



void gst_oeplpreprocess_dispose(GObject *object)
{
    GstOeplPreprocess *oeplpreprocess = GST_OEPL_PREPROCESS(object);
    GST_DEBUG_OBJECT(oeplpreprocess, "dispose");

    /* clean up as possible.  may be called multiple times */

    G_OBJECT_CLASS(gst_oeplpreprocess_parent_class)->dispose(object);
}

void gst_oeplpreprocess_finalize(GObject *object)
{
    GstOeplPreprocess *oeplpreprocess = GST_OEPL_PREPROCESS(object);
    GST_DEBUG_OBJECT(oeplpreprocess, "finalize");

    /* clean up object here */

       g_print("oepl preprocess finalize\n");
       if (shmdt(pr_shmp) == -1) 
       {
           perror("oepl preprocess | Shared memory detach error\n");
       }
       if(destBuff!=nullptr)
       {
	  delete[] destBuff;
	  destBuff=nullptr;
       }
       if(destMat.empty()!=true)
       {
          destMat.release(); 
       }
       G_OBJECT_CLASS(gst_oeplpreprocess_parent_class)->finalize(object);
}

static gboolean
gst_oeplpreprocess_start(GstBaseTransform *trans)
{
    GstOeplPreprocess *oeplpreprocess = GST_OEPL_PREPROCESS(trans);
    GST_DEBUG_OBJECT(oeplpreprocess, "start");

    return TRUE;
}

static gboolean
gst_oeplpreprocess_stop(GstBaseTransform *trans)
{
    GstOeplPreprocess *oeplpreprocess = GST_OEPL_PREPROCESS(trans);
    GST_DEBUG_OBJECT(oeplpreprocess, "stop");

    return TRUE;
}

static GstFlowReturn
gst_oeplpreprocess_transform_ip(GstBaseTransform *trans,
                              GstBuffer *buffer)
{

    GstFlowReturn status = GST_FLOW_OK;
    GstOeplPreprocess *oeplpreprocess = GST_OEPL_PREPROCESS(trans);
    GstCaps *caps;
    cv::Mat mat;
  
    GST_DEBUG_OBJECT(oeplpreprocess, "transform_ip");

    caps = gst_pad_get_current_caps(trans->sinkpad);

    GstVideoInfo *info = gst_video_info_new();
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READWRITE);
    gst_video_info_from_caps(info, caps);
    mat = get_mat(info, &map);
    gst_video_info_free(info);

   
    memcy(&pr_shm,pr_shmp,size(struct pr_shmseg));

    //pr_shmp->preprocess_count++;

    /*SCALE and ZOOM  */ 
    if(pr_shm.bResize==0) 
    {
          //scale mode
          cv::resize(mat,destMat,cv::Size(pr_shm.dest_image_width,pr_shm.dest_image_height));  //1920*1080-->640*360
    }
   // else 
   // {
   //       //zoom mode
   //      cv::resize(mat(cv::Rect(pr_shmp->zoomcrop_x,pr_shmp->zoomcrop_y,pr_shmp->zoomcrop_width,pr_shmp->zoomcrop_height)),destMat,cv::Size(pr_shmp->dest_image_width,pr_shmp->				dest_image_height)); //middle 1280*1080 of 1920*1080-->640*360
   // }

    /*CONTRAST AND BRIGHTNESS */
    if (pr_shm.bDynamicContrast == 1)
    {
	/*contrast correction:*/
	{

		//compute mean and standard deviation	
		cv::meanStdDev(destMat, scalarAvg_contrast, scalarSdv_contrast);

		double avgValue_contrast = (114 * scalarAvg_contrast[0] + 587 * scalarAvg_contrast[1] + 299 * scalarAvg_contrast[2]) / 1000.0f;
		double stdDeviation_contrast = (114 * scalarSdv_contrast[0] + 587 * scalarSdv_contrast[1] + 299 * scalarSdv_contrast[2]) / 1000.0f;

		// Compute new multiplier
		double measuredContrast = (5 * (double)stdDeviation_contrast / maxIntensity); //assumption: optimium contrast is when 5sigma=255,but practically 1.0-->1.2 is considered good contrast level.

		if (measuredContrast < 1.0)
		{
			newMultiplier = 1.0f + 0.75*(1.0 - measuredContrast);
			if (newMultiplier > 1.5)
			{
				newMultiplier = 1.5;
			}
		}
		else if(measuredContrast > 1.2)
		{
			newMultiplier = 1.0 + 0.5*(1.2 - measuredContrast);

			if (newMultiplier < 0.9)
			{
				newMultiplier = 0.9;
			}
		}
		else
		{
			newMultiplier=1.0;
		}


		//find change in contrast multiplier value
		double  deltaContrast = abs(newMultiplier - currentMultiplier) / currentMultiplier;

		//update multiplier
		if (deltaContrast > 0.1)
		{
			currentMultiplier = newMultiplier;
		}

		//apply contrast correction
		int contrastMul = (int)(currentMultiplier * 16);
		destMat = (contrastMul * destMat + (16 - contrastMul)* (int)(avgValue_contrast)) / 16;

	}
	
	/*gamma correction:*/
	{

		//Compute mean 
		scalarAvg_gamma = cv::mean(destMat);
		double avgValue_gamma = (114 * scalarAvg_gamma[0] + 587 * scalarAvg_gamma[1] + 299 * scalarAvg_gamma[2]) / 1000.0f;

		//Compute new gamma 
		double measured_brightness = avgValue_gamma / maxIntensity;

		if (measured_brightness > 0.5)//0.5 is the optimum
		{
			newGammaValue = 1 + (0.5 - measured_brightness) / 2.0; 
		}
		else
		{
			newGammaValue = 1 + pr_shm.gamma_scale * (0.5 - measured_brightness);
		}

		//find change in gamma value
		double  deltaGamma = abs(newGammaValue - currentGammaValue) / currentGammaValue;

		//update gamma 
		if (deltaGamma > 0.1)
		{
			currentGammaValue = newGammaValue;
			for (int pixelIndx = 0; pixelIndx < 256; pixelIndx++)
			{
				double value=0;
				if(pr_shm.bCustomGamma==1)
				{
				    value = (maxIntensity*(1 - pow( (maxIntensity - pixelIndx) / (double)maxIntensity, currentGammaValue)));
				}
				else
				{
				    value = (pow(pixelIndx / (double)maxIntensity, 1 / currentGammaValue) * maxIntensity);
				}
				GammaTable8Bit[pixelIndx][0] = (unsigned char)value;
				GammaTable8Bit[pixelIndx][1] = (unsigned char)value;
				GammaTable8Bit[pixelIndx][2] = (unsigned char)value;
			}
		}

		//apply gamma
		cv::LUT(destMat, pGamma8BitImage, destMat);
	}
   }

    /*update final image*/
    mat(cv::Rect(0,pr_shm.dest_image_height,pr_shm.dest_image_width,(STREAM_HEIGHT)-pr_shm.dest_image_height)).setTo(0); //set 640*280 portion of 640*640(model input)to 0
    destMat.copyTo(mat(cv::Rect(0,0,pr_shm.dest_image_width,pr_shm.dest_image_height))); //copy 640*360 to tl 640*360 of 1920*1080

    if(status==GST_FLOW_ERROR)
    {
	goto cleanup;
    }

    status = GST_FLOW_OK;
cleanup:
    mat.release();
    gst_buffer_unmap(buffer, &map);
    gst_caps_unref(caps);
    return status;
}
