/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once

#include <gst/base/gstbasetransform.h>
#include <vector>
#include "hailo_objects.hpp"

G_BEGIN_DECLS

#define GST_TYPE_OEPL_PREPROCESS (gst_oeplpreprocess_get_type())
#define GST_OEPL_PREPROCESS(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_OEPL_PREPROCESS, GstOeplPreprocess))
#define GST_OEPL_PREPROCESS_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_OEPL_PREPROCESS, GstOeplPreprocessClass))
#define GST_IS_OEPL_PREPROCESS(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_OEPL_PREPROCESS))
#define GST_IS_OEPL_PREPROCESS_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_OEPL_PREPROCESS))

typedef struct _GstOeplPreprocess GstOeplPreprocess;
typedef struct _GstOeplPreprocessClass GstOeplPreprocessClass;

struct _GstOeplPreprocess
{
    GstBaseTransform base_oeplpreprocess;

};

struct _GstOeplPreprocessClass
{
    GstBaseTransformClass base_oeplpreprocess_class;
};

GType gst_oeplpreprocess_get_type(void);

G_END_DECLS