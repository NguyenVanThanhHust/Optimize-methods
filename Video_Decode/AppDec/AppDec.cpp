/*
* Copyright 2017-2021 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

//---------------------------------------------------------------------------
//! \file AppDec.cpp
//! \brief Source file for AppDec sample
//!
//! This sample application illustrates the demuxing and decoding of a media file followed by resize and crop of the output frames.
//! The application supports both planar (YUV420P and YUV420P16) and non-planar (NV12 and P016) output formats.
//---------------------------------------------------------------------------

#include <ctime>
#include <iostream>
#include <algorithm>
#include <thread>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../NvDecoder/NvDecoder.h"
#include "../Utils/NvCodecUtils.h"
#include "../Utils/FFmpegDemuxer.h"
#include "../Common/AppDecUtils.h"

using std::cout;
using std::cin;
using std::endl;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int main()
{
    try
    {
        /* code */
        std::string inputFile = "basic_4k_traffic.h264";
        const char * inputFileCstring = inputFile.c_str();
        cout<<"Video name: ";
        cout<<inputFileCstring<<endl;
        int width = 3840, height = 2160;
        FFmpegDemuxer demuxer(inputFileCstring);
        int videoCodec = demuxer.GetVideoCodec();
        cout<<videoCodec<<endl;
    }
    catch(const std::exception& e)
    {
        cout<<"Exception: ";
        std::cerr << e.what() << '\n';
    }
    

    return 0;
}
