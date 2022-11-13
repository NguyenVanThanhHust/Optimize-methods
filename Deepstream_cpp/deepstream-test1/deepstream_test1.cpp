#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include <iostream> 
#include <string>
#include <cstring>

using std::cout;
using std::cin;
using std::endl;

#define MAX_DISPLAY_LEN 64
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/*
Set resolution so that every input streams would be scaled to this resolution
*/
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/*
Muxer batch formation timeouts, should be set based one the fastest souorce's framerate
Below is 40 milisec
*/
#defin MUXER_BATCH_TIMEOUT_USEC 40000

int frame_number = 0;
std::string pgire_classes_str[4] = {
    "Vehicle", "TwoWheeler", "Person", "Roadsign"
};
    
// Onscreen display
static GstPadProbeReturn  osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info)
int main(int argc, char *argv[])
{
    cout<<"some sample"<<endl;
    return 0;

}