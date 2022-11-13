import os

import sys
sys.path.append('../')
sys.path.append('../bindings/build')
from loguru import logger
try:
    import gi

    gi.require_version('Gst', '1.0')
    from gi.repository import GLib, Gst
    logger.info("Import gi successfully.")

except Exception as e:
    logger.info(e)
    import pgi
    pgi.install_as_gi()
    from gi.repository import GLib, Gio, Gst

from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

try:
    import pyds
    logger.info("Import pyds successfully.")

except Exception as e:
    logger.debug("Can't import pyds, check built path ??")
    sys.exit()
    
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3


def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            #frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        print("frame number", frame_number)
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	

def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    # Initialize Gstreamer
    Gst.init(None)

    # Create gstreamer elements
    logger.info("Creating Pipeline")
    pipeline = Gst.Pipeline()

    if not pipeline:
        logger.debug("Can't create Pipeline")
        sys.exit()

    # Source element for readking from file
    logger.info("Create source")
    # type filesrc, name:file-source
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        logger.debug("Can't create source")
    
    # Create h264 parser
    logger.info("Creating h264 parse")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        logger.debug("Can't create h264parser")
        sys.exit()

    # nvdec_h264 for hardware accelerated decode on GPU
    logger.info("Creating decoder")
    decoder = Gst.ElementFactory.make("nnv412decoder", "nnv412-decoder")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        logger.debug("Can't create decoder")

    # Create nvstreammux instances to form batches from ones or more sources
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    if not streammux:
        logger.debug("Can't create nvstreammux")
    
    # Use nvinfer to run inference on decoder's output
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        logger.debug("Can't create primary inference engine")
    
    logger.info("Create converter to convert from nv12 to rgba")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        logger.debug("Can't create nv converter")
    
    # Create on screen display to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        logger.debug("Can't create nv on screen display")

    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    

    logger.info("Creating EGL Sink")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        logger.debug("Can't create EGL to render")
    
    logger.info("Playing file {}".format(args[1]))
    source.set_property('location', args[1])
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 400000)
    pgie.set_property('config-file-path', 'config.txt')

    logger.info("Adding elements to Pipeline")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    if is_aarch64():
        pipeline.add(transform)

    # Link the elements together
    # file-source -> h264-parser -> nv h264 decoder
    # -> nvinfer- > nvvidconv -> nvosd -> video-renderer
    logger.info("Linking elements in the Pipeline")
    source.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        logger.debug("Can't get the sinkpad of streammux")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        logger.debug("Can't create the source pad of decoder")
    
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # Create an event loop and feed gstreamer bus message to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        logger.debug("Can't get sink pad of nvosd")
    
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    logger.info("Staring the pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except Exception as e:
        logger.debug(e)

    # Clean up
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))