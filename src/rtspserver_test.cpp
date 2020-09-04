#include "gst/gst.h"
#include "gst/rtsp-server/rtsp-server.h"

static gchar* port = (gchar*) "22093";
static gchar* gstp = (gchar*) "( udpsrc port=25011 caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=RAW,sampling=YCbCr-4:2:0,depth=(string)8,width=(string)480,height=(string)360,payload=96 ! queue ! rtpvrawdepay ! queue ! x264enc tune=zerolatency ! h264parse ! queue ! rtph264pay name=pay0 )";
static GOptionEntry entries[] = {
  {"port", 'p', 0, G_OPTION_ARG_STRING, &port, "listening port", "PORT"},
  {NULL}
};

int main (int argc, char* argv[]) {

  GMainLoop* loop;
  GstRTSPServer* server;
  GstRTSPMountPoints* mounts;
  GstRTSPMediaFactory* factory;
  GOptionContext *optctx;
  GError *error = NULL;

  optctx = g_option_context_new("");
  g_option_context_add_main_entries(optctx, entries, NULL);
  g_option_context_add_group(optctx, gst_init_get_option_group());
  if (!g_option_context_parse(optctx, &argc, &argv, &error)) {
    g_printerr("Error parsing options: %s\n", error->message);
    g_option_context_free(optctx);
    g_clear_error(&error);
    return -1;
  }
  g_option_context_free(optctx);

  /* create a main loop NULL */
  loop = g_main_loop_new(NULL, FALSE);

  /* create a server instance */
  server = gst_rtsp_server_new();
  g_object_set(server, "service", port, NULL);

  /* get the mount points for the server */
  mounts = gst_rtsp_server_get_mount_points(server);

  /* make a media factory for media streams and set a launch pipeline in which each element with pay%d names will be a stream */
  factory = gst_rtsp_media_factory_new();
  gst_rtsp_media_factory_set_launch(factory, gstp);
  gst_rtsp_media_factory_set_shared(factory, TRUE);

  /* attach the test factory to the /test url */
  gst_rtsp_mount_points_add_factory(mounts, "/test", factory);

  /* unref the mapper */
  g_object_unref(mounts);

  /* attach the server to the default maincontext */
  gst_rtsp_server_attach(server, NULL);

  /* start serving */
  g_print("stream ready at rtsp://192.168.1.6:%s/test\n", port);
  g_main_loop_run(loop);

  return 0;

}