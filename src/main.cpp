// rife implemented with ncnn library

#include <stdio.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <clocale>
#include <thread>

#if _WIN32
// image decoder and encoder with wic
#include "wic_image.h"
#else // _WIN32
// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // _WIN32
#include "webp_image.h"

#if _WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
    if (optind >= argc || argv[optind][0] != L'-')
        return -1;

    wchar_t opt = argv[optind][1];
    const wchar_t* p = wcschr(optstring, opt);
    if (p == NULL)
        return L'?';

    optarg = NULL;

    if (p[1] == L':')
    {
        optind++;
        if (optind >= argc)
            return L'?';

        optarg = argv[optind];
    }

    optind++;

    return opt;
}

static std::vector<int> parse_optarg_int_array(const wchar_t* optarg)
{
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t* p = wcschr(optarg, L',');
    while (p)
    {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}
#else // _WIN32
#include <unistd.h> // getopt()
#include <fcntl.h>  // O_CREAT, O_RDWR

static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}
#endif // _WIN32

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"
#include "benchmark.h"

#include "rife.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stderr, "Usage: rife-ncnn-vulkan -0 infile -1 infile1 -o outfile [options]...\n");
    fprintf(stderr, "       rife-ncnn-vulkan -i indir -o outdir [options]...\n\n");
    fprintf(stderr, "  -h                   show this help\n");
    fprintf(stderr, "  -v                   verbose output\n");
    fprintf(stderr, "  -0 input0-path       input image0 path (jpg/png/webp)\n");
    fprintf(stderr, "  -1 input1-path       input image1 path (jpg/png/webp)\n");
    fprintf(stderr, "  -i input-path        input image directory (jpg/png/webp)\n");
    fprintf(stderr, "  -o output-path       output image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -I input-video-path  input video path (mp4/mov)\n");
    fprintf(stderr, "  -n num-frame         target frame count (default=N*2)\n");
    fprintf(stderr, "  -s time-step         time step (0~1, default=0.5)\n");
    fprintf(stderr, "  -m model-path        rife model path (default=rife-v2.3)\n");
    fprintf(stderr, "  -g gpu-id            gpu device to use (-1=cpu, default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stderr, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stdout, "  -x                   enable spatial tta mode\n");
    fprintf(stdout, "  -z                   enable temporal tta mode\n");
    fprintf(stdout, "  -u                   enable UHD mode\n");
    fprintf(stderr, "  -f pattern-format    output image filename pattern format (%%08d.jpg/png/webp, default=ext/%%08d.png)\n");
}

static int decode_image(const path_t& imagepath, ncnn::Mat& image, int* webp)
{
    *webp = 0;

    unsigned char* pixeldata = 0;
    int w;
    int h;
    int c;

#if _WIN32
    FILE* fp = _wfopen(imagepath.c_str(), L"rb");
#else
    FILE* fp = fopen(imagepath.c_str(), "rb");
#endif
    if (fp)
    {
        // read whole file
        unsigned char* filedata = 0;
        int length = 0;
        {
            fseek(fp, 0, SEEK_END);
            length = ftell(fp);
            rewind(fp);
            filedata = (unsigned char*)malloc(length);
            if (filedata)
            {
                fread(filedata, 1, length, fp);
            }
            fclose(fp);
        }

        if (filedata)
        {
            pixeldata = webp_load(filedata, length, &w, &h, &c);
            if (pixeldata)
            {
                *webp = 1;
            }
            else
            {
                // not webp, try jpg png etc.
#if _WIN32
                pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else // _WIN32
                pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 3);
                c = 3;
#endif // _WIN32
            }

            free(filedata);
        }
    }

    if (!pixeldata)
    {
#if _WIN32
        fwprintf(stderr, L"decode image %ls failed\n", imagepath.c_str());
#else // _WIN32
        fprintf(stderr, "decode image %s failed\n", imagepath.c_str());
#endif // _WIN32

        return -1;
    }

    image = ncnn::Mat(w, h, (void*)pixeldata, (size_t)3, 3);

    return 0;
}

static int encode_image(const path_t& imagepath, const ncnn::Mat& image)
{
    int success = 0;

    path_t ext = get_file_extension(imagepath);

    if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
    {
        success = webp_save(imagepath.c_str(), image.w, image.h, image.elempack, (const unsigned char*)image.data);
    }
    else if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
    {
#if _WIN32
        success = wic_encode_image(imagepath.c_str(), image.w, image.h, image.elempack, image.data);
#else
        success = stbi_write_png(imagepath.c_str(), image.w, image.h, image.elempack, image.data, 0);
#endif
    }
    else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
    {
#if _WIN32
        success = wic_encode_jpeg_image(imagepath.c_str(), image.w, image.h, image.elempack, image.data);
#else
        success = stbi_write_jpg(imagepath.c_str(), image.w, image.h, image.elempack, image.data, 100);
#endif
    }

    if (!success)
    {
#if _WIN32
        fwprintf(stderr, L"encode image %ls failed\n", imagepath.c_str());
#else
        fprintf(stderr, "encode image %s failed\n", imagepath.c_str());
#endif
    }

    return success ? 0 : -1;
}

class Task
{
public:
    int id;
    int webp0;
    int webp1;

    path_t in0path;
    path_t in1path;
    path_t outpath;
    float timestep;

    ncnn::Mat in0image;
    ncnn::Mat in1image;
    ncnn::Mat outimage;

    bool operator < (const Task& rhs) const
    {
        return id > rhs.id;
    }
};

class TaskQueue
{
public:
    TaskQueue()
    {
    }

    void put(const Task& v)
    {
        lock.lock();

        while (tasks.size() >= 8) // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        condition.signal();
    }

    void get(Task& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.top();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::priority_queue<Task> tasks;
};

TaskQueue toproc;
TaskQueue tosave;

class LoadThreadParams
{
public:
    int jobs_load;

    // session data
    std::vector<path_t> input0_files;
    std::vector<path_t> input1_files;
    std::vector<path_t> output_files;
    std::vector<float> timesteps;
};

void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    const int count = ltp->output_files.size();

    #pragma omp parallel for schedule(static,1) num_threads(ltp->jobs_load)
    for (int i=0; i<count; i++)
    {
        const path_t& image0path = ltp->input0_files[i];
        const path_t& image1path = ltp->input1_files[i];

        Task v;
        v.id = i + 1;
        v.in0path = image0path;
        v.in1path = image1path;
        v.outpath = ltp->output_files[i];
        v.timestep = ltp->timesteps[i];

        int ret0 = decode_image(image0path, v.in0image, &v.webp0);
        int ret1 = decode_image(image1path, v.in1image, &v.webp1);

        if (ret0 != 0 || ret1 != 1)
        {
            v.outimage = ncnn::Mat(v.in0image.w, v.in0image.h, (size_t)3, 3);
            toproc.put(v);
        }
    }

    return 0;
}

#include <opencv2/opencv.hpp>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaMuxer.h>
#include <media/NdkMediaFormat.h>
#include <media/NdkMediaError.h>
class LoadVideoThreadParams
{
public:
    path_t input_videopath;
    path_t outputpath;
    path_t pattern;
    path_t format;
};

void* load_video(void* args) {
    const LoadVideoThreadParams* ltp = (const LoadVideoThreadParams*)args;
    const path_t& input_videopath = ltp->input_videopath;
    const path_t& outputpath = ltp->outputpath;
    const path_t& pattern = ltp->pattern;
    const path_t& format = ltp->format;

    FILE* video_file = fopen(input_videopath.c_str(), "r");
    int fd = fileno(video_file);

    AMediaExtractor* extractor = AMediaExtractor_new();
    // int status = AMediaExtractor_setDataSource(extractor, input_videopath.c_str());
    int status = AMediaExtractor_setDataSourceFd(extractor, fd, 0, INT64_MAX);

    if (status != AMEDIA_OK) {
        fprintf(stderr, "Failed to set data source: %d\n", status);

        fclose(video_file);
        AMediaExtractor_delete(extractor);
    
        return 0;
    }

    // get video track
    int videoTrackIndex = -1;
    int trackCount = AMediaExtractor_getTrackCount(extractor);
    for (int i = 0; i < trackCount; ++i) {
        AMediaFormat* media_format = AMediaExtractor_getTrackFormat(extractor, i);
        const char* mime = nullptr;
        if (AMediaFormat_getString(media_format, AMEDIAFORMAT_KEY_MIME, &mime) && strncmp(mime, "video/", 6) == 0) {
            videoTrackIndex = i;
            break;
        }
        AMediaFormat_delete(media_format);
    }

    if (videoTrackIndex == -1) {
        printf("Failed to find video track\n");

        fclose(video_file);
        AMediaExtractor_delete(extractor);
        return 0;
    }

    // init decoder
    AMediaFormat* videoFormat = AMediaExtractor_getTrackFormat(extractor, videoTrackIndex);
    AMediaExtractor_selectTrack(extractor, videoTrackIndex);

    // get width and height
    int width = 0;
    int height = 0;
    AMediaFormat_getInt32(videoFormat, AMEDIAFORMAT_KEY_WIDTH, &width);
    AMediaFormat_getInt32(videoFormat, AMEDIAFORMAT_KEY_HEIGHT, &height);

    const char* mime = nullptr;
    AMediaFormat_getString(videoFormat, AMEDIAFORMAT_KEY_MIME, &mime);
    AMediaCodec* decoder = AMediaCodec_createDecoderByType(mime);
    AMediaCodec_configure(decoder, videoFormat, nullptr, nullptr, 0);
    AMediaCodec_start(decoder);

    int frame_id = 0;
    unsigned char *pre_pixeldata = nullptr;
    size_t bufferSize = 0;

    bool sawInputEOS = false;
    bool sawOutputEOS = false;
    while (!sawOutputEOS) {
        // 处理输入缓冲区
        if (!sawInputEOS) {
            ssize_t inputBufferIndex = AMediaCodec_dequeueInputBuffer(decoder, 1000);
            if (inputBufferIndex >= 0) {
                size_t inputBufferSize;
                unsigned char* inputBuffer = AMediaCodec_getInputBuffer(decoder, inputBufferIndex, &inputBufferSize);

                ssize_t sampleSize = AMediaExtractor_readSampleData(extractor, inputBuffer, inputBufferSize);
                if (sampleSize < 0) {
                    sawInputEOS = true;
                    sampleSize = 0;
                }

                uint64_t presentationTimeUs = AMediaExtractor_getSampleTime(extractor);
                AMediaCodec_queueInputBuffer(decoder, inputBufferIndex, 0, sampleSize,
                                                presentationTimeUs,
                                                sawInputEOS ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0);

                AMediaExtractor_advance(extractor);
            }
        }

        // 处理输出缓冲区
        AMediaCodecBufferInfo bufferInfo;
        ssize_t outputBufferIndex = AMediaCodec_dequeueOutputBuffer(decoder, &bufferInfo, 1000);

        if (outputBufferIndex >= 0) {
            if (bufferInfo.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                sawOutputEOS = true;
            }
            
            size_t outputBufferSize; // Y 占 1个字节，U,V 各占 1/4，outBufferSize = width * height * 1.5
            uint8_t* outputBuffer = AMediaCodec_getOutputBuffer(decoder, outputBufferIndex, &outputBufferSize);
            // printf("load %d\n", frame_id * 2);
            
            if (pre_pixeldata == nullptr) {
                unsigned char* pixeldata = (unsigned char*)malloc(width * height * 3);
                ncnn::yuv420sp2rgb_nv12(outputBuffer, width, height, pixeldata);

                bufferSize = width * height * 3;
                pre_pixeldata = (uint8_t*)malloc(bufferSize);
                
                memcpy(pre_pixeldata, pixeldata, bufferSize);

                char tmp[256];
                sprintf(tmp, pattern.c_str(), 1);
                path_t output_filename = path_t(tmp) + PATHSTR('.') + format;

                Task v;
                v.id = frame_id * 2 + 1;
                v.in0image = ncnn::Mat(width, height, (void*)pixeldata, (size_t)3, 3);
                v.in1image = ncnn::Mat(width, height, (size_t)3, 3);
                v.timestep = 0.f;
                v.outimage = ncnn::Mat(width, height, (size_t)3, 3);
                v.outpath = outputpath + PATHSTR('/') + output_filename;
                v.webp0 = 0;
                v.webp1 = 0;

                toproc.put(v);
                frame_id++;

                AMediaCodec_releaseOutputBuffer(decoder, outputBufferIndex, false);
                continue;
            }

            /*
            pre_pixeldata    + pixeldata          -> interpolate
            null             + pixeldata_skip     -> skip interpolate
            pixeldata_to_pre +  (next) pixel data -> interpolate
            */

            unsigned char* pixeldata = (unsigned char*)malloc(width * height * 3);
            unsigned char* pixeldata_skip = (unsigned char*)malloc(width * height * 3);
            unsigned char* pixeldata_to_pre = (unsigned char*)malloc(width * height * 3);

            ncnn::yuv420sp2rgb_nv12(outputBuffer, width, height, pixeldata);
            memcpy(pixeldata_skip, pixeldata, width * height * 3);
            memcpy(pixeldata_to_pre, pixeldata, width * height * 3);


            char tmp[256];
            sprintf(tmp, pattern.c_str(), frame_id * 2);
            path_t output_filename = path_t(tmp) + PATHSTR('.') + format;

            Task v;
            v.id = frame_id * 2; // Use timestamp as ID
            v.in0image = ncnn::Mat(width, height, (void*)pre_pixeldata, (size_t)3, 3);
            v.in1image = ncnn::Mat(width, height, (void*)pixeldata, (size_t)3, 3); 
            v.outpath = outputpath + PATHSTR('/') + output_filename;
            v.timestep = 0.5f; // Default timestep
            v.outimage = ncnn::Mat(width, height, (size_t)3, 3);

            toproc.put(v);


            sprintf(tmp, pattern.c_str(), frame_id * 2 + 1);
            output_filename = path_t(tmp) + PATHSTR('.') + format;

            Task v2;
            v2.id = frame_id * 2 + 1; // Use timestamp as I
            v2.in1image = ncnn::Mat(width, height, (void*)pixeldata_skip, (size_t)3, 3); 
            v2.outpath = outputpath + PATHSTR('/') + output_filename;
            v2.timestep = 1.f; // Default timestep
            v2.outimage = ncnn::Mat(width, height, (size_t)3, 3);

            toproc.put(v2);

            pre_pixeldata = pixeldata_to_pre;


            frame_id++;

            AMediaCodec_releaseOutputBuffer(decoder, outputBufferIndex, false);
        }
        // else if (outputBufferIndex == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
        // }
        // else if (outputBufferIndex == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
        //     AMediaFormat* newFormat = AMediaCodec_getOutputFormat(decoder);
        // }
    }

    AMediaCodec_stop(decoder);
    AMediaCodec_delete(decoder);
    AMediaExtractor_delete(extractor);
    fclose(video_file);
    free(pre_pixeldata);
    pre_pixeldata = nullptr;
    
    return 0;
}

class ProcThreadParams
{
public:
    const RIFE* rife;
};

void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const RIFE* rife = ptp->rife;

    for (;;)
    {
        Task v;

        toproc.get(v);

        if (v.id == -233)
            break;

        rife->process(v.in0image, v.in1image, v.timestep, v.outimage);

        tosave.put(v);
    }

    return 0;
}

class SaveThreadParams
{
public:
    int verbose;
};

void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    const int verbose = stp->verbose;

    for (;;)
    {
        Task v;

        tosave.get(v);

        if (v.id == -233)
            break;

        int ret = encode_image(v.outpath, v.outimage);

        // free input pixel data
        {
            unsigned char* pixeldata = (unsigned char*)v.in0image.data;
            if (v.webp0 == 1)
            {
                free(pixeldata);
            }
            else
            {
#if _WIN32
                free(pixeldata);
#else
                stbi_image_free(pixeldata);
#endif
            }
        }
        {
            unsigned char* pixeldata = (unsigned char*)v.in1image.data;
            if (v.webp1 == 1)
            {
                free(pixeldata);
            }
            else
            {
#if _WIN32
                free(pixeldata);
#else
                stbi_image_free(pixeldata);
#endif
            }
        }

        if (ret == 0)
        {
            if (verbose)
            {
#if _WIN32
                fwprintf(stderr, L"%ls %ls %f -> %ls done\n", v.in0path.c_str(), v.in1path.c_str(), v.timestep, v.outpath.c_str());
#else
                fprintf(stderr, "%s %s %f -> %s done\n", v.in0path.c_str(), v.in1path.c_str(), v.timestep, v.outpath.c_str());
#endif
            }
        }
    }

    return 0;
}

class SaveVideoThreadParams
{
public:
    int verbose;
    int width;
    int height;
    int fps;
    std::string output_video_path;
};

void* save_video(void* args)
{

    const SaveVideoThreadParams* stp = (const SaveVideoThreadParams*)args;
    const int verbose = stp->verbose;

    // ================== 初始化编码器和 Muxer ==================
    AMediaCodec* encoder = nullptr;
    AMediaMuxer* muxer = nullptr;
    int muxerTrackIndex = -1;
    bool muxerStarted = false;
    int64_t presentationTimeUs = 0;
    const int frameIntervalUs = 1000000 / stp->fps;

    // 创建编码器
    encoder = AMediaCodec_createEncoderByType("video/avc");
    AMediaFormat* format = AMediaFormat_new();


    AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, "video/avc");
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, stp->width);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT, stp->height);

    /*
    MediaCodecInfo在NDK中没有相关接口，直接使用数值
    Codec: c2.android.avc.encoder, Type: video/avc, 
    Supported Color Formats: 
    2135033992, // COLOR_FormatYUV420Flexible
    19,         // COLOR_FormatYUV420Planar
    21,         // COLOR_FormatYUV420SemiPlanar
    20,         // COLOR_FormatYUV420PackedPlanar
    39,         // COLOR_FormatYUV420PackedSemiPlanar
    2130708361  // COLOR_FormatSurface
    Codec: OMX.google.h264.encoder, Type: video/avc, 
    Supported Color Formats: 2135033992, 19, 21, 20, 39, 2130708361
    */
    // AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, 0x7f420888); // COLOR_FormatYUV420Flexible
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, 19);

    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_BIT_RATE, 8000000);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_FRAME_RATE, stp->fps);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_I_FRAME_INTERVAL, 1);

    
    
    media_status_t status = AMediaCodec_configure(encoder, format, nullptr, nullptr, AMEDIACODEC_CONFIGURE_FLAG_ENCODE);
    if (status != AMEDIA_OK) {
        printf("Failed to configure encoder: %d, format: %s\n", status, AMediaFormat_toString(format));
        AMediaCodec_delete(encoder);
        AMediaFormat_delete(format);
    }

    status = AMediaCodec_start(encoder);
    if (status != AMEDIA_OK) {
        printf("Failed to start encoder: %d\n", status);
        AMediaCodec_delete(encoder);
        AMediaFormat_delete(format);
    }

    // 创建 Muxer
    int fd = open(stp->output_video_path.c_str(), O_CREAT | O_RDWR, 0644);
    muxer = AMediaMuxer_new(fd, AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4);

    if (fd < 0 || muxer == nullptr) {
        printf("Failed to create muxer: %d\n", fd);
        if (fd >= 0) close(fd);
        AMediaCodec_delete(encoder);
        return 0;
    }

    std::thread inputThread([&]() {
        for (;;)
        {
            Task v;
            tosave.get(v);

            if (v.id == -233) {
                // 发送 EOS 信号
                ssize_t inputBufferIndex = AMediaCodec_dequeueInputBuffer(encoder, 1000);
                if (inputBufferIndex >= 0) {
                    AMediaCodec_queueInputBuffer(
                        encoder,
                        inputBufferIndex,
                        0,
                        0,
                        presentationTimeUs,
                        AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM
                    );
                }
                break;
            }

            // 将 RGB 转换为 YUV420
            cv::Mat rgbMat(v.outimage.h, v.outimage.w, CV_8UC3, (void*)v.outimage.data);
            cv::Mat yuvMat;
            cv::cvtColor(rgbMat, yuvMat, cv::COLOR_RGB2YUV_I420); // match COLOR_FormatYUV420Planar

            // 提交到编码器输入缓冲区
            ssize_t inputBufferIndex = AMediaCodec_dequeueInputBuffer(encoder, 1000000); // 1s
            while (inputBufferIndex < 0) {
                inputBufferIndex = AMediaCodec_dequeueInputBuffer(encoder, 1000000);
                printf("reget inputBufferIndex: %zd\n", inputBufferIndex);
            }
            size_t inputBufferSize;
            uint8_t* inputBuffer = AMediaCodec_getInputBuffer(encoder, inputBufferIndex, &inputBufferSize);

            if (yuvMat.total() * yuvMat.elemSize() <= inputBufferSize) {
                memcpy(inputBuffer, yuvMat.data, yuvMat.total() * yuvMat.elemSize());
                AMediaCodec_queueInputBuffer(
                    encoder,
                    inputBufferIndex,
                    0,
                    yuvMat.total() * yuvMat.elemSize(),
                    presentationTimeUs,
                    0
                );
                presentationTimeUs += frameIntervalUs;
            } else {
                printf("Input buffer too small: %zu < %zu\n", inputBufferSize, yuvMat.total() * yuvMat.elemSize());
                break;
            }

            // 释放资源（保持原有逻辑）
            {
                // ... 原有释放 in0image 和 in1image 的代码 ...
                unsigned char* pixeldata = (unsigned char*)v.in0image.data;
                if (v.webp0 == 1) {
                    free(pixeldata);
                } else {
                    stbi_image_free(pixeldata);
                }
                pixeldata = (unsigned char*)v.in1image.data;
                if (v.webp1 == 1) {
                    free(pixeldata);
                } else {
                    stbi_image_free(pixeldata);
                }
            }
        }
    });
    

    std::thread outputThread( [&]() {
        for(;;) {
            AMediaCodecBufferInfo bufferInfo;
            ssize_t outputBufferIndex = AMediaCodec_dequeueOutputBuffer(encoder, &bufferInfo, 0);
            while (outputBufferIndex >= 0) {
                if (!muxerStarted) {
                    AMediaFormat* outputFormat = AMediaCodec_getOutputFormat(encoder);
                    muxerTrackIndex = AMediaMuxer_addTrack(muxer, outputFormat);
                    AMediaMuxer_start(muxer);
                    muxerStarted = true;
                    AMediaFormat_delete(outputFormat);
                }

                size_t outputBufferSize;
                uint8_t* outputBuffer = AMediaCodec_getOutputBuffer(encoder, outputBufferIndex, &outputBufferSize);

                AMediaMuxer_writeSampleData(
                    muxer,
                    muxerTrackIndex,
                    outputBuffer,
                    &bufferInfo
                );
                AMediaCodec_releaseOutputBuffer(encoder, outputBufferIndex, false);
                outputBufferIndex = AMediaCodec_dequeueOutputBuffer(encoder, &bufferInfo, 0);
            }

            if (bufferInfo.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                AMediaCodec_releaseOutputBuffer(encoder, outputBufferIndex, false);
                break;
            }
        }
    });

    inputThread.join();
    outputThread.join();
    

    // ================== 清理资源 ==================
    // 处理剩余输出数据
    while (true) {
        AMediaCodecBufferInfo bufferInfo;
        ssize_t outputBufferIndex = AMediaCodec_dequeueOutputBuffer(encoder, &bufferInfo, 1000);
        if (outputBufferIndex < 0) break;
        
        if (bufferInfo.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
            AMediaCodec_releaseOutputBuffer(encoder, outputBufferIndex, false);
            break;
        }
        
        // 写入剩余数据
        if (muxerStarted) {
            size_t outputBufferSize;
            uint8_t* outputBuffer = AMediaCodec_getOutputBuffer(encoder, outputBufferIndex, &outputBufferSize);
            AMediaMuxer_writeSampleData(muxer, muxerTrackIndex, outputBuffer, &bufferInfo);
        }
        AMediaCodec_releaseOutputBuffer(encoder, outputBufferIndex, false);
    }

    AMediaFormat_delete(format);

    // 释放编码器
    AMediaCodec_stop(encoder);
    AMediaCodec_delete(encoder);
    
    // 释放 Muxer
    if (muxerStarted) {
        AMediaMuxer_stop(muxer);
    }
    AMediaMuxer_delete(muxer);
    close(fd);

    return 0;
}

#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    path_t input0path;
    path_t input1path;
    path_t inputpath;
    path_t outputpath;
    path_t input_videopath;
    int numframe = 0;
    float timestep = 0.5f;
    path_t model = PATHSTR("rife-v2.3");
    std::vector<int> gpuid;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 2;
    int verbose = 0;
    int tta_mode = 0;
    int tta_temporal_mode = 0;
    int uhd_mode = 0;
    path_t pattern_format = PATHSTR("%08d.png");


    int opt;
    while ((opt = getopt(argc, argv, "0:1:i:o:I:n:s:m:g:j:f:vxzuh")) != -1)
    {
        switch (opt)
        {
        case '0':
            input0path = optarg;
            break;
        case '1':
            input1path = optarg;
            break;
        case 'i':
            inputpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
            break;
        case 'I':
            input_videopath = optarg;
            break;
        case 'n':
            numframe = atoi(optarg);
            break;
        case 's':
            timestep = atof(optarg);
            break;
        case 'm':
            model = optarg;
            break;
        case 'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            break;
        case 'f':
            pattern_format = optarg;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'x':
            tta_mode = 1;
            break;
        case 'z':
            tta_temporal_mode = 1;
            break;
        case 'u':
            uhd_mode = 1;
            break;
        case 'h':
        default:
            print_usage();
            return -1;
        }
    }

    if (((input0path.empty() || input1path.empty()) && inputpath.empty() && input_videopath.empty()) || outputpath.empty())
    {
        print_usage();
        return -1;
    }

    if (inputpath.empty() && (timestep <= 0.f || timestep >= 1.f))
    {
        fprintf(stderr, "invalid timestep argument, must be 0~1\n");
        return -1;
    }

    if (!inputpath.empty() && numframe < 0)
    {
        fprintf(stderr, "invalid numframe argument, must not be negative\n");
        return -1;
    }

    if (jobs_load < 1 || jobs_save < 1)
    {
        fprintf(stderr, "invalid thread count argument\n");
        return -1;
    }

    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) && !jobs_proc.empty())
    {
        fprintf(stderr, "invalid jobs_proc thread count argument\n");
        return -1;
    }

    for (int i=0; i<(int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "invalid jobs_proc thread count argument\n");
            return -1;
        }
    }

    path_t pattern = get_file_name_without_extension(pattern_format);
    path_t format = get_file_extension(pattern_format);

    if (format.empty())
    {
        pattern = PATHSTR("%08d");
        format = pattern_format;
    }

    if (pattern.empty())
    {
        pattern = PATHSTR("%08d");
    }

    if (!path_is_directory(outputpath))
    {
        // guess format from outputpath no matter what format argument specified
        path_t ext = get_file_extension(outputpath);

        if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
        {
            format = PATHSTR("png");
        }
        else if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
        {
            format = PATHSTR("webp");
        }
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
        {
            format = PATHSTR("jpg");
        }
        else if (ext == PATHSTR("mp4") || ext == PATHSTR("MP4"))
        {
            format = PATHSTR("mp4");
        }
        else
        {
            fprintf(stderr, "invalid outputpath extension type\n");
            return -1;
        }
    }

    if (format != PATHSTR("png") && format != PATHSTR("webp") && format != PATHSTR("jpg") && format != PATHSTR("mp4"))
    {
        fprintf(stderr, "invalid format argument\n");
        return -1;
    }

    bool rife_v2 = false;
    bool rife_v4 = false;
    if (model.find(PATHSTR("rife-v2")) != path_t::npos)
    {
        // fine
        rife_v2 = true;
    }
    else if (model.find(PATHSTR("rife-v3")) != path_t::npos)
    {
        // fine
        rife_v2 = true;
    }
    else if (model.find(PATHSTR("rife-v4")) != path_t::npos)
    {
        // fine
        rife_v4 = true;
    }
    else if (model.find(PATHSTR("rife")) != path_t::npos)
    {
        // fine
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    if (!rife_v4 && (numframe != 0 || timestep != 0.5))
    {
        fprintf(stderr, "only rife-v4 model support custom numframe and timestep\n");
        return -1;
    }

    // collect input and output filepath
    std::vector<path_t> input0_files;
    std::vector<path_t> input1_files;
    std::vector<path_t> output_files;
    std::vector<float> timesteps;
    bool input_video_flag = false;
    bool save_video_flag = false;
    {
        if (!input_videopath.empty() && !path_is_directory(input_videopath) && path_is_directory(outputpath)) {
            input_video_flag = true;
            save_video_flag = false;
            // printf("use video input\n");
        }
        else if (!input_videopath.empty() && !path_is_directory(input_videopath) && !path_is_directory(outputpath)) {
            input_video_flag = true;
            save_video_flag = true;
        }
        else if (!inputpath.empty() && path_is_directory(inputpath) && path_is_directory(outputpath))
        {
            // -i, -o, input_frames output_frames 一堆图片
            std::vector<path_t> filenames;
            int lr = list_directory(inputpath, filenames);
            if (lr != 0)
                return -1;

            const int count = filenames.size();
            if (numframe == 0)
                numframe = count * 2;

            input0_files.resize(numframe);
            input1_files.resize(numframe);
            output_files.resize(numframe);
            timesteps.resize(numframe);

            double scale = (double)(count - 1.0) / (numframe - 1.0);
            for (int i=0; i<numframe; i++)
            {
                // TODO provide option to control timestep interpolate method
//                 float fx = (float)((i + 0.5) * scale - 0.5);
                float fx = i * scale;
                int sx = static_cast<int>(floor(fx));
                fx -= sx;

                if (sx < 0)
                {
                    sx = 0;
                    fx = 0.f;
                }
                if (sx >= count - 1)
                {
                    sx = count - 2;
                    fx = 1.f;
                }

//                 fprintf(stderr, "%d %f %d\n", i, fx, sx);

                path_t filename0 = filenames[sx];
                path_t filename1 = filenames[sx + 1];

#if _WIN32
                wchar_t tmp[256];
                swprintf(tmp, pattern.c_str(), i+1);
#else
                char tmp[256];
                sprintf(tmp, pattern.c_str(), i+1); // ffmpeg start from 1
#endif
                path_t output_filename = path_t(tmp) + PATHSTR('.') + format;

                input0_files[i] = inputpath + PATHSTR('/') + filename0;
                input1_files[i] = inputpath + PATHSTR('/') + filename1;
                output_files[i] = outputpath + PATHSTR('/') + output_filename;
                timesteps[i] = fx;
            }
        }
        else if (inputpath.empty() && !path_is_directory(input0path) && !path_is_directory(input1path) && !path_is_directory(outputpath))
        {
            // -0, -1, 输入两张图片
            input0_files.push_back(input0path);
            input1_files.push_back(input1path);
            output_files.push_back(outputpath);
            timesteps.push_back(timestep);
        }
        else
        {
            fprintf(stderr, "input0path, input1path and outputpath must be file at the same time\n");
            fprintf(stderr, "inputpath and outputpath must be directory at the same time\n");
            return -1;
        }
    }

    path_t modeldir = sanitize_dirpath(model);

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    int gpu_count = ncnn::get_gpu_count();
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] < -1 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "invalid gpu device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] == -1)
        {
            jobs_proc[i] = std::min(jobs_proc[i], cpu_count);
            total_jobs_proc += 1;
        }
        else
        {
            total_jobs_proc += jobs_proc[i];
        }
    }

    {
        std::vector<RIFE*> rife(use_gpu_count);

        for (int i=0; i<use_gpu_count; i++)
        {
            int num_threads = gpuid[i] == -1 ? jobs_proc[i] : 1;

            rife[i] = new RIFE(gpuid[i], tta_mode, tta_temporal_mode, uhd_mode, num_threads, rife_v2, rife_v4);

            rife[i]->load(modeldir);
        }

        // main routine
        {
            ncnn::Thread *load_thread_ptr = nullptr;
            LoadThreadParams ltp; // 放 if 里面会先被析构掉
            LoadVideoThreadParams lvtp;

            if (!input_video_flag) {
                // load image
                ltp.jobs_load = jobs_load;
                ltp.input0_files = input0_files;
                ltp.input1_files = input1_files;
                ltp.output_files = output_files;
                ltp.timesteps = timesteps;

                // ncnn::Thread load_thread(load, (void*)&ltp);
                load_thread_ptr = new ncnn::Thread(load, (void*)&ltp);
            }
            else {
                // load video
                lvtp.input_videopath = input_videopath;
                lvtp.outputpath = outputpath;
                lvtp.pattern = pattern;
                lvtp.format = format;

                // ncnn::Thread load_thread(load_video, (void*)&ltp);
                load_thread_ptr = new ncnn::Thread(load_video, (void*)&lvtp);
            }

            // rife proc
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (int i=0; i<use_gpu_count; i++)
            {
                ptp[i].rife = rife[i];
            }

            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (int i=0; i<use_gpu_count; i++)
                {
                    if (gpuid[i] == -1)
                    {
                        proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                    }
                    else
                    {
                        for (int j=0; j<jobs_proc[i]; j++)
                        {
                            proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                        }
                    }
                }
            }

            // save image
            SaveThreadParams stp;
            SaveVideoThreadParams svtp;
            if (save_video_flag) {
                // save video
                svtp.output_video_path = outputpath;
                svtp.width = 1280;
                svtp.height = 720;
                svtp.fps = 60;
                jobs_save = 1;
            }
            else {
                // save image
                stp.verbose = verbose;
            }

            std::vector<ncnn::Thread*> save_threads(jobs_save);
            for (int i=0; i<jobs_save; i++)
            {
                if (save_video_flag) {
                    save_threads[i] = new ncnn::Thread(save_video, (void*)&svtp);
                }
                else {
                    save_threads[i] = new ncnn::Thread(save, (void*)&stp);
                }
            }

            // end
            load_thread_ptr->join();

            Task end;
            end.id = -233;

            for (int i=0; i<total_jobs_proc; i++)
            {
                toproc.put(end);
            }

            for (int i=0; i<total_jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

            for (int i=0; i<jobs_save; i++)
            {
                tosave.put(end);
            }

            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i]->join();
                delete save_threads[i];
            }
        }

        for (int i=0; i<use_gpu_count; i++)
        {
            delete rife[i];
        }
        rife.clear();
    }

    ncnn::destroy_gpu_instance();

    return 0;
}
