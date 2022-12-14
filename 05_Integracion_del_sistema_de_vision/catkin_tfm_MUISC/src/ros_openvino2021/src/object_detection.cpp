/* 
 * This file is part of the ros_openvino package (https://github.com/gbr1/ros_openvino or http://gbr1.github.io).
 * Copyright (c) 2019 Giovanni di Dio Bruno.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include <ros/ros.h>
#include <inference_engine.hpp>
#include <ie_iextension.h>
//#include <extension/ext_list.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <ros_openvino/Object.h>
#include <ros_openvino/Objects.h>
#include <ros_openvino/ObjectBox.h>
#include <ros_openvino/ObjectBoxList.h>
#include <string>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sstream>
#include <image_transport/image_transport.h>
#include "std_msgs/Bool.h"


using namespace InferenceEngine;

//Frames required in object detection with async api
cv::Mat frame_now;  
cv::Mat frame_next;
cv::Mat depth_frame; 
cv::Mat frame_mask;

//Parameters for camera calibration
float fx;
float fy;
float cx;
float cy;

//Frames sizes
size_t color_width;
size_t color_height;
size_t depth_width;
size_t depth_height;

//lockers
bool cf_available=false;
bool is_first_frame = true;
bool is_last_frame=true;

//ROS parameters
std::string device;
float confidence_threshold;
std::string network_path;
std::string weights_path;
std::string labels_path;
std::string colors_path;
bool output_as_image;
bool output_as_list;
bool depth_analysis;
bool output_markers;
bool output_markerslabel;
std::string depth_frameid;
float markerduration;
bool output_boxlist;

std::string net_type;
float iou_threshold;
float rate;
ros::Time foto_time;

int ssd_results_number;
int ssd_object_size;
std::string outputName;

//Mask RCNN

size_t mrcnnInputHeight;
size_t mrcnnInputWidth;
std::string mrcnnOutputName;

//ROS messages
sensor_msgs::Image output_image_msg;
sensor_msgs::Image output_mask_msg;
ros_openvino::Object tmp_object;
ros_openvino::Objects results_list;
visualization_msgs::Marker marker;
visualization_msgs::Marker marker_label;
visualization_msgs::MarkerArray markers;
ros_openvino::ObjectBox tmp_box;
ros_openvino::ObjectBoxList box_list;

bool running=true;

//Couple of hex in string format to uint value (FF->255)
uint8_t hexToUint8(std::string s){
    unsigned int x;
    std::stringstream ss;
    ss<<std::hex<<s;
    ss>>x;
    return x;
}

//OpenCV mat to blob
static InferenceEngine::Blob::Ptr mat_to_blob(const cv::Mat &image) {
    InferenceEngine::TensorDesc tensor(InferenceEngine::Precision::U8,{1, (size_t)image.channels(), (size_t)image.size().height, (size_t)image.size().width},InferenceEngine::Layout::NHWC);
    return InferenceEngine::make_shared_blob<uint8_t>(tensor, image.data);
}

//Image to blob
void frame_to_blob(const cv::Mat& image,InferRequest::Ptr& analysis,const std::string& descriptor) {
    /*Blob::Ptr input = analysis->GetBlob(descriptor);
        auto data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
        size_t image_size = mrcnnInputHeight * mrcnnInputWidth;*/

    //ROS_INFO_STREAM("frame_to_blob: Descriptor: " << descriptor);
    /** Iterate over all pixels in image (b,g,r) **/
    //for (size_t pid = 0; pid < image_size; pid++) {
    /** Iterate over all channels **/
    //for (size_t ch = 0; ch < 3; ++ch) {
    /**          [images stride + channels stride + pixel id ] all in bytes            **/
    //    data[ch * image_size + pid] = 128; //image.at((int) (pid * 3 + ch));
    //}
    //}

    analysis->SetBlob(descriptor, mat_to_blob(image));
}




//Callback: a new RGB image is arrived
void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg){


    if (ros::Time::now() - image_msg->header.stamp > ros::Duration(0.5)) {
        ROS_INFO_STREAM("Frame antiguo recibido. Retraso: " << ros::Time::now().toSec() - image_msg->header.stamp.toSec());
        return;
    }

    foto_time=ros::Time::now();
    cv::Mat color_mat(image_msg->height,image_msg->width,CV_MAKETYPE(CV_8U,3),const_cast<uchar*>(&image_msg->data[0]), image_msg->step);
    //cv::cvtColor(color_mat,color_mat,cv::COLOR_BGR2RGB);
    //ROS_INFO("cb: Recogido nuevo frame");
    //if(!cf_available){
    if(is_first_frame){
        //ROS_INFO("cb: Recogido primer frame");
        color_mat.copyTo(frame_now);
        //cf_available=true;
    }
    color_mat.copyTo(frame_next);
    is_last_frame=false;
    color_width  = (size_t)color_mat.size().width;
    color_height = (size_t)color_mat.size().height;
    cf_available=true; // Carlos
}


void cb_running(const std_msgs::Bool::ConstPtr& start_ptr)
{

    bool new_running= start_ptr->data;
    if (new_running) {
        if (running) {
            ROS_INFO("Ejecucion ya iniciada");
        } else {
            ROS_INFO("Iniciando ejecucion");
        }

    } else {
        if (running)
            ROS_INFO("Deteniendo ejecucion. LLevara un tiempo ...");
        else
            ROS_INFO("Ejecucion ya detenida.");
    }
    running=new_running;
}

//Callaback: a new CameraInfo is arrived
void infoCallback(const sensor_msgs::CameraInfo::ConstPtr& info_msg){
    fx=info_msg->K[0];
    fy=info_msg->K[4];
    cx=info_msg->K[2];
    cy=info_msg->K[5];
}

//Callback: a new depth image is arrived
void depthCallback(const sensor_msgs::Image::ConstPtr& depth_msg){
    //cv::Mat depth_mat(depth_msg->height, depth_msg->width, CV_MAKETYPE(CV_16U,1),const_cast<uchar*>(&depth_msg->data[0]), depth_msg->step);
    cv::Mat depth_mat(depth_msg->height, depth_msg->width, CV_MAKETYPE(CV_32F,1),const_cast<uchar*>(&depth_msg->data[0]), depth_msg->step);
    /*float a,b,c,d;
        a=depth_mat.at<float>(427,460);
        b=depth_mat.at<float>(299,438);
        c=depth_mat.at<float>(385,483);
        d=depth_mat.at<float>(288,472);*/

    //depth_mat.copyTo(depth_frame);
    depth_mat.convertTo(depth_frame,CV_16U,1000);
    /*unsigned short  ai, bi, ci,di;
        ai=depth_frame.at<unsigned short>(427,460);
        bi=depth_frame.at<unsigned short>(299,438);
        ci=depth_frame.at<unsigned short>(385,483);
        di=depth_frame.at<unsigned short>(288,472);*/

    //ROS_INFO("A=%f, B=%f, C=%f, D=%f, AI=%u, BI=%u, CI=%u, DI=%u",a,b,c,d,ai,bi,ci,di);

    depth_width  = (size_t)depth_mat.size().width;
    depth_height = (size_t)depth_mat.size().height;
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}


struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;
    std::string label, color;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale, std::string lb, std::string col) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
        this->label=lb;
        this->color=col;
    }

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

void ParseYOLOV3Output(const DataPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<std::string> labels, std::vector<std::string> colors,
                       std::vector<DetectionObject> &objects) {

    // --------------------------- Validating output parameters -------------------------------------
    //if (layer->type != "RegionYolo")
    //	throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
    /*const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
        const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
        if (out_blob_h != out_blob_w)
                throw std::runtime_error("Invalid size of output " + layer->getName() +
                                " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
                                ", current W = " + std::to_string(out_blob_h));
        // --------------------------- Extracting layer parameters -------------------------------------
        auto num = layer->GetParamAsInt("num");
        auto coords = layer->GetParamAsInt("coords");
        auto classes = layer->GetParamAsInt("classes");
        std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                156.0, 198.0, 373.0, 326.0};
        try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
        try {
                auto mask = layer->GetParamAsInts("mask");
                num = mask.size();

                std::vector<float> maskedAnchors(num * 2);
                for (int i = 0; i < num; ++i) {
                        maskedAnchors[i * 2] = anchors[mask[i] * 2];
                        maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
                }
                anchors = maskedAnchors;
        } catch (...) {}

        auto side = out_blob_h;
        auto side_square = side * side;
        const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
        // --------------------------- Parsing YOLO Region output -------------------------------------
        for (int i = 0; i < side_square; ++i) {
                int row = i / side;
                int col = i % side;
                for (int n = 0; n < num; ++n) {
                        int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
                        int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
                        float scale = output_blob[obj_index];
                        if (scale < threshold)
                                continue;
                        double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
                        double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
                        double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[2 * n + 1];
                        double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[2 * n];
                        for (int j = 0; j < classes; ++j) {
                                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                                float prob = scale * output_blob[class_index];
                                if (prob < threshold)
                                        continue;
                                std::string text= (j < labels.size() ? labels[j] : std::string("unknown ") + std::to_string(j));
                                std::string color=(j < colors.size() ? colors[j] : std::string("00FF00"));
                                DetectionObject obj(x, y, height, width, j, prob,
                                                static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                                                static_cast<float>(original_im_w) / static_cast<float>(resized_im_w),text,color);
                                objects.push_back(obj);
                        }
                }
        }*/
}





//Main Function
int main(int argc, char **argv){
    try{
        //Initialize Ros
        ros::init(argc, argv, "object_detection");
        //Handle creation
        ros::NodeHandle n;
        image_transport::ImageTransport it(n);

        //--------- ROS PARAMETERS ----------//

        //target device, by default is MYRIAD due ai core as mainly supported device
        if (n.getParam("/object_detection/target",device)){
            ROS_INFO("Target: %s", device.c_str());
        }
        else{
            device="MYRIAD";
            ROS_INFO("[Default] Target: %s", device.c_str());
        }

        //threshold for confidence
        if (n.getParam("/object_detection/threshold", confidence_threshold)){
            ROS_INFO("Confidence Threshold: %f", confidence_threshold);

        }
        else{
            confidence_threshold=0.5;
            ROS_INFO("[Default] Confidence Threshold: %f", confidence_threshold);
        }

        //network-> model.xml
        if (n.getParam("/object_detection/model_network",network_path)){
            ROS_INFO("Model Network: %s", network_path.c_str());
        }
        else{
            network_path="/opt/intel/computer_vision_sdk/deployment_tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.xml";
            ROS_INFO("[Default] Model Network: %s", network_path.c_str());
        }

        //weight -> model.bin
        if (n.getParam("/object_detection/model_weights", weights_path)){
            ROS_INFO("Model Weights: %s", weights_path.c_str());
        }
        else{
            weights_path="/opt/intel/computer_vision_sdk/deployment_tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.bin";
            ROS_INFO("[Default] Model Weights: %s", weights_path.c_str());
        }

        //labels -> model.labels
        if (n.getParam("/object_detection/model_labels", labels_path)){
            ROS_INFO("Model Labels: %s", labels_path.c_str());
        }
        else{
            labels_path="/opt/intel/computer_vision_sdk/deployment_tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.labels";
            ROS_INFO("[Default] Model Labels: %s", labels_path.c_str());
        }

        //labels -> model.labels
        if (n.getParam("/object_detection/model_colors", colors_path)){
            ROS_INFO("Model Colors: %s", colors_path.c_str());
        }
        else{
            colors_path="";
            ROS_INFO("[Default] Model Colors: %s", colors_path.c_str());
        }

        //check if frame publisher is wanted
        if (n.getParam("/object_detection/output_as_image", output_as_image)){
            ROS_INFO("Publish Analyzed Image Topic: %s", output_as_image ? "true" : "false");
        }
        else{
            output_as_image=true;
            ROS_INFO("[Default] Publish Analyzed Image Topic: %s", output_as_image ? "true" : "false");
        }

        //check if results list publisher is wanted
        if (n.getParam("/object_detection/output_as_list", output_as_list)){
            ROS_INFO("Publish Results List: %s", output_as_list ? "true" : "false");
        }
        else{
            output_as_list=true;
            ROS_INFO("[Default] Publish Results List: %s", output_as_list ? "true" : "false");
        }

        //frame id used as TF reference
        if (n.getParam("/object_detection/frame_id", depth_frameid)){
            ROS_INFO("Frame id used: %s", depth_frameid.c_str());
        }
        else{
            depth_frameid="/camera_link";
            ROS_INFO("[Default] Frame id: %s", depth_frameid.c_str());
        }

        //check if depth analysis is wanted
        if (n.getParam("/object_detection/depth_analysis", depth_analysis)){
            ROS_INFO("Depth Analysis: %s", depth_analysis ? "ENABLED" : "DISABLED");
        }
        else{
            depth_analysis=true;
            ROS_INFO("[Default] Depth Analysis: %s", depth_analysis ? "ENABLED" : "DISABLED");
        }

        if (depth_analysis){
            //check if markers are wanted
            if (n.getParam("/object_detection/output_as_markers", output_markers)){
                ROS_INFO("Output Markers: %s", output_markers ? "true" : "false");
            }
            else{
                output_markers=true;
                ROS_INFO("[Default] Output Markers: %s", output_markers ? "true" : "false");
            }

            //check if markers label are wanted
            if (n.getParam("/object_detection/output_as_markerslabel", output_markerslabel)){
                ROS_INFO("Output Markers Label: %s", output_markerslabel ? "true" : "false");
            }
            else{
                output_markerslabel=true;
                ROS_INFO("[Default] Output Markers Label: %s", output_markerslabel ? "true" : "false");
            }

            if (output_markers||output_markerslabel){
                //lifetime of markers
                if (n.getParam("/object_detection/output_markers_lifetime", markerduration)){
                    ROS_INFO("Output Markers Lifetime: %f", markerduration);
                }
                else{
                    markerduration=0.1;
                    ROS_INFO("[Default] Output Markers Lifetime: %f", markerduration);
                }
            }

            //check if output as box list is wanted
            if (n.getParam("/object_detection/output_as_box_list", output_boxlist)){
                ROS_INFO("Output as Box List: %s", output_boxlist ? "true" : "false");
            }
            else{
                output_boxlist=true;
                ROS_INFO("[Default] Output as Box List: %s", output_boxlist ? "true" : "false");
            }

        }

        // Tipo de Red

        if (n.getParam("/object_detection/net_type", net_type)){
            ROS_INFO("Network Type: %s", net_type.c_str());
        }
        else{
            net_type="SSD";
            ROS_INFO("[Default] Network Type: %s", net_type.c_str());
        }

        // iou threshold
        if (net_type == "YoloV3") {
            if (n.getParam("/object_detection/iou_threshold", iou_threshold)){
                ROS_INFO("IoU Threshold: %f", iou_threshold);

            }
            else{
                iou_threshold=0.4;
                ROS_INFO("[Default] IoU Threshold: %f", iou_threshold);
            }
        } else if (net_type == "MRCNN") {
            if (n.getParam("/object_detection/output_name", mrcnnOutputName)){
                ROS_INFO("MRCNN Output: '%s'", mrcnnOutputName);

            }
            else{
                mrcnnOutputName="detection_output";
                ROS_INFO("[Default] MRCNN Output: '%s'", mrcnnOutputName);
            }
        }


        if (n.getParam("/object_detection/rate", rate)){
            ROS_INFO("Rate: %f", rate);

        }
        else{
            rate=15;
            ROS_INFO("[Default] Rate: %f", rate);
        }

        //ROS subscribers
        image_transport::Subscriber image_sub = it.subscribe("/object_detection/input_image",1,imageCallback);
        ROS_INFO("Version con it");
        //ros::Subscriber image_sub = n.subscribe("/object_detection/input_image",1,imageCallback);
        ros::Subscriber camerainfo_sub;
        ros::Subscriber depth_sub;
        ros::Subscriber running_sub=n.subscribe("/object_detection/running",1,cb_running);;

        //ROS publishers
        image_transport::Publisher image_pub; //ros::Publisher image_pub;
        image_transport::Publisher mask_pub;
        if (output_as_image){
            //image_pub = n.advertise<sensor_msgs::Image>("/object_detection/output_image",1);
            image_pub = it.advertise("/object_detection/output_image",1);
            mask_pub = it.advertise("/object_detection/output_mask",1);
        }

        ros::Publisher result_pub;
        if (output_as_list){
            result_pub = n.advertise<ros_openvino::Objects>("/object_detection/results",1);
        }

        //Depth analysis allow subscription and publishing
        ros::Publisher marker_pub;
        ros::Publisher boxlist_pub;
        if (depth_analysis){
            depth_sub = n.subscribe("/object_detection/input_depth",1,depthCallback);
            camerainfo_sub = n.subscribe("/object_detection/camera_info",1,infoCallback);
            if (output_markers||output_markerslabel){
                marker_pub = n.advertise<visualization_msgs::MarkerArray>("/object_detection/markers", 1);
            }
            if (output_boxlist){
                boxlist_pub = n.advertise<ros_openvino::ObjectBoxList>("/object_detection/box_list", 1);
            }
        }

        foto_time=ros::Time::now();
        // Version nueva

        Core ie;

        //slog::info << "Device info: " << slog::endl;
        //std::cout << ie.GetVersions(device);

        if (device.find("CPU") != std::string::npos) {
            /**
                     * cpu_extensions library is compiled from "extension" folder containing
                     * custom MKLDNNPlugin layer implementations. These layers are not supported
                     * by mkldnn, but they can be useful for inferring custom topologies.
                     **/
            //ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
        }

        //setup inference engine device, default is MYRIAD, but you can choose also GPU or CPU (CPU is not tested)
        //InferencePlugin OpenVino_plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(device);

        //Setup model, weights, labels and colors
        //CNNNetReader network_reader; -- Carlos
        CNNNetwork network_reader;
        network_reader=ie.ReadNetwork(network_path,weights_path);
        //network_reader.ReadWeights(weights_path);
        //network_reader.ReadNetwork(network_path);
        //network_reader.ReadWeights(weights_path);
        std::vector<std::string> vector_labels;
        std::ifstream inputFileLabel(labels_path);
        std::copy(std::istream_iterator<std::string>(inputFileLabel),std::istream_iterator<std::string>(),std::back_inserter(vector_labels));
        std::vector<std::string> vector_colors;
        std::ifstream inputFileColor(colors_path);
        std::copy(std::istream_iterator<std::string>(inputFileColor),std::istream_iterator<std::string>(),std::back_inserter(vector_colors));

        //setup input stuffs - Cambios para MRCNN

        //InputsDataMap input_info(network_reader.getNetwork().getInputsInfo());
        InputsDataMap input_info(network_reader.getInputsInfo());
        InputInfo::Ptr& input_data= input_info.begin()->second;;
        std::string inputName;

        if (net_type == "MRCNN") {

            network_reader.addOutput(mrcnnOutputName);

            for (const auto & inputInfoItem : input_info) {
                if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // first input contains images
                    inputName = inputInfoItem.first;
                    ROS_INFO_STREAM("Capa de entrada: " << inputName);
                    inputInfoItem.second->setPrecision(Precision::U8);
                    inputInfoItem.second->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
                    inputInfoItem.second->getInputData()->setLayout(Layout::NCHW); //NCHW
                } else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // second input contains image info

                    inputInfoItem.second->setPrecision(Precision::FP32);
                    ROS_INFO("Input item de tamano 2");
                } else {
                    ROS_FATAL("Unsupported input shape with size = %d", inputInfoItem.second->getTensorDesc().getDims().size());
                }
            }
            /** network dimensions for image input - Esto hay que verlo mas**/
            const TensorDesc& inputDesc = input_info[inputName]->getTensorDesc();
            IE_ASSERT(inputDesc.getDims().size() == 4);
            size_t mrcnnBatchSize = inputDesc.getDims().at(0);	//getTensorBatch(inputDesc);
            size_t mrcnnInputChannels = inputDesc.getDims().at(1); //getTensorChannels(inputDesc);
            mrcnnInputHeight = inputDesc.getDims().at(2); //getTensorHeight(inputDesc);
            mrcnnInputWidth = inputDesc.getDims().at(3); //getTensorWidth(inputDesc);

            ROS_INFO("Network batch size is %d",mrcnnBatchSize);
            if (mrcnnBatchSize != 1)
                ROS_FATAL("El Batch size debe ser 1");


        } else {

            input_data = input_info.begin()->second;
            inputName = input_info.begin()->first;
            input_data->setPrecision(Precision::U8);

            input_data->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input_data->getInputData()->setLayout(Layout::NHWC);
        }


        //setup output
        OutputsDataMap output_info(network_reader.getOutputsInfo());
        if (net_type == "SSD") {
            DataPtr& output_data = output_info.begin()->second;
            outputName = output_info.begin()->first;
            //const int num_classes = network_reader.getLayerByName(outputName.c_str())->GetParamAsInt("num_classes");
            /*const int num_classes = output_info[outputName]->GetParamAsInt("num_classes");
                        if (vector_labels.size() != num_classes) {
                                if (vector_labels.size() == (num_classes - 1))
                                        vector_labels.insert(vector_labels.begin(), "no-label");
                                else
                                        vector_labels.clear();
                        }*/
            const SizeVector output_dimension = output_data->getTensorDesc().getDims();
            ssd_results_number = output_dimension[2];
            ssd_object_size = output_dimension[3];
            if ((ssd_object_size != 7 || output_dimension.size() != 4)) {
                ROS_FATAL("There is a problem with output dimension");
                return -3;
            }

            output_data->setPrecision(Precision::FP32);
            output_data->setLayout(Layout::NCHW);

        } else if (net_type == "YoloV3") {
            for (auto &output : output_info) {
                output.second->setPrecision(Precision::FP32);
                output.second->setLayout(Layout::NCHW);
            }
        } else if (net_type == "MRCNN") {
            for (auto & item : output_info) {
                ROS_INFO_STREAM("MRCNN: Capa salida: " << item.first);
                item.second->setPrecision(Precision::FP32);
            }

        } else {
            ROS_FATAL("Unknown network type: %s",net_type.c_str());
            return -2;
        }


        // Nueva version

        ExecutableNetwork model_network = ie.LoadNetwork(network_reader, device ,{});
        //        ExecutableNetwork model_network = ie.LoadNetwork(network_reader, device, {});
        //load model into plugin
        //ExecutableNetwork model_network = OpenVino_plugin.LoadNetwork(network_reader.getNetwork(), {});

        //inference request to engine
        InferRequest::Ptr engine_next = model_network.CreateInferRequestPtr();
        InferRequest::Ptr engine_curr = model_network.CreateInferRequestPtr();


        // Aqu?? hay que poner lo de la segunda input del Mask RCNN

        if (net_type == "MRCNN") {

            for (const auto & inputInfoItem : input_info) {
                Blob::Ptr input = engine_next->GetBlob(inputInfoItem.first);
                Blob::Ptr input2 = engine_curr->GetBlob(inputInfoItem.first);

                /** Fill second input tensor with image info **/
                if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {
                    auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                    data[0] = static_cast<float>(mrcnnInputHeight);  // height
                    data[1] = static_cast<float>(mrcnnInputWidth);  // width
                    data[2] = 1;
                    data = input2->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                    data[0] = static_cast<float>(mrcnnInputHeight);  // height
                    data[1] = static_cast<float>(mrcnnInputWidth);  // width
                    data[2] = 1;
                }
            }


        }
        //start stuffs
        //bool is_first_frame = true;
        int markers_size=0;
        ros::Rate loop_rate(rate);
        bool infer_finish=true;
        //loop while roscore is up
        while(ros::ok()){
            if (running) {
                int kmarker=0;
                //if there is a frame
                if (cf_available && infer_finish){

                    //if first frame is available -- Solo al principio de la ejecuci??n
                    if (is_first_frame) {
                        ROS_INFO("Preparando frames iniciales");
                        frame_to_blob(frame_now,engine_curr, inputName);
                    }

                    //if there are other frames
                    if (!is_last_frame) {
                        ROS_INFO("Preparando frame consecutivo");
                        frame_to_blob(frame_next, engine_next, inputName);
                    }

                    if (is_first_frame) {
                        engine_curr->StartAsync();
                    }
                    if (!is_last_frame) {
                        engine_next->StartAsync();
                    }

                    //update internal lockers system
                    if (is_first_frame) {
                        is_first_frame = false;
                    }
                    //set to 0 size of markers and delete entire markersarray

                    markers.markers.clear();
                    infer_finish=false;
                }  // cf_available


                InferenceEngine::StatusCode status;

                status=engine_curr->Wait(IInferRequest::WaitMode::STATUS_ONLY);

                //ROS_INFO("Status: %d", status);

                if (status != RESULT_NOT_READY && status != INFER_NOT_STARTED && !infer_finish) {



                    if (engine_curr->Wait(IInferRequest::WaitMode::RESULT_READY) == OK) {

                        ROS_INFO_STREAM("Infer_finish: " << infer_finish);
                        //if engine give us something
                        //if (OK == engine_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                        //get results stuffs
                        std::vector<DetectionObject> objects;
                        unsigned long resized_im_h = input_info.begin()->second.get()->getTensorDesc().getDims()[0];
                        unsigned long resized_im_w = input_info.begin()->second.get()->getTensorDesc().getDims()[1];

                        if (net_type == "MRCNN") {

                            // ROS_INFO("Procesando MRCNN");

                            const auto do_blob = engine_curr->GetBlob(mrcnnOutputName);
                            const auto do_data = do_blob->buffer().as<float*>();

                            const auto masks_blob = engine_curr->GetBlob("masks");
                            const auto masks_data = masks_blob->buffer().as<float*>();

                            ROS_INFO("BLOBs recogidos");

                            //const float PROBABILITY_THRESHOLD = 0.2f;
                            const float MASK_THRESHOLD = 0.5f;  // threshold used to determine whether mask pixel corresponds to object or to background
                            // amount of elements in each detected box description (batch, label, prob, x1, y1, x2, y2)
                            IE_ASSERT(do_blob->getTensorDesc().getDims().size() == 4);
                            size_t BOX_DESCRIPTION_SIZE = do_blob->getTensorDesc().getDims().back();

                            const TensorDesc& masksDesc = masks_blob->getTensorDesc();
                            IE_ASSERT(masksDesc.getDims().size() == 4);
                            size_t BOXES = masksDesc.getDims().at(0);	//getTensorBatch(masksDesc);
                            size_t C = masksDesc.getDims().at(1); //getTensorChannels(masksDesc);
                            size_t H = masksDesc.getDims().at(2); //getTensorHeight(masksDesc);
                            size_t W = masksDesc.getDims().at(3); //getTensorWidth(masksDesc);

                            //ROS_INFO("MRCNN: Obtenidos %d Boxes", BOXES);

                            size_t box_stride = W * H * C;

                            //M??scara
                            //cv::Mat blackImage(frame_now.cols, frame_now.rows, CV_MAKETYPE(CV_8U,3), cv::Scalar(0, 0, 0));
                            frame_mask = frame_now.clone();
                            frame_mask.setTo(cv::Scalar(0,0,0));


                            for (size_t box = 0; box < BOXES; ++box) {
                                float* box_info = do_data + box * BOX_DESCRIPTION_SIZE;
                                auto batch = static_cast<int>(box_info[0]);
                                if (batch < 0)
                                    break;
                                if (batch >= 1)
                                    ROS_ERROR("Invalid batch ID within detection output box");
                                float prob = box_info[2];
                                float x1 = std::min(std::max(0.0f, box_info[3]), 1.0f);
                                float y1 = std::min(std::max(0.0f, box_info[4]), 1.0f);
                                float x2 = std::min(std::max(0.0f, box_info[5]), 1.0f);
                                float y2 = std::min(std::max(0.0f, box_info[6]), 1.0f);
                                float box_width = std::min(std::max(0.0f, x2 - x1), 1.0f);
                                float box_height = std::min(std::max(0.0f, y2 - y1), 1.0f);
                                double box_x=x1+(x2-x1)/2;
                                double box_y=y1+(y2-y1)/2;

                                int class_id = static_cast<int>(box_info[1] + 1e-6f);
                                if (prob > confidence_threshold) {

                                    std::string text= (class_id < vector_labels.size() ? vector_labels[class_id] : std::string("unknown ") + std::to_string(class_id));
                                    std::string color=(class_id < vector_colors.size() ? vector_colors[class_id] : std::string("00FF00"));


                                    DetectionObject obj(box_x, box_y, box_height, box_width, class_id, prob,
                                                        static_cast<float>(color_height),
                                                        static_cast<float>(color_width),text,color);  // / static_cast<float>(resized_im_h)  / static_cast<float>(resized_im_w)
                                    objects.push_back(obj);

                                    ROS_INFO_STREAM("Detected class " << class_id << "(" << text << ") with probability " << prob << " from batch " << batch
                                                    << ": [" << x1 << ", " << y1 << "], [" << x2 << ", " << y2 << "]");

                                    // Vamos a pintar la m??scara en la imagen

                                    uint8_t col[3];

                                    col[0] = hexToUint8(color.substr(0,2));
                                    col[1] = hexToUint8(color.substr(2,2));
                                    col[2] = hexToUint8(color.substr(4,2));

                                    float* mask_arr = masks_data + box_stride * box + H * W * (class_id - 1);
                                    cv::Mat mask_mat(H, W, CV_32FC1, mask_arr);

                                    int frw=frame_now.cols;
                                    int frh=frame_now.rows;

                                    cv::Rect roi = cv::Rect(static_cast<int>(x1*frw), static_cast<int>(y1*frh), box_width*frw, box_height*frh);
                                    cv::Mat roi_input_img = frame_now(roi);
                                    cv::Mat roi_input_mask = frame_mask(roi);
                                    const float alpha = 0.7f;

                                    cv::Mat resized_mask_mat(box_height*frh, box_width*frw, CV_32FC1);
                                    cv::resize(mask_mat, resized_mask_mat, cv::Size(box_width*frw, box_height*frh));

                                    cv::Mat uchar_resized_mask(box_height*frh, box_width*frw, frame_now.type());

                                    for (int h = 0; h < resized_mask_mat.size().height; ++h)
                                        for (int w = 0; w < resized_mask_mat.size().width; ++w)
                                            for (int ch = 0; ch < uchar_resized_mask.channels(); ++ch)
                                                uchar_resized_mask.at<cv::Vec3b>(h, w)[ch] = resized_mask_mat.at<float>(h, w) > MASK_THRESHOLD ?
                                                            255 * col[ch]: roi_input_img.at<cv::Vec3b>(h, w)[ch];

                                    cv::addWeighted(uchar_resized_mask, alpha, roi_input_img, 1.0f - alpha, 0.0f, roi_input_img);
                                    //cv::rectangle(frame_now, roi, cv::Scalar(0, 0, 1), 1);

                                    //a??adimos a la m??scara
                                    for (int h = 0; h < resized_mask_mat.size().height; ++h)
                                        for (int w = 0; w < resized_mask_mat.size().width; ++w)
                                            for (int ch = 0; ch < uchar_resized_mask.channels(); ++ch)
                                                uchar_resized_mask.at<cv::Vec3b>(h, w)[ch] = resized_mask_mat.at<float>(h, w) > MASK_THRESHOLD ?
                                                            255 * 1: roi_input_mask.at<cv::Vec3b>(h, w)[ch];
                                    cv::addWeighted(uchar_resized_mask, 1.0f, roi_input_mask, 0.0f, 0.0f, roi_input_mask);
                                }
                            }


                        } else if (net_type == "SSD") {
                            const float *compute_results = engine_curr->GetBlob(outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                            //for all results
                            for (int i = 0; i < ssd_results_number; i++) {
                                float result_id = compute_results[i * ssd_object_size + 0];
                                int result_label = static_cast<int>(compute_results[i * ssd_object_size + 1]);
                                //get confidence
                                float result_confidence = compute_results[i * ssd_object_size + 2];

                                if (result_confidence < confidence_threshold)
                                    continue;
                                //get 2d box
                                float result_xmin = compute_results[i * ssd_object_size + 3];
                                float result_ymin = compute_results[i * ssd_object_size + 4];
                                float result_xmax = compute_results[i * ssd_object_size + 5];
                                float result_ymax = compute_results[i * ssd_object_size + 6];
                                //get label and color
                                std::string text= (result_label < vector_labels.size() ? vector_labels[result_label] : std::string("unknown ") + std::to_string(result_label));
                                std::string color=(result_label < vector_colors.size() ? vector_colors[result_label] : std::string("00FF00"));


                                //improve 2d box
                                result_xmin=result_xmin<0.0? 0.0 : result_xmin;
                                result_xmax=result_xmax>1.0? 1.0 : result_xmax;
                                result_ymin=result_ymin<0.0? 0.0 : result_ymin;
                                result_ymax=result_ymax>1.0? 1.0 : result_ymax;
                                double x=result_xmin+(result_xmax-result_xmin)/2;
                                double y=result_ymin+(result_ymax-result_ymin)/2;
                                double width=result_xmax-result_xmin;
                                double height=result_ymax-result_ymin;
                                DetectionObject obj(x, y, height, width, result_label, result_confidence,
                                                    static_cast<float>(color_height),
                                                    static_cast<float>(color_width),text,color);  // / static_cast<float>(resized_im_h)  / static_cast<float>(resized_im_w)
                                objects.push_back(obj);
                            }
                        } else if (net_type=="YoloV3") {
                            for (auto &output : output_info) {
                                auto output_name = output.first;
                                //CNNLayerPtr layer = network_reader.getNetwork().getLayerByName(output_name.c_str());
                                DataPtr layer = output_info[outputName];
                                Blob::Ptr blob = engine_curr->GetBlob(output_name);
                                ParseYOLOV3Output(layer, blob, resized_im_h, resized_im_w, color_height, color_width, confidence_threshold,
                                                  vector_labels, vector_colors, objects);
                            }
                            // Filtering overlapping boxes
                            std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
                            for (size_t i = 0; i < objects.size(); ++i) {
                                if (objects[i].confidence == 0)
                                    continue;
                                for (size_t j = i + 1; j < objects.size(); ++j)
                                    if (IntersectionOverUnion(objects[i], objects[j]) >= iou_threshold)
                                        objects[j].confidence = 0;
                            }

                        }

                        ROS_INFO("Vot a procesar %d objetos", objects.size());
                        for (auto &object : objects) {
                            //check threshold
                            if (object.confidence > confidence_threshold){
                                float r_xmin,r_xmax,r_ymin,r_ymax;
                                r_xmin = object.xmin / (float) color_width;
                                r_xmax = object.xmax / (float) color_width;
                                r_ymin = object.ymin / (float) color_height;
                                r_ymax = object.ymax / (float) color_height;
                                //result topic
                                if (output_as_list){
                                    tmp_object.label=object.label;
                                    tmp_object.confidence=object.confidence;
                                    tmp_object.x=r_xmin;
                                    tmp_object.y=r_ymin;
                                    tmp_object.width=r_xmax-r_xmin;
                                    tmp_object.height=r_ymax-r_ymin;
                                    results_list.objects.push_back(tmp_object);
                                }
                                //rgb color
                                uint8_t colorR = hexToUint8(object.color.substr(0,2));
                                uint8_t colorG = hexToUint8(object.color.substr(2,2));
                                uint8_t colorB = hexToUint8(object.color.substr(4,2));
                                //if depth analysis for 3d world detection is active
                                if (depth_analysis){

                                    cv::Mat subdepthP = depth_frame(cv::Rect(r_xmin*depth_width,r_ymin*depth_height,(r_xmax-r_xmin)*depth_width,(r_ymax-r_ymin)*depth_height));
                                    cv::Mat subdepth_full;
                                    subdepthP.copyTo(subdepth_full);
                                    //cv::rectangle(depth_frame, cv::Point2f(xmin*depth_width, ymin*depth_height), cv::Point2f(xmax*depth_width, ymax*depth_height), 0xffff);
                                    cv::Mat subdepth;
                                    subdepth_full.convertTo(subdepth,CV_8U,0.0390625);

                                    std::vector<std::vector<cv::Point> > contours;
                                    std::vector<cv::Vec4i> hierarchy;

                                    cv::Scalar m=cv::mean(subdepth, subdepth!=0);
                                    cv::threshold(subdepth,subdepth,m[0]*1.0,100,4); // Original: m[0]*2.0

                                    cv::findContours(subdepth,contours,hierarchy,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));
                                    for (int i=0; i<contours.size(); i++){
                                        cv::drawContours(subdepth,contours,i,0xffff,cv::FILLED,8,hierarchy, 0, cv::Point());
                                    }
                                    subdepth.convertTo(subdepth,CV_16U);
                                    subdepth=subdepth+subdepth*256;
                                    cv::bitwise_and(subdepth,subdepth_full,subdepth_full);
                                    cv::Mat mask = cv::Mat(subdepth_full!=0);
                                    cv::Scalar avg, dstd;
                                    cv::meanStdDev(subdepth_full,avg,dstd,mask);

                                    //cv::imshow("ruggero",mask);
                                    //cv::waitKey(1);
                                    //ROS_INFO("m[0]=%f, avg=%f, dstd=%f",(float) m[0],(float) avg[0], (float) dstd[0]);

                                    float box_x=avg[0]/1000.0;
                                    float box_y=-(((r_xmax+r_xmin)/2.0)*depth_width-cx)/fx*avg[0]/1000.0;
                                    float box_z=-(((r_ymax+r_ymin)/2.0)*depth_height-cy)/fy*avg[0]/1000.0;

                                    float box_width = avg[0]*depth_width/fx*(r_xmax-r_xmin)/1000.0;
                                    float box_height= avg[0]*depth_height/fy*(r_ymax-r_ymin)/1000.0;
                                    float box_depth = dstd[0]*2.0/1000.0;

                                    //if output as markers is true show cubes in rviz
                                    if (output_markers){
                                        marker.header.frame_id = depth_frameid;
                                        marker.header.stamp = ros::Time::now();

                                        marker.ns = "objects_box";
                                        marker.id = kmarker;
                                        marker.type = visualization_msgs::Marker::CUBE;
                                        marker.action = visualization_msgs::Marker::ADD;

                                        marker.pose.position.x = box_x;
                                        marker.pose.position.y = box_y;
                                        marker.pose.position.z = box_z;
                                        marker.pose.orientation.x = 0.0;
                                        marker.pose.orientation.y = 0.0;
                                        marker.pose.orientation.z = 0.0;
                                        marker.pose.orientation.w = 1.0;

                                        marker.scale.x = box_depth;
                                        marker.scale.y = box_width;
                                        marker.scale.z = box_height;

                                        marker.color.r = colorR/255.0;
                                        marker.color.g = colorG/255.0;
                                        marker.color.b = colorB/255.0;
                                        marker.color.a = 0.4; //0.2f;

                                        marker.lifetime = ros::Duration(markerduration);
                                        markers.markers.push_back(marker);
                                        kmarker++;
                                    }

                                    //if output as markers is true show flaoting text in rviz
                                    if (output_markerslabel){
                                        char prob[50];
                                        sprintf(prob,": %.0f%%",object.confidence*100);
                                        marker_label.header.frame_id= depth_frameid;
                                        marker_label.header.stamp = foto_time; //ros::Time::now();
                                        marker_label.ns="objects_label";
                                        marker_label.id = kmarker;
                                        marker_label.text = object.label+prob;
                                        marker_label.type=visualization_msgs::Marker::TEXT_VIEW_FACING;
                                        marker_label.action = visualization_msgs::Marker::ADD;
                                        marker_label.pose.position.x = box_x;
                                        marker_label.pose.position.y = box_y;
                                        marker_label.pose.position.z = box_z+box_height/2.0+0.05;
                                        marker_label.pose.orientation.x=0.0;
                                        marker_label.pose.orientation.y=0.0;
                                        marker_label.pose.orientation.z=0.0;
                                        marker_label.pose.orientation.w=1.0;
                                        marker_label.scale.z=0.4; //box_height/2;
                                        marker_label.color.r = colorR/255.0;
                                        marker_label.color.g = colorG/255.0;
                                        marker_label.color.b = colorB/255.0;
                                        marker_label.color.a = 0.8f;
                                        marker_label.lifetime = ros::Duration(markerduration);
                                        markers.markers.push_back(marker_label);
                                        kmarker++;
                                    }

                                    //if you want a topic as list of data
                                    if (output_boxlist){
                                        tmp_box.label=object.label;
                                        tmp_box.confidence=object.confidence;
                                        tmp_box.x=box_x;
                                        tmp_box.y=box_y;
                                        tmp_box.z=box_z;
                                        tmp_box.width=box_width;
                                        tmp_box.height=box_height;
                                        tmp_box.depth=box_depth;
                                        box_list.objectboxes.push_back(tmp_box);
                                    }
                                }

                                //if output is a rgb image
                                if (output_as_image){

                                    //compose a label on the top
                                    std::ostringstream c;
                                    c << ":" << std::fixed << std::setprecision(3) << object.confidence;
                                    cv::putText(frame_now, object.label + c.str(),cv::Point2f(object.xmin, object.ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,cv::Scalar(colorB, colorG, colorR));
                                    cv::rectangle(frame_now, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(colorB, colorG, colorR),2);
                                }
                            }
                        }



                        //result output
                        if (output_as_list){
                            results_list.header.stamp=foto_time; //ros::Time::now();
                            result_pub.publish(results_list);
                            results_list.objects.clear();
                        }

                        //frame output
                        if (output_as_image){
                            output_image_msg.header.stamp=foto_time; //ros::Time::now();
                            output_image_msg.header.frame_id=depth_frameid;
                            output_image_msg.height=frame_now.rows;
                            output_image_msg.width=frame_now.cols;
                            output_image_msg.encoding="bgr8";
                            output_image_msg.is_bigendian=false;
                            output_image_msg.step=frame_now.cols*3;
                            size_t size = output_image_msg.step * frame_now.rows;
                            output_image_msg.data.resize(size);
                            memcpy((char*)(&output_image_msg.data[0]), frame_now.data, size);
                            image_pub.publish(output_image_msg);

                            //m??scara
                            output_mask_msg.header.stamp=foto_time; //ros::Time::now();
                            output_mask_msg.header.frame_id=depth_frameid;
                            output_mask_msg.height=frame_mask.rows;
                            output_mask_msg.width=frame_mask.cols;
                            output_mask_msg.encoding="bgr8";
                            output_mask_msg.is_bigendian=false;
                            output_mask_msg.step=frame_mask.cols*3;
                            size_t mask_size = output_mask_msg.step * frame_mask.rows;
                            output_mask_msg.data.resize(mask_size);
                            memcpy((char*)(&output_mask_msg.data[0]), frame_mask.data, mask_size);
                            mask_pub.publish(output_mask_msg);
                        }

                        //markers output
                        if (depth_analysis&&(output_markers||output_markerslabel)){
                            marker_pub.publish(markers);
                        }

                        //boxlist output
                        if (depth_analysis&&output_boxlist){
                            box_list.header.stamp=foto_time; //ros::Time::now();
                            box_list.header.frame_id=depth_frameid;
                            boxlist_pub.publish(box_list);
                            box_list.objectboxes.clear();
                        }
                    }
                    //call roscore
                    //ros::spinOnce();


                    frame_now = frame_next;
                    frame_next = cv::Mat();
                    engine_curr.swap(engine_next);
                    cf_available=false;
                    is_last_frame=true;
                    infer_finish=true;
                }
            } else {
                infer_finish=true;
                is_first_frame=true;
                is_last_frame=true;
                if (engine_curr->Wait(IInferRequest::WaitMode::RESULT_READY) != OK)
                    ROS_ERROR("Error al finalizar 'engine_curr'");
                if (engine_next->Wait(IInferRequest::WaitMode::RESULT_READY) != OK)
                    ROS_ERROR("Error al finalizar 'engine_next'");
                if (engine_curr->Wait(IInferRequest::WaitMode::RESULT_READY) != OK)
                    ROS_ERROR("Error al finalizar 'engine_curr' - 2");
                if (engine_next->Wait(IInferRequest::WaitMode::RESULT_READY) != OK)
                    ROS_ERROR("Error al finalizar 'engine_next' - 2");

                ROS_INFO_THROTTLE(10,"Ejecucion detenida");
                //engine_curr->Cancel();
                //engine_curr->Cancel();
            }

            ros::spinOnce();
            loop_rate.sleep();

        } // while
    } // try

    //hey! there is something not working here!
    catch(const std::exception& e){
        ROS_ERROR("%s",e.what());
        return -1;
    }
    catch (...) {
         ROS_ERROR("Error desconocido.");
         return -1;
     }
    return 0;
}
//canova play ramen, gomma tamburo
