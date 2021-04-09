#include "OpenPose.hpp"

#pragma warning(push, 0)
    #include <chrono>
    #include <filesystem>
    #include <iomanip>
    #include <numeric>
    #include <stdlib.h>
    #include <string>
    #include <sstream>
    #include <time.h>
    #include <vector>

    #include <boost/throw_exception.hpp>
    #include <boost/archive/text_oarchive.hpp>
    #include <boost/archive/text_iarchive.hpp>
    #include <boost/serialization/vector.hpp>

    #include <opencv2/opencv.hpp>
    #include <opencv2/videoio.hpp>

    //#define OPENPOSE_FLAGS_DISABLE_POSE
    #include <openpose/flags.hpp>
    #include <openpose/headers.hpp>
#pragma warning(pop)

class InputParser
{
    public:
        InputParser (int argc, char **argv)
        {
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }

        const std::string& getCmdOption(const std::string& option) const
        {
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }

        bool cmdOptionExists(const std::string& option) const
        {
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }

    private:
        std::vector<std::string> tokens;
};

std::vector<std::vector<float>> tensorrt(int argc, char** argv)
{
    // TODO: benchmark performance of different settings
    std::cout << "usage: path/to/testopenpose --prototxt path/to/prototxt --caffemodel path/to/caffemodel/ \
                  --save_engine path/to/save_engine --input path/to/input/video --run_mode 0/1/2" << std::endl;
    InputParser cmdparams(argc, argv);
    const std::string& prototxt = cmdparams.getCmdOption("--prototxt");
    const std::string& caffemodel = cmdparams.getCmdOption("--caffemodel");
    const std::string& save_engine = cmdparams.getCmdOption("--save_engine");
    const std::string& videoPath = cmdparams.getCmdOption("--input");
    // tensorrt run mode 0:fp32, 1:fp16, 2:int8
    int run_mode = std::stoi(cmdparams.getCmdOption("--run_mode"));
    int H = std::stoi(cmdparams.getCmdOption("--h"));
    int W = std::stoi(cmdparams.getCmdOption("--w"));

    std::vector<std::string> outputBlobname{ "net_output" };
    std::vector<std::vector<float>> calibratorData;
    calibratorData.resize(3);
    for (size_t i = 0; i < calibratorData.size(); i++) {
        calibratorData[i].resize(3 * H * W);
        for (size_t j = 0; j < calibratorData[i].size(); j++) {
            calibratorData[i][j] = 0.05;
        }
    }

    std::vector<float> result;
    int maxBatchSize = 1;

    // calibratorData not used in tinytrt
    // TODO: add argument for max workspace size depending on GPU 
    OpenPose openpose(prototxt, caffemodel, save_engine, outputBlobname, calibratorData, maxBatchSize, run_mode);

    cv::VideoCapture cap;
    if (!cap.open(videoPath))
    {
        std::cerr << "Video could not be opened for reading." << std::endl;
        std::exit(EXIT_FAILURE);        
    }
    int loopCount = 0;
    long totalDuration = 0;

    std::vector<std::vector<float>> tensorrtKeypoints;
    while (cap.grab())
    {
        int N = maxBatchSize;
        int C = 3;
        std::vector<float> inputData;
        inputData.resize(N * C * H * W);

        std::vector<cv::Mat> imgs(N);
        for (int n = 0; n < N; n++)
        {
            // TODO: for multi-batch processing grab() has to repeated for the remaining N-1 frames
            if (!cap.retrieve(imgs[n]))
            {
                std::cerr << "Image could not be read from video stream." << std::endl;
            }

            loopCount++;
            cv::resize(imgs[n], imgs[n], cv::Size(W, H));
            unsigned char* data = imgs[n].data;

            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < W * H; i++)
                {
                    // TODO: necessary? faster/more compact alternative?
                    inputData[i + c * W * H + n * 3 * H * W] = (float)data[i * 3 + c];
                }
            }
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        openpose.DoInference(inputData, result);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        totalDuration += duration_in_ms.count();

        std::vector<float> result_copy(result);
        tensorrtKeypoints.push_back(result_copy);
        for (size_t i = 0; i < result.size() / 3; i++) {
            cv::circle(imgs[0], cv::Point(result[i * 3], result[i * 3 + 1]), 2, cv::Scalar(0, 255, 0), -1);
        }
        cv::imshow("output", imgs[0]);
        cv::waitKey(1);

        std::cout << "Averaged inference time of TensorRT in ms: " << totalDuration / loopCount << ", person count: " << result.size()/75 << std::endl;
    }

    return tensorrtKeypoints;
}

std::vector<std::vector<float>> openpose(int argc, char** argv)
{
    InputParser cmdparams(argc, argv);
    const std::string& videoPath = cmdparams.getCmdOption("--input");
    cv::VideoCapture cap;
    if (!cap.open(videoPath))
    {
        std::cerr << "Video could not be opened for reading." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::vector<float>> openposeKeypoints;
    try
    {
        // Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{ op::ThreadManagerMode::Asynchronous };
        op::WrapperStructPose wrapperStructPose;
        wrapperStructPose.gpuNumber = 1;
        wrapperStructPose.gpuNumberStart = 0;
        wrapperStructPose.modelFolder = "C:/bundesrat_v4.0.new_checkout_try/data/models/";
        wrapperStructPose.poseModel = op::flagsToPoseModel("BODY_25");
        wrapperStructPose.poseMode = op::flagsToPoseMode(FLAGS_body);
        wrapperStructPose.netInputSize = op::Point<int>(960, 464);
        wrapperStructPose.outputSize = op::Point<int>(960, 464);
        opWrapper.configure(wrapperStructPose);

        // Starting OpenPose
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        int loopCount = 0;
        long totalDuration = 0;

        cv::Mat img;
        while (cap.read(img))
        {
            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(img);

            loopCount++;
            auto startTime = std::chrono::high_resolution_clock::now();
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            totalDuration += duration_in_ms.count();

            if (datumProcessed != nullptr)
            {
                const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
                if (!cvMat.empty())
                {
                    cv::imshow(OPEN_POSE_NAME_AND_VERSION, cvMat);
                    cv::waitKey(1);
                }
                std::cout << "Averaged inference time of OpenPose in ms: " << duration_in_ms.count() << \
                    ", person count: " << datumProcessed->at(0)->poseKeypoints.printSize() << std::endl;

                float* dataPtr = datumProcessed->at(0)->poseKeypoints.getPtr();
                std::vector<float> keypoints;
                for (size_t i = 0; i < datumProcessed->at(0)->poseKeypoints.getVolume(); ++i, ++dataPtr)
                {
                    // OpenPose did not scale the poseKeypoints to the declared outputSize
                    if (i % 3 != 2) {
                        keypoints.push_back(*dataPtr / 2);
                    }
                    else {
                        keypoints.push_back(*dataPtr);
                    }
                }
                openposeKeypoints.push_back(keypoints);
            }
            else
            {
                std::cerr << "datumProcessed == nullptr" << std::endl;
            }
        }
    }
    catch (const std::exception&)
    {
        std::cerr << "OpenPose error occurred" << std::endl;
    }

    cv::destroyWindow(OPEN_POSE_NAME_AND_VERSION);
    return openposeKeypoints;
}

// from https://stackoverflow.com/a/12399290
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T>& v)
{
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

#pragma optimize("", off)
int main(int argc, char** argv)
{
    InputParser cmdparams(argc, argv);
    const std::string videoPath = cmdparams.getCmdOption("--input");
    std::size_t videoPathLastSlashPos = videoPath.find_last_of("\\");
    std::string videoName = videoPath;
    if (videoPathLastSlashPos != std::string::npos) {
        videoName = videoPath.substr(videoPathLastSlashPos + 1);
    }

    std::vector<std::vector<float>> openposeKeypoints;
    std::vector<std::vector<float>> tensorrtKeypoints;
    // check if keypoints have already been computed and saved to file, from https://stackoverflow.com/a/28120993
    if (std::filesystem::exists("tensorrtKeypoints_" + videoName + ".vec"))
    {
        std::ifstream ifs("tensorrtKeypoints_" + videoName + ".vec");
        boost::archive::text_iarchive ia(ifs);
        ia & tensorrtKeypoints;
    }
    else
    {
        tensorrtKeypoints = tensorrt(argc, argv);
        std::ofstream ofs("tensorrtKeypoints_" + videoName + ".vec");
        boost::archive::text_oarchive oa(ofs);
        oa & tensorrtKeypoints;
    }
    
    if (std::filesystem::exists("openposeKeypoints_" + videoName + ".vec"))
    {
        std::ifstream ifs("openposeKeypoints_" + videoName + ".vec");
        boost::archive::text_iarchive ia(ifs);
        ia & openposeKeypoints;
    }
    else
    {
        openposeKeypoints = openpose(argc, argv);
        std::ofstream ofs("openposeKeypoints_" + videoName + ".vec");
        boost::archive::text_oarchive oa(ofs);
        oa & openposeKeypoints;
    }

    int H = std::stoi(cmdparams.getCmdOption("--h"));
    int W = std::stoi(cmdparams.getCmdOption("--w"));
    cv::VideoCapture cap;
    if (!cap.open(videoPath))
    {
        std::cerr << "Video could not be opened for reading." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cv::namedWindow("OpenPose & TensorRT keypoints");
    for (int frameNr = 0; frameNr < std::min(tensorrtKeypoints.size(), openposeKeypoints.size()); ++frameNr)
    {
        std::cout << "person count: TensorRT: " << tensorrtKeypoints[frameNr].size() / 75 << ", OpenPose: " << openposeKeypoints[frameNr].size() / 75 << std::endl;

        // equivalent to a mapping keypoints -> middle point
        std::vector<cv::Point2f> op_midPoints;
        for (int op_person = 0; op_person < openposeKeypoints[frameNr].size() / 75; ++op_person)
        {
            cv::Point2f acc(0.f, 0.f);
            int nr_valid_keypoints = 0;
            for (int keypoint = 0; keypoint < 25; ++keypoint)
            {
                if (openposeKeypoints[frameNr][25 * 3 * op_person + 3 * keypoint] > 0.f &&
                    openposeKeypoints[frameNr][25 * 3 * op_person + 3 * keypoint + 1] > 0.f &&
                    openposeKeypoints[frameNr][25 * 3 * op_person + 3 * keypoint + 2] > 0.f)
                {
                    acc += cv::Point2f(openposeKeypoints[frameNr][25 * 3 * op_person + 3 * keypoint], openposeKeypoints[frameNr][25 * 3 * op_person + 3 * keypoint + 1]);
                    nr_valid_keypoints++;
                }
            }
            op_midPoints.push_back(acc / std::max(1, nr_valid_keypoints));
        }

        std::vector<cv::Point2f> trt_midPoints;
        for (int trt_person = 0; trt_person < tensorrtKeypoints[frameNr].size() / 75; ++trt_person)
        {
            cv::Point2f acc(0.f, 0.f);
            int nr_valid_keypoints = 0;
            for (int keypoint = 0; keypoint < 25; ++keypoint)
            {
                if (tensorrtKeypoints[frameNr][25 * 3 * trt_person + 3 * keypoint] > 0.f &&
                    tensorrtKeypoints[frameNr][25 * 3 * trt_person + 3 * keypoint + 1] > 0.f &&
                    tensorrtKeypoints[frameNr][25 * 3 * trt_person + 3 * keypoint + 2]> 0.f)
                {
                    acc += cv::Point2f(tensorrtKeypoints[frameNr][25 * 3 * trt_person + 3 * keypoint], tensorrtKeypoints[frameNr][25 * 3 * trt_person + 3 * keypoint + 1]);
                    nr_valid_keypoints++;
                }
            }
            trt_midPoints.push_back(acc / std::max(1, nr_valid_keypoints));
        }

        cv::Mat img;
        if (!cap.read(img))
        {
            std::cerr << "video frame could not be read." << std::endl;
        }
        
        std::vector<float> minDistancesFromOpenPose;
        for (int op_person = 0; op_person < openposeKeypoints[frameNr].size() / 75; ++op_person)
        {
            float tmpMinDist = std::numeric_limits<float>::max();
            for (int trt_person = 0; trt_person < tensorrtKeypoints[frameNr].size() / 75; ++trt_person)
            {
                float t = cv::norm(trt_midPoints[trt_person] - op_midPoints[op_person]);
                if (t < tmpMinDist) {
                    tmpMinDist = t;
                }
            }
            minDistancesFromOpenPose.push_back(tmpMinDist);
        }
        std::vector<size_t> distIndices = sort_indexes(minDistancesFromOpenPose);

        cv::resize(img, img, cv::Size(W, H));
        for (int op_person = 0; op_person < openposeKeypoints[frameNr].size() / 75; ++op_person)
        {
            for (int keypoint = 0; keypoint < 25; ++keypoint)
            {
                cv::circle(img, cv::Point(openposeKeypoints[frameNr][75 * op_person + 3 * keypoint],
                    openposeKeypoints[frameNr][75 * op_person + 3 * keypoint + 1]), 2, cv::Scalar(0, 255, 0), -1);
            }
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << minDistancesFromOpenPose[op_person];
            cv::putText(img, stream.str(), op_midPoints[op_person], cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
        for (int trt_person = 0; trt_person < tensorrtKeypoints[frameNr].size() / 75; ++trt_person)
        {
            for (int keypoint = 0; keypoint < 25; ++keypoint)
            {
                cv::circle(img, cv::Point(tensorrtKeypoints[frameNr][75 * trt_person + 3 * keypoint],
                    tensorrtKeypoints[frameNr][75 * trt_person + 3 * keypoint + 1]), 2, cv::Scalar(0, 0, 255), -1);
            }
        }
        cv::imshow("OpenPose & TensorRT keypoints", img);
        cv::waitKey(0);



        int openposeTensorRT_personDifference = openposeKeypoints[frameNr].size() / 75 - tensorrtKeypoints[frameNr].size() / 75;
        //if (openposeTensorRT_personDifference != 0)
        //{
        //    std::vector<size_t> distIndices = sort_indexes(minDistances);
        //    for (int r = 0; r < abs(openposeTensorRT_personDifference); ++r)
        //    {
        //        for (int keypoint = 0; keypoint < 25; ++keypoint)
        //        {
        //            if (openposeTensorRT_personDifference > 0)
        //            {
        //                cv::circle(img, cv::Point(openposeKeypoints[frameNr][{static_cast<int>(*(distIndices.rbegin() + r)), keypoint, 0}],
        //                    openposeKeypoints[frameNr][{static_cast<int>(*(distIndices.rbegin() + r)), keypoint, 1}]), 2, cv::Scalar(0, 255, 0), -1);
        //            }
        //            else
        //            {
        //                /*cv::circle(img, cv::Point(tensorrtKeypoints[frameNr][75*static_cast<int>(*(distIndices.rbegin() + r))+3*keypoint],
        //                    openposeKeypoints[frameNr][75*static_cast<int>(*(distIndices.rbegin()+r))+3*keypoint+1]), 2, cv::Scalar(0, 0, 255), -1);*/
        //            }                    
        //        }
        //    }

        //    cv::imshow("output", img);
        //    cv::waitKey(0);
        //}
    }

    return EXIT_SUCCESS;
}
