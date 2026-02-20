#include "irisOptimizationUtils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "HmrInferenceUtils.h" // For SmplResult, MakeFrameName, etc
#include "HmrOverlayHelpers.h"
#include "SmplLBS.h"
#include "ModNetMatte.h"

namespace fs = std::filesystem;

namespace {

// --- Data Structures ---

struct CameraInfo {
    int id;
    cv::Matx33f K;
    std::vector<float> D;
    cv::Matx33f R;
    cv::Vec3f t;
    int width;
    int height;
    std::string video_path;
    std::shared_ptr<cv::VideoCapture> cap;
};

struct Frame3DData {
    int frame_idx;
    // 3D keypoints (x, y, z) * N
    std::vector<cv::Point3f> joints; 
};

// --- Simple JSON Parsers (Manual) ---

std::string ReadFileToString(const std::string& path) {
    std::ifstream t(path);
    if (!t.is_open()) return "";
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

std::vector<float> ParseFloatArray(const std::string& json, const std::string& key) {
    std::vector<float> vals;
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) pos = json.find(key); // try without quotes
    if (pos == std::string::npos) return vals;

    size_t start = json.find('[', pos);
    size_t end = json.find(']', start);
    if (start == std::string::npos || end == std::string::npos) return vals;

    std::string content = json.substr(start + 1, end - start - 1);
    std::stringstream ss(content);
    std::string segment;
    while (std::getline(ss, segment, ',')) {
        try {
            vals.push_back(std::stof(segment));
        } catch (...) {}
    }
    return vals;
}

int ParseInt(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return 0;
    size_t colon = json.find(':', pos);
    if (colon == std::string::npos) return 0;
    
    // Simple parse until comma or closing brace
    size_t val_start = colon + 1;
    while (val_start < json.size() && (isspace(json[val_start]))) val_start++;
    size_t val_end = val_start;
    while (val_end < json.size() && (isdigit(json[val_end]) || json[val_end] == '-')) val_end++;
    
    try {
        return std::stoi(json.substr(val_start, val_end - val_start));
    } catch (...) { return 0; }
}

// Load intrinsics_camX.json
bool LoadIntrinsics(const std::string& path, CameraInfo& cam) {
    std::string s = ReadFileToString(path);
    if (s.empty()) return false;
    
    std::vector<float> k_vec = ParseFloatArray(s, "K");
    if (k_vec.size() >= 9) {
        cam.K = cv::Matx33f(k_vec[0], k_vec[1], k_vec[2],
                            k_vec[3], k_vec[4], k_vec[5],
                            k_vec[6], k_vec[7], k_vec[8]);
    }
    cam.D = ParseFloatArray(s, "D");
    cam.width = ParseInt(s, "width");
    cam.height = ParseInt(s, "height");
    return true;
}

// Load extrinsics.json
void LoadExtrinsics(const std::string& path, std::map<int, CameraInfo>& cams) {
    std::string s = ReadFileToString(path);
    if (s.empty()) return;

    // Very hacky parser for array of objects. 
    // Assumes standard formatting from easy-mocap or similar.
    size_t pos = 0;
    while ((pos = s.find("\"cam_id\"", pos)) != std::string::npos) {
        // Find local block
        size_t block_start = s.rfind('{', pos);
        size_t block_end = s.find('}', pos);
        if (block_start == std::string::npos || block_end == std::string::npos) { pos++; continue; }
        
        std::string block = s.substr(block_start, block_end - block_start + 1);
        int id = ParseInt(block, "cam_id");
        
        if (cams.count(id)) {
            std::vector<float> r_vec = ParseFloatArray(block, "R");
            std::vector<float> t_vec = ParseFloatArray(block, "t");
            if (r_vec.size() == 9) {
                cams[id].R = cv::Matx33f(r_vec[0], r_vec[1], r_vec[2],
                                         r_vec[3], r_vec[4], r_vec[5],
                                         r_vec[6], r_vec[7], r_vec[8]);
            }
            if (t_vec.size() == 3) {
                cams[id].t = cv::Vec3f(t_vec[0], t_vec[1], t_vec[2]);
            }
        }
        pos = block_end + 1;
    }
}

// Load poses.jsonl
std::vector<Frame3DData> LoadPoses(const std::string& path) {
    std::vector<Frame3DData> data;
    std::ifstream infile(path);
    std::string line;
    int line_idx = 0;
    while (std::getline(infile, line)) {
        // Naive assumption: line index correlates to frame index if not specified
        // Or we parse "frame_id"? The sample didn't show frame_id, just "people".
        // We will assume lines are sequential frames.
        
        Frame3DData frame;
        frame.frame_idx = line_idx++;
        
        // Find "joints" inside "people"
        size_t joints_pos = line.find("\"joints\"");
        if (joints_pos != std::string::npos) {
            std::vector<float> vals = ParseFloatArray(line, "joints");
            // Assuming (x,y,z) triplets
            for (size_t i = 0; i + 2 < vals.size(); i += 3) {
                frame.joints.emplace_back(vals[i], vals[i+1], vals[i+2]);
            }
        }
        data.push_back(frame);
    }
    return data;
}

// --- 3D Optimization Logic ---

// Standard mapping from RTMPose (Halpe26) to SMPL joints
// If poses.jsonl is not Halpe26, this mapping needs adjustment.
struct KptMap { int src_idx; int smpl_idx; };
const KptMap kSrcToSmpl[] = {
    {5, 16}, {6, 17}, {7, 18}, {8, 19}, {9, 20}, {10, 21}, // Arms
    {11, 1}, {12, 2}, {13, 4}, {14, 5}, {15, 7}, {16, 8},  // Legs
    // Feet (BigToe/SmallToe/Heel -> Foot) - simplifed to just one if available
    {15, 7}, {16, 8} 
    // Add more if needed and available in 3D data
};

// Optimizes SMPL to fit 3D keypoints directly
bool Smplify3D(SMPLLayer& smpl,
               const std::vector<cv::Point3f>& target_joints,
               SmplResult& result,
               int num_iters = 100,
               float lr = 1e-2,
               float pos_weight = 1.0f,
               float reg_weight = 0.05f) {

    auto device = smpl.v_template.device();
    
    // 1. Prepare Target Tensor
    // We map the 3D target joints to SMPL indices
    std::vector<int> smpl_indices;
    std::vector<float> target_flat;
    
    for (const auto& map : kSrcToSmpl) {
        if (map.src_idx < target_joints.size()) {
            const auto& pt = target_joints[map.src_idx];
            // Filter invalid/zero joints if necessary
            if (pt.x == 0.0f && pt.y == 0.0f && pt.z == 0.0f) continue;
            
            smpl_indices.push_back(map.smpl_idx);
            target_flat.push_back(pt.x);
            target_flat.push_back(pt.y);
            target_flat.push_back(pt.z);
        }
    }
    
    if (target_flat.empty()) return false;

    auto target_tensor = torch::from_blob(target_flat.data(), 
        { (long)smpl_indices.size(), 3 }, torch::kFloat).to(device).clone();
    
    auto smpl_idx_tensor = torch::from_blob(smpl_indices.data(), 
        { (long)smpl_indices.size() }, torch::kInt).to(torch::kLong).to(device).clone();

    // 2. Initialize Params
    // Start with mean pose
    auto pose_aa = torch::zeros({1, 72}, torch::TensorOptions().device(device).dtype(torch::kFloat)).requires_grad_(true);
    auto betas = torch::zeros({1, 10}, torch::TensorOptions().device(device).dtype(torch::kFloat)).requires_grad_(true);
    
    // Initialize translation to the centroid of the hips to help convergence
    cv::Vec3f centroid(0,0,0);
    int count = 0;
    // Indices for hips in source might be 11, 12 usually
    if (11 < target_joints.size() && 12 < target_joints.size()) {
         centroid = (cv::Vec3f(target_joints[11]) + cv::Vec3f(target_joints[12])) * 0.5f;
    } else if (!target_joints.empty()) {
        centroid = target_joints[0];
    }
    
    auto trans = torch::tensor({centroid[0], centroid[1], centroid[2]}, 
        torch::TensorOptions().device(device).dtype(torch::kFloat)).reshape({1,3}).requires_grad_(true);

    // 3. Optimize
    std::vector<torch::Tensor> params = {pose_aa, betas, trans};
    torch::optim::Adam optimizer(params, torch::optim::AdamOptions(lr));

    for (int i = 0; i < num_iters; ++i) {
        optimizer.zero_grad();
        
        auto smpl_out = smpl.forward(betas, pose_aa, trans);
        auto model_joints = smpl_out.joints.squeeze(0); // [24, 3] or [45, 3]
        
        // Select corresponding joints
        auto selected_model = model_joints.index_select(0, smpl_idx_tensor);
        
        // L2 Loss
        auto diff = selected_model - target_tensor;
        auto loss_pos = (diff * diff).sum();
        
        // Regularization
        auto loss_reg = (pose_aa.pow(2).mean() + betas.pow(2).mean());
        
        auto total_loss = loss_pos * pos_weight + loss_reg * reg_weight;
        
        total_loss.backward();
        optimizer.step();
    }
    
    // 4. Save Results
    auto p_cpu = pose_aa.detach().cpu();
    auto b_cpu = betas.detach().cpu();
    auto t_cpu = trans.detach().cpu();
    
    result.pose.assign(p_cpu.data_ptr<float>(), p_cpu.data_ptr<float>() + 72);
    result.shape.assign(b_cpu.data_ptr<float>(), b_cpu.data_ptr<float>() + 10);
    
    // We store the global translation in the 'camera' field for now, 
    // treating it as [tx, ty, tz] instead of [scale, tx, ty]
    // The dataset writer needs to handle this distinction.
    // However, existing HmrInferenceUtils uses [scale, tx, ty] for weak perspective.
    // Since we are doing FULL 3D, we should probably output the translation directly 
    // and let the viewer/trainer handle 3D world space.
    // We will reuse the camera vector but strictly store [tx, ty, tz].
    // Note: The Gaussian trainer usually expects weak-perspective param 'cam' 
    // OR a full 'extrinsics' matrix.
    // To support the existing pipeline, we will calculate the equivalent 
    // weak-perspective camera for *each* view later, 
    // but here we just store the global SMPL translation.
    
    // Actually, let's just store world trans in result.camera for storage
    // [tx, ty, tz]
    result.camera = {t_cpu[0][0].item<float>(), t_cpu[0][1].item<float>(), t_cpu[0][2].item<float>()};
    
    return true;
}

// Helper to create the JSONL record for training
void WriteFolderTrainRecord(std::ofstream& out,
                            int frame_idx,
                            const SmplResult& res_world, // Contains World Trans in .camera
                            const CameraInfo& cam,
                            const std::string& overlay_path,
                            const std::string& crop_path,
                            const cv::Rect& crop_rect) {
    
    // We need to convert the World SMPL to Camera-Local SMPL for the 'cam' params (weak perspective)
    // Or, better, simply provide the camera extrinsics and intrinsics if the trainer supports it.
    // Assuming the standard GaussianAvatar trainer which uses "cam" [s, tx, ty] relative to crop...
    
    // Project the SMPL root (trans) into this camera to get crop center and scale
    cv::Vec3f world_t(res_world.camera[0], res_world.camera[1], res_world.camera[2]);
    
    // Transform point to camera space: P_cam = R * P_world + t
    cv::Vec3f p_cam = cam.R * world_t + cam.t;
    
    float f_geo = (cam.K(0,0) + cam.K(1,1)) * 0.5f;
    float cx = cam.K(0,2);
    float cy = cam.K(1,2);
    
    // Project to image
    float u = (f_geo * p_cam[0] / p_cam[2]) + cx;
    float v = (f_geo * p_cam[1] / p_cam[2]) + cy;
    
    // Calculate weak perspective params [s, tx, ty] for the crop
    // crop is at crop_rect.x, crop_rect.y with size crop_rect.width
    
    float crop_cx = crop_rect.x + crop_rect.width * 0.5f;
    float crop_cy = crop_rect.y + crop_rect.height * 0.5f;
    float crop_size = (float)crop_rect.width;
    
    // Scale s = f_geo / z
    float s = 2.0f * f_geo / (crop_size * p_cam[2]); // Standard HMR definition
    float tx = (u - crop_cx) / (crop_size * s * 0.5f + 1e-9f); // Normalized -1..1 approx? 
    // Check SmplifyLite.cpp definition: 
    // u = s * crop_size/2 * tx + crop_cx -> tx = (u - crop_cx) / (s * crop_size/2)
    // Yes.
    float ty = (v - crop_cy) / (crop_size * s * 0.5f + 1e-9f);

    std::vector<float> cam_params = {s, tx, ty};
    
    // Helper for JSON escaping
    auto escape = [](const std::string& s) {
        std::string o; 
        for(char c:s) { if(c=='\\') o+="\\\\"; else if(c=='"') o+="\\\""; else o+=c; } 
        return o;
    };

    out << "{"
        << "\"frame\":" << frame_idx << ","
        << "\"camera_id\":" << cam.id << ","
        << "\"image\":\"" << escape(overlay_path) << "\","
        << "\"crop\":\"" << escape(crop_path) << "\","
        << "\"img_w\":" << cam.width << ","
        << "\"img_h\":" << cam.height << ","
        << "\"crop_cx\":" << crop_cx << ","
        << "\"crop_cy\":" << crop_cy << ","
        << "\"crop_size\":" << crop_size << ","
        << "\"crop_x0\":" << crop_rect.x << ","
        << "\"crop_y0\":" << crop_rect.y << ","
        << "\"crop_w\":" << crop_rect.width << ","
        << "\"crop_h\":" << crop_rect.height << ","
        << "\"focal_length\":" << f_geo << ","
        << "\"pose\":[";
    for(size_t i=0; i<res_world.pose.size(); ++i) out << (i?",":"") << res_world.pose[i];
    out << "],\"betas\":[";
    for(size_t i=0; i<res_world.shape.size(); ++i) out << (i?",":"") << res_world.shape[i];
    out << "],\"cam\":[";
    for(size_t i=0; i<3; ++i) out << (i?",":"") << cam_params[i];
    // Also save the World Translation separately for advanced trainers
    out << "],\"world_trans\":[" << world_t[0] << "," << world_t[1] << "," << world_t[2];
    out << "]}\n";
}

std::string MakeCropName(int cam_id, int frame_idx) {
    std::ostringstream oss;
    oss << "crops/cam" << cam_id << "_" << std::setw(6) << std::setfill('0') << frame_idx << ".png";
    return oss.str();
}

std::string MakeOverlayName(int cam_id, int frame_idx) {
    std::ostringstream oss;
    oss << "overlays/cam" << cam_id << "_" << std::setw(6) << std::setfill('0') << frame_idx << ".png";
    return oss.str();
}

bool EnsureDir(const std::string& path) {
    std::error_code ec;
    fs::create_directories(path, ec);
    return !ec;
}

} // namespace

bool RunFolderOptimization(const FolderOptimizerOptions& options) {
    std::cout << "Starting Folder Optimization..." << std::endl;
    std::cout << "Input: " << options.input_folder << std::endl;
    
    // 1. Load Metadata
    std::map<int, CameraInfo> cameras;
    
    // Look for intrinsics_camX.json
    for (const auto& entry : fs::directory_iterator(options.input_folder)) {
        std::string fn = entry.path().filename().string();
        if (fn.rfind("intrinsics_cam", 0) == 0 && fn.find(".json") != std::string::npos) {
            // Extract ID
            std::string num = fn.substr(14, fn.find(".json") - 14);
            int id = std::stoi(num);
            CameraInfo cam;
            cam.id = id;
            if (LoadIntrinsics(entry.path().string(), cam)) {
                cameras[id] = cam;
            }
        }
    }
    
    if (cameras.empty()) {
        std::cerr << "No intrinsics_cam*.json found." << std::endl;
        return false;
    }
    
    LoadExtrinsics(options.input_folder + "/extrinsics.json", cameras);
    
    // Load 3D Poses
    std::vector<Frame3DData> poses = LoadPoses(options.input_folder + "/poses.jsonl");
    if (poses.empty()) {
        std::cerr << "No poses found in poses.jsonl" << std::endl;
        return false;
    }
    std::cout << "Loaded " << poses.size() << " frames of 3D pose data." << std::endl;

    // Load Videos
    for (auto& kv : cameras) {
        std::string vid_path = options.input_folder + "/recording_cam" + std::to_string(kv.first) + ".mp4";
        
        if (!fs::exists(vid_path)) {
            // Fallbacks: camX.mp4 or video_X.mp4
            std::string fallback1 = options.input_folder + "/cam" + std::to_string(kv.first) + ".mp4";
            if (fs::exists(fallback1)) {
                vid_path = fallback1;
            } else {
                std::string fallback2 = options.input_folder + "/video_" + std::to_string(kv.first) + ".mp4";
                if (fs::exists(fallback2)) {
                    vid_path = fallback2;
                }
            }
        }
        
        if (fs::exists(vid_path)) {
            kv.second.video_path = vid_path;
            kv.second.cap = std::make_shared<cv::VideoCapture>(vid_path);
            if (!kv.second.cap->isOpened()) {
                std::cerr << "Failed to open video: " << vid_path << std::endl;
            } else {
                std::cout << "Opened camera " << kv.first << ": " << vid_path << std::endl;
            }
        } else {
            std::cerr << "Warning: No video found for camera " << kv.first << std::endl;
        }
    }

    // 2. Setup Models
    SMPLLayer smpl_layer(options.smpl_model_path);
    ModNetMatte modnet({options.modnet_input_size, options.modnet_use_cuda});
    if (options.use_modnet) {
        if (!modnet.Load(options.modnet_model_path)) {
            std::cerr << "Failed to load ModNet." << std::endl;
            return false;
        }
    }

    // 3. Prepare Outputs
    EnsureDir(options.output_dir);
    EnsureDir(options.output_dir + "/crops");
    EnsureDir(options.output_dir + "/overlays");
    if (options.use_modnet) EnsureDir(options.output_dir + "/mattes");
    
    std::ofstream trainfile(options.output_dir + "/gaussian_train.jsonl");

    // 4. Main Loop
    int processed_count = 0;
    
    for (const auto& frame_data : poses) {
        int idx = frame_data.frame_idx;
        if (idx % options.frame_stride != 0) continue;
        
        std::cout << "\rProcessing Frame " << idx << std::flush;
        
        // A. Optimize SMPL for this frame
        SmplResult smpl_res;
        if (!Smplify3D(smpl_layer, frame_data.joints, smpl_res, 
                       options.smplify_iters, options.smplify_lr,
                       options.smplify_pos_weight, options.smplify_reg_weight)) {
            continue;
        }
        
        // B. Process each camera
        for (auto& kv : cameras) {
            CameraInfo& cam = kv.second;
            if (!cam.cap || !cam.cap->isOpened()) continue;
            
            // Sync video (naive seek)
            // Assumes perfect sync. For robust sync, we might need to read frame-by-frame.
            // But 'set' is slow. If frames are sequential, we should just read.
            // Since we iterate sequentially through poses.jsonl, we assume video is sequential.
            // BUT: frame_stride might skip.
            
            // To be safe, we calculate target pos.
            // Note: poses.jsonl frames might not match video frames 1:1 if fps differs,
            // but user said "each frame matches the json id".
            
            // Ideally we just read(); but if stride > 1, we must skip.
            // Since `cap` is stateful, we need to track current pos or seek.
            // Seek is safer for "matching ID".
            cam.cap->set(cv::CAP_PROP_POS_FRAMES, (double)idx);
            
            cv::Mat frame;
            if (!cam.cap->read(frame)) continue;
            
            // 1. Calculate Crop based on projected SMPL
            // Project centroid
            cv::Vec3f world_t(smpl_res.camera[0], smpl_res.camera[1], smpl_res.camera[2]);
            cv::Vec3f p_cam = cam.R * world_t + cam.t;
            if (p_cam[2] <= 0.1f) continue; // Behind camera
            
            float f_geo = (cam.K(0,0) + cam.K(1,1)) * 0.5f;
            float cx = cam.K(0,2);
            float cy = cam.K(1,2);
            float u = (f_geo * p_cam[0] / p_cam[2]) + cx;
            float v = (f_geo * p_cam[1] / p_cam[2]) + cy;
            
            // Determine size: heuristic based on depth
            // 1.7m human at depth Z takes approx f * 1.7 / Z pixels
            float approx_height = f_geo * 1.8f / p_cam[2];
            float crop_size = approx_height * 1.2f; // Padding
            
            cv::Rect crop_rect;
            crop_rect.x = (int)(u - crop_size*0.5f);
            crop_rect.y = (int)(v - crop_size*0.5f);
            crop_rect.width = (int)crop_size;
            crop_rect.height = (int)crop_size;
            
            // Clip
            cv::Rect img_rect(0, 0, frame.cols, frame.rows);
            cv::Rect clipped = crop_rect & img_rect;
            
            if (clipped.width <= 10 || clipped.height <= 10) continue;
            
            cv::Mat crop_img = frame(clipped).clone();
            
            // 2. Matting
            if (options.use_modnet) {
                cv::Mat matte;
                if (modnet.ComputeMatte(crop_img, &matte)) {
                     crop_img = modnet.ApplyMatte(crop_img, matte);
                     // Optionally save matte
                }
            }
            
            // 3. Save
            std::string crop_name = MakeCropName(cam.id, idx);
            std::string overlay_name = MakeOverlayName(cam.id, idx);
            
            cv::imwrite(options.output_dir + "/" + crop_name, crop_img);
            
            // 4. Vis Overlay (Project SMPL mesh)
            if (options.save_visualization) {
                cv::Mat overlay = frame.clone();
                // We need to re-project vertices using Extrinsics
                // SMPL -> World -> Camera
                torch::NoGradGuard no_grad;
                auto betas = torch::from_blob(smpl_res.shape.data(), {1, 10}, torch::kFloat);
                auto pose = torch::from_blob(smpl_res.pose.data(), {1, 72}, torch::kFloat);
                auto trans = torch::tensor({world_t[0], world_t[1], world_t[2]}, torch::kFloat).reshape({1,3});
                
                auto smpl_out = smpl_layer.forward(betas, pose, trans);
                auto verts = smpl_out.vertices.squeeze(0); // [6890, 3]
                
                // Project manually
                // This is slow in CPU loop, but fine for vis
                std::vector<cv::Point2f> points;
                auto v_acc = verts.accessor<float,2>();
                for(int k=0; k<v_acc.size(0); ++k) {
                    cv::Vec3f P(v_acc[k][0], v_acc[k][1], v_acc[k][2]);
                    cv::Vec3f Pc = cam.R * P + cam.t;
                    if (Pc[2] > 0.1f) {
                        float pu = (f_geo * Pc[0] / Pc[2]) + cx;
                        float pv = (f_geo * Pc[1] / Pc[2]) + cy;
                        points.emplace_back(pu, pv);
                    } else {
                        points.emplace_back(-100,-100);
                    }
                }
                
                // Draw
                for(auto& pt : points) {
                    if (pt.x > 0 && pt.x < overlay.cols && pt.y > 0 && pt.y < overlay.rows) {
                        overlay.at<cv::Vec3b>(pt) = cv::Vec3b(0,255,0);
                    }
                }
                cv::imwrite(options.output_dir + "/" + overlay_name, overlay);
            }
            
            // 5. Write Record
            WriteFolderTrainRecord(trainfile, idx, smpl_res, cam, 
                                   overlay_name, crop_name, crop_rect);
        }
        
        processed_count++;
    }
    
    std::cout << "\nFinished. Processed " << processed_count << " frames." << std::endl;
    return true;
}