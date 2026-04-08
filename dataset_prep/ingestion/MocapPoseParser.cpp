#include "dataset_prep/ingestion/MocapPoseParser.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace dataset_prep {
namespace {

std::string MakeLinePreview(const std::string& line, size_t max_chars = 160u) {
    if (line.size() <= max_chars) {
        return line;
    }
    return line.substr(0u, max_chars) + "...";
}

class JsonCursor {
public:
    explicit JsonCursor(std::string_view text)
        : text_(text) {}

    void SkipWhitespace() {
        while (position_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[position_])) != 0) {
            ++position_;
        }
    }

    bool Consume(char expected) {
        SkipWhitespace();
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    bool Peek(char expected) {
        SkipWhitespace();
        return position_ < text_.size() && text_[position_] == expected;
    }

    bool ParseString(std::string* out_value) {
        SkipWhitespace();
        if (position_ >= text_.size() || text_[position_] != '"') {
            return false;
        }

        ++position_;
        std::string value;
        while (position_ < text_.size()) {
            const char current = text_[position_++];
            if (current == '"') {
                if (out_value != nullptr) {
                    *out_value = std::move(value);
                }
                return true;
            }

            if (current != '\\') {
                value.push_back(current);
                continue;
            }

            if (position_ >= text_.size()) {
                return false;
            }

            const char escaped = text_[position_++];
            switch (escaped) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'u':
                    if (position_ + 4u > text_.size()) {
                        return false;
                    }
                    position_ += 4u;
                    value.push_back('?');
                    break;
                default:
                    return false;
            }
        }
        return false;
    }

    bool ParseNumber(double* out_value) {
        if (out_value == nullptr) {
            return false;
        }

        SkipWhitespace();
        const size_t start = position_;

        if (position_ < text_.size() &&
            (text_[position_] == '-' || text_[position_] == '+')) {
            ++position_;
        }

        bool saw_digit = false;
        while (position_ < text_.size() &&
               std::isdigit(static_cast<unsigned char>(text_[position_])) != 0) {
            ++position_;
            saw_digit = true;
        }

        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            while (position_ < text_.size() &&
                   std::isdigit(static_cast<unsigned char>(text_[position_])) != 0) {
                ++position_;
                saw_digit = true;
            }
        }

        if (!saw_digit) {
            return false;
        }

        if (position_ < text_.size() &&
            (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() &&
                (text_[position_] == '-' || text_[position_] == '+')) {
                ++position_;
            }
            bool exponent_digit = false;
            while (position_ < text_.size() &&
                   std::isdigit(static_cast<unsigned char>(text_[position_])) != 0) {
                ++position_;
                exponent_digit = true;
            }
            if (!exponent_digit) {
                return false;
            }
        }

        try {
            *out_value = std::stod(std::string(text_.substr(start, position_ - start)));
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    bool SkipValue() {
        SkipWhitespace();
        if (position_ >= text_.size()) {
            return false;
        }

        const char current = text_[position_];
        if (current == '{') {
            ++position_;
            SkipWhitespace();
            if (Consume('}')) {
                return true;
            }
            while (true) {
                if (!ParseString(nullptr) || !Consume(':') || !SkipValue()) {
                    return false;
                }
                if (Consume('}')) {
                    return true;
                }
                if (!Consume(',')) {
                    return false;
                }
            }
        }

        if (current == '[') {
            ++position_;
            SkipWhitespace();
            if (Consume(']')) {
                return true;
            }
            while (true) {
                if (!SkipValue()) {
                    return false;
                }
                if (Consume(']')) {
                    return true;
                }
                if (!Consume(',')) {
                    return false;
                }
            }
        }

        if (current == '"') {
            return ParseString(nullptr);
        }

        if (std::isdigit(static_cast<unsigned char>(current)) != 0 ||
            current == '-' || current == '+') {
            double ignored_value = 0.0;
            return ParseNumber(&ignored_value);
        }

        if (text_.substr(position_, 4u) == "true") {
            position_ += 4u;
            return true;
        }
        if (text_.substr(position_, 5u) == "false") {
            position_ += 5u;
            return true;
        }
        if (text_.substr(position_, 4u) == "null") {
            position_ += 4u;
            return true;
        }

        return false;
    }

private:
    std::string_view text_;
    size_t position_ = 0u;
};

bool ReadInteger(double numeric_value, int* out_value) {
    if (out_value == nullptr) {
        return false;
    }
    *out_value = static_cast<int>(numeric_value);
    return true;
}

bool ParseNumericArray(JsonCursor* cursor, std::vector<double>* out_values) {
    if (cursor == nullptr || out_values == nullptr || !cursor->Consume('[')) {
        return false;
    }

    out_values->clear();
    if (cursor->Consume(']')) {
        return true;
    }

    while (true) {
        double value = 0.0;
        if (!cursor->ParseNumber(&value)) {
            return false;
        }
        out_values->push_back(value);
        if (cursor->Consume(']')) {
            return true;
        }
        if (!cursor->Consume(',')) {
            return false;
        }
    }
}

bool ParseJointCentersValue(JsonCursor* cursor,
                            std::array<Joint3D, kMocapJointCount>* out_joints) {
    if (cursor == nullptr || out_joints == nullptr || !cursor->Consume('[')) {
        return false;
    }

    if (cursor->Consume(']')) {
        return false;
    }

    if (cursor->Peek('[')) {
        size_t joint_index = 0u;
        while (true) {
            std::vector<double> values;
            if (!ParseNumericArray(cursor, &values) || values.size() < 3u ||
                joint_index >= kMocapJointCount) {
                return false;
            }

            (*out_joints)[joint_index].xyz.x = static_cast<float>(values[0]);
            (*out_joints)[joint_index].xyz.y = static_cast<float>(values[1]);
            (*out_joints)[joint_index].xyz.z = static_cast<float>(values[2]);
            (*out_joints)[joint_index].confidence =
                values.size() >= 4u ? static_cast<float>(values[3]) : 1.0f;
            ++joint_index;

            if (cursor->Consume(']')) {
                return joint_index == kMocapJointCount;
            }
            if (!cursor->Consume(',')) {
                return false;
            }
        }
    }

    std::vector<double> values;
    while (true) {
        double value = 0.0;
        if (!cursor->ParseNumber(&value)) {
            return false;
        }
        values.push_back(value);
        if (cursor->Consume(']')) {
            break;
        }
        if (!cursor->Consume(',')) {
            return false;
        }
    }

    if (values.size() < kMocapJointCount * 3u) {
        return false;
    }

    const size_t values_per_joint = values.size() / kMocapJointCount;
    if (values_per_joint < 3u) {
        return false;
    }

    for (size_t joint_index = 0u; joint_index < kMocapJointCount; ++joint_index) {
        const size_t base_index = joint_index * values_per_joint;
        (*out_joints)[joint_index].xyz.x = static_cast<float>(values[base_index + 0u]);
        (*out_joints)[joint_index].xyz.y = static_cast<float>(values[base_index + 1u]);
        (*out_joints)[joint_index].xyz.z = static_cast<float>(values[base_index + 2u]);
        (*out_joints)[joint_index].confidence =
            values_per_joint >= 4u ? static_cast<float>(values[base_index + 3u]) : 1.0f;
    }
    return true;
}

bool ParseJointAnglesValue(JsonCursor* cursor,
                           std::array<Joint3D, kMocapJointCount>* out_joints) {
    if (cursor == nullptr || out_joints == nullptr || !cursor->Consume('[')) {
        return false;
    }

    if (cursor->Consume(']')) {
        return false;
    }

    if (cursor->Peek('[')) {
        size_t joint_index = 0u;
        while (true) {
            std::vector<double> values;
            if (!ParseNumericArray(cursor, &values) || values.size() < 4u ||
                joint_index >= kMocapJointCount) {
                return false;
            }

            (*out_joints)[joint_index].quaternion = cv::Vec4f(
                static_cast<float>(values[0]),
                static_cast<float>(values[1]),
                static_cast<float>(values[2]),
                static_cast<float>(values[3]));
            ++joint_index;

            if (cursor->Consume(']')) {
                return joint_index == kMocapJointCount;
            }
            if (!cursor->Consume(',')) {
                return false;
            }
        }
    }

    std::vector<double> values;
    while (true) {
        double value = 0.0;
        if (!cursor->ParseNumber(&value)) {
            return false;
        }
        values.push_back(value);
        if (cursor->Consume(']')) {
            break;
        }
        if (!cursor->Consume(',')) {
            return false;
        }
    }

    if (values.size() < kMocapJointCount * 4u) {
        return false;
    }

    const size_t values_per_joint = values.size() / kMocapJointCount;
    if (values_per_joint < 4u) {
        return false;
    }

    for (size_t joint_index = 0u; joint_index < kMocapJointCount; ++joint_index) {
        const size_t base_index = joint_index * values_per_joint;
        (*out_joints)[joint_index].quaternion = cv::Vec4f(
            static_cast<float>(values[base_index + 0u]),
            static_cast<float>(values[base_index + 1u]),
            static_cast<float>(values[base_index + 2u]),
            static_cast<float>(values[base_index + 3u]));
    }
    return true;
}

bool ParsePerson(JsonCursor* cursor, MocapPerson3D* out_person) {
    if (cursor == nullptr || out_person == nullptr || !cursor->Consume('{')) {
        return false;
    }

    MocapPerson3D person;
    bool has_joints = false;

    if (cursor->Consume('}')) {
        return false;
    }

    while (true) {
        std::string key;
        if (!cursor->ParseString(&key) || !cursor->Consume(':')) {
            return false;
        }

        if (key == "person_id" || key == "id") {
            double value = 0.0;
            if (!cursor->ParseNumber(&value) || !ReadInteger(value, &person.person_id)) {
                return false;
            }
        } else if (key == "score" || key == "bbox_score") {
            double value = 0.0;
            if (!cursor->ParseNumber(&value)) {
                return false;
            }
            person.score = static_cast<float>(value);
        } else if (key == "joint_centers" || key == "pose_keypoints_3d" || key == "keypoints_3d") {
            if (!ParseJointCentersValue(cursor, &person.joints)) {
                return false;
            }
            has_joints = true;
        } else if (key == "joint_angles") {
            if (!ParseJointAnglesValue(cursor, &person.joints)) {
                return false;
            }
        } else {
            if (!cursor->SkipValue()) {
                return false;
            }
        }

        if (cursor->Consume('}')) {
            break;
        }
        if (!cursor->Consume(',')) {
            return false;
        }
    }

    if (!has_joints) {
        return false;
    }
    *out_person = std::move(person);
    return true;
}

bool ParsePeopleArray(JsonCursor* cursor, std::vector<MocapPerson3D>* out_people) {
    if (cursor == nullptr || out_people == nullptr || !cursor->Consume('[')) {
        return false;
    }

    out_people->clear();
    if (cursor->Consume(']')) {
        return true;
    }

    while (true) {
        MocapPerson3D person;
        if (!ParsePerson(cursor, &person)) {
            return false;
        }
        out_people->push_back(std::move(person));
        if (cursor->Consume(']')) {
            return true;
        }
        if (!cursor->Consume(',')) {
            return false;
        }
    }
}

}  // namespace

bool MocapPoseParser::Parse(const std::filesystem::path& jsonl_path,
                            MocapSequence3D* out_sequence,
                            const ParseOptions& options) const {
    if (out_sequence == nullptr) {
        return false;
    }

    std::ifstream input(jsonl_path);
    if (!input.is_open()) {
        std::cerr << "MocapPoseParser: failed to open " << jsonl_path.string() << std::endl;
        return false;
    }

    MocapSequence3D sequence;
    sequence.source_path = jsonl_path;

    std::string line;
    size_t line_number = 0u;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        MocapFrame3D frame;
        if (!ParseLine(line, &frame, options.timestamp_scale, line_number)) {
            continue;
        }
        if (options.skip_empty_people && frame.people.empty()) {
            continue;
        }

        sequence.frames_by_seq[frame.frame_seq] = std::move(frame);
    }

    for (const auto& entry : sequence.frames_by_seq) {
        const double timestamp_ms = entry.second.ReferenceTimestampMs();
        if (timestamp_ms >= 0.0) {
            sequence.frame_seq_by_timestamp_ms.emplace(timestamp_ms, entry.first);
        }
    }

    *out_sequence = std::move(sequence);
    return !out_sequence->frames_by_seq.empty();
}

bool MocapPoseParser::ParseLine(const std::string& line,
                                MocapFrame3D* out_frame,
                                double timestamp_scale,
                                size_t line_number) const {
    if (out_frame == nullptr) {
        return false;
    }

    try {
        JsonCursor cursor(line);
        if (!cursor.Consume('{')) {
            std::cerr << "MocapPoseParser: failed to open JSON on line "
                      << line_number << ": " << MakeLinePreview(line) << std::endl;
            return false;
        }

        MocapFrame3D frame;
        bool has_frame_seq = false;

        if (!cursor.Consume('}')) {
            while (true) {
                std::string key;
                if (!cursor.ParseString(&key) || !cursor.Consume(':')) {
                    throw std::runtime_error("invalid JSON object structure");
                }

                if (key == "frame_seq") {
                    double value = 0.0;
                    if (!cursor.ParseNumber(&value) || !ReadInteger(value, &frame.frame_seq)) {
                        throw std::runtime_error("invalid frame_seq");
                    }
                    has_frame_seq = true;
                } else if (key == "timestamp") {
                    double value = 0.0;
                    if (!cursor.ParseNumber(&value)) {
                        throw std::runtime_error("invalid timestamp");
                    }
                    frame.timestamp_ms = value * timestamp_scale;
                } else if (key == "slot_timestamp") {
                    double value = 0.0;
                    if (!cursor.ParseNumber(&value)) {
                        throw std::runtime_error("invalid slot_timestamp");
                    }
                    frame.slot_timestamp_ms = value * timestamp_scale;
                } else if (key == "people") {
                    if (!ParsePeopleArray(&cursor, &frame.people)) {
                        throw std::runtime_error("invalid people array");
                    }
                } else {
                    if (!cursor.SkipValue()) {
                        throw std::runtime_error("failed to skip unsupported field");
                    }
                }

                if (cursor.Consume('}')) {
                    break;
                }
                if (!cursor.Consume(',')) {
                    throw std::runtime_error("expected ',' or '}'");
                }
            }
        }

        if (!has_frame_seq) {
            std::cerr << "MocapPoseParser: missing frame_seq on line "
                      << line_number << ": " << MakeLinePreview(line) << std::endl;
            return false;
        }

        *out_frame = std::move(frame);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "MocapPoseParser: failed to parse line "
                  << line_number << ": " << e.what() << std::endl
                  << "  preview: " << MakeLinePreview(line) << std::endl;
        return false;
    }
}

}  // namespace dataset_prep
