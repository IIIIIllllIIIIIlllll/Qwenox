#include "vision.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}

std::vector<float> read_npy_f32(const std::string& path) {
    std::vector<uint8_t> bytes = read_bytes(path);
    if (bytes.size() < 12 || std::string(reinterpret_cast<char*>(bytes.data() + 1), 5) != "NUMPY")
        return {};
    const int major = bytes[6];
    size_t header = 0, data = 0;
    if (major == 1) {
        header = static_cast<size_t>(bytes[8]) | (static_cast<size_t>(bytes[9]) << 8);
        data = 10 + header;
    } else {
        header = static_cast<size_t>(bytes[8]) | (static_cast<size_t>(bytes[9]) << 8) |
                 (static_cast<size_t>(bytes[10]) << 16) |
                 (static_cast<size_t>(bytes[11]) << 24);
        data = 12 + header;
    }
    if (data > bytes.size() || (bytes.size() - data) % sizeof(float) != 0) return {};
    std::vector<float> result((bytes.size() - data) / sizeof(float));
    std::memcpy(result.data(), bytes.data() + data, result.size() * sizeof(float));
    return result;
}

std::string base64_encode(const std::vector<uint8_t>& input) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve((input.size() + 2) / 3 * 4);
    for (size_t i = 0; i < input.size(); i += 3) {
        const uint32_t value = static_cast<uint32_t>(input[i]) << 16 |
                               (i + 1 < input.size()
                                    ? static_cast<uint32_t>(input[i + 1]) << 8
                                    : 0u) |
                               (i + 2 < input.size() ? input[i + 2] : 0u);
        output.push_back(alphabet[(value >> 18) & 63]);
        output.push_back(alphabet[(value >> 12) & 63]);
        output.push_back(i + 1 < input.size() ? alphabet[(value >> 6) & 63] : '=');
        output.push_back(i + 2 < input.size() ? alphabet[value & 63] : '=');
    }
    return output;
}

void check(bool ok, const std::string& name, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " ", detail.c_str());
    if (!ok) std::exit(1);
}

// 4×4纹理由 Pillow 12.3.0/libwebp 1.6.0 生成，无需运行时图片依赖。
constexpr char kRgbPng[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAHElEQVR4nGMU1I62ldWHIBZxd2FxWSjC"
    "xXEXBgCy3AY88QbffgAAAABJRU5ErkJggg==";
constexpr char kJpeg[] =
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsN"
    "DhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQU"
    "FBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAAEAAQDASIAAhEBAxEB/8QA"
    "FQABAQAAAAAAAAAAAAAAAAAAAAf/xAAeEAABBAMBAQEAAAAAAAAAAAACAQMEEQAFBjEYIf/EABUBAQEA"
    "AAAAAAAAAAAAAAAAAAUH/8QAHREAAgIDAAMAAAAAAAAAAAAAAQMCBAAFQREhMv/aAAwDAQACEQMRAD8A"
    "lv0L0/Jc9zcPnoul0cI9aDpx4+sbdEnFcNFK3kMroRSkWvzz22MYDuNhcVfatbpCIPoCRA5zzlT1ldJp"
    "JJgPkcGf/9k=";
constexpr char kVp8[] =
    "UklGRlYAAABXRUJQVlA4IEoAAACwAQCdASoEAAQAAUAmJagCdAEO+KbQAP3HBM/f6OT0j9jWC6C/m0SP"
    "N/onDHyNsQoYcUHb5Gsy/mS7LvuHe9nIziE/gc0vKBKAAA==";
constexpr char kVp8RgbPng[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAPElEQVR4nGPkUnW11jJyUNA1UdFhcTK3"
    "15SUtdHV15SWY3HUtTZWkTWVEprQXM2oKx9lrCJjoyh45sA2AO8DCw0u/J5CAAAAAElFTkSuQmCC";
constexpr char kVp8l[] =
    "UklGRmwAAABXRUJQVlA4TF8AAAAvA8AAAH9gkG0kheOYyB/d8zzIZ2xqIkmK7gUc4GBD/GsZIdSFSiJJ"
    "cvjFD3Txi3j/UkbD4fmP/AroDAAA0BmAuZcSJXeMEiX7nq8PBAEQA0qUuJGjRIkCVxH9DzMTegA=";
constexpr char kVp8lAlpha[] =
    "UklGRngAAABXRUJQVlA4TGsAAAAvA8AAEH9gkG0kheOYyB/d8zzIZ2xqIkmK7gUc4GBD/GsZIdSFSiJJ"
    "cvjFD3Txi3j/UkbDYQMBy9BszX7NNBP5NcCdGWAAgM4UYO59pSnljp1SlLrv6esXBAEQA0qUuJGjRIkC"
    "VxH9DzMTegA=";
constexpr char kVp8xAlpha[] =
    "UklGRpIAAABXRUJQVlA4WAoAAAAQAAAAAwAAAwAAQUxQSA8AAAABEAgQ0qQ5txERgd2I3QgAVlA4IFwA"
    "AAAwAgCdASoEAAQAAUAmJbACdEyATaADX/BngAD9xwTP3+jk9I/Y1gugv5tCT/M7fzp/+LUud3cunb4d"
    "o5MVEXy6vxO/Q3wA/8uP3AHuZM/7MNz9ef/TXW8X0MAAAA==";
constexpr char kVp8xAlphaRgbPng[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAPUlEQVR4nGMUVTa21rBykLU1VTZi8TT0"
    "VZeSddZR15IUZfFRCjdUF1SR+7G0JYFl6YRV1zU4bTVfPr99BAAZhQ76D27NDAAAAABJRU5ErkJggg==";

void test_decode_images() {
    struct ImageCase {
        const char* name;
        const char* mime;
        const char* encoded;
        const char* rgb_png;
    };
    // 对照取 WebPDecodeRGBInto 输出；透明像素只丢弃 alpha，不合成背景。
    const ImageCase cases[] = {
        {"PNG", "png", kRgbPng, nullptr},
        {"JPEG", "jpeg", kJpeg, nullptr},
        {"VP8有损", "webp", kVp8, kVp8RgbPng},
        {"VP8L无损", "webp", kVp8l, kRgbPng},
        {"VP8L透明", "webp", kVp8lAlpha, kRgbPng},
        {"VP8X+ALPH透明", "webp", kVp8xAlpha, kVp8xAlphaRgbPng},
    };
    for (const auto& image : cases) {
        const std::string name = image.name;
        std::string error;
        std::vector<uint8_t> bytes;
        check(vision::decode_image_url(std::string("data:image/") + image.mime +
                                           ";base64," + image.encoded,
                                       &bytes, &error) &&
                  base64_encode(bytes) == image.encoded,
              name + " 数据URL", error);
        vision::Frame frame;
        check(vision::preprocess(bytes, &frame, &error), name + " 解码", error);
        check(frame.grid == std::array<int, 3>{1, 16, 16}, name + " grid尺寸");
        check(frame.patches.size() == 256u * 256u * 3u * 2u &&
                  std::all_of(frame.patches.begin(), frame.patches.end(),
                              [](float value) { return std::isfinite(value); }),
              name + " patches尺寸和有限值");
        if (image.rgb_png != nullptr) {
            std::vector<uint8_t> reference_bytes;
            vision::Frame reference;
            check(vision::decode_image_url(image.rgb_png, &reference_bytes, &error) &&
                      vision::preprocess(reference_bytes, &reference, &error),
                  name + " RGB对照PNG", error);
            check(frame.grid == reference.grid &&
                      frame.patches.size() == reference.patches.size(),
                  name + " 对照grid和patches尺寸");
            check(std::memcmp(frame.patches.data(), reference.patches.data(),
                              frame.patches.size() * sizeof(float)) == 0,
                  name + " RGB对照逐位一致");
        }
        if (std::strcmp(image.mime, "webp") != 0) continue;

        const auto reject_and_recover = [&](const std::vector<uint8_t>& bad,
                                           const std::string& reason) {
            vision::Frame recovered;
            error.clear();
            check(!vision::preprocess(bad, &recovered, &error) && !error.empty(),
                  name + " " + reason + "拒绝", error);
            error.clear();
            check(vision::preprocess(bytes, &recovered, &error),
                  name + " " + reason + "后有效图成功", error);
            check(recovered.grid == frame.grid &&
                      recovered.patches.size() == frame.patches.size() &&
                      std::memcmp(recovered.patches.data(), frame.patches.data(),
                                  frame.patches.size() * sizeof(float)) == 0,
                  name + " " + reason + "后输出不变");
        };
        std::vector<uint8_t> bad = bytes;
        bad.resize(bad.size() - 8);
        reject_and_recover(bad, "截断");
        if (image.encoded == kVp8) {
            bad = bytes;
            bad[0] = 'X';
            reject_and_recover(bad, "损坏RIFF");
            bad = bytes;
            bad[8] = 'X';
            reject_and_recover(bad, "损坏WEBP");
            bad = {'R', 'I', 'F', 'F', 4, 0, 0, 0, 'W', 'E', 'B', 'P'};
            reject_and_recover(bad, "空WebP容器");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool decode_only = argc == 2 && std::string(argv[1]) == "--decode-only";
    if (!decode_only && argc != 3) {
        std::fprintf(stderr,
                     "usage: vision_test --decode-only\n"
                     "       vision_test <data/vision-ref> <tokenizer-dir>\n");
        return 2;
    }
    test_decode_images();
    if (decode_only) {
        std::puts("RESULT PASS");
        return 0;
    }
    const std::string root = argv[1];
    for (const char* name : {"text", "landscape", "table"}) {
        const std::vector<uint8_t> encoded = read_bytes(root + "/" + name + ".png");
        vision::Frame frame;
        std::string error;
        check(vision::preprocess(encoded, &frame, &error), std::string(name) + " preprocess", error);
        const std::vector<float> expected =
            read_npy_f32(root + "/" + name + ".patches.npy");
        check(frame.grid == std::array<int, 3>{1, 68, 120},
              std::string(name) + " grid");
        check(frame.patches.size() == expected.size(), std::string(name) + " shape");
        float max_diff = 0.0f;
        size_t differing = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            const float diff = std::abs(frame.patches[i] - expected[i]);
            max_diff = std::max(max_diff, diff);
            if (diff != 0.0f) ++differing;
        }
        check(max_diff <= 1e-6f, std::string(name) + " pixels",
              "max_diff=" + std::to_string(max_diff) +
                  " differing=" + std::to_string(differing));
    }

    const std::vector<uint8_t> png = read_bytes(root + "/text.png");
    const std::string encoded = base64_encode(png);
    std::vector<uint8_t> decoded;
    std::string error;
    check(vision::decode_image_url(encoded, &decoded, &error) && decoded == png,
          "bare-base64", error);
    check(vision::decode_image_url("data:image/png;base64," + encoded, &decoded, &error) &&
              decoded == png,
          "data-url", error);
    check(!vision::decode_image_url("https://example.invalid/image.png", &decoded, &error) &&
              error.find("no outbound requests") != std::string::npos,
          "http-url-rejected", error);
    check(!vision::decode_image_url("not base64", &decoded, &error),
          "bad-base64-rejected", error);

    using json = vision::json;
    const std::string data_url = "data:image/png;base64," + encoded;
    json messages = json::array(
        {json{{"role", "user"},
              {"content", json::array({json{{"type", "image_url"},
                                             {"image_url", {{"url", data_url}}}},
                                        json{{"type", "text"}, {"text", "describe"}}})}}});
    std::vector<vision::Frame> frames;
    check(vision::prepare_messages(messages, &frames, &error) && frames.size() == 1,
          "prepare-chat-image", error);

    json too_many = json::array({json{{"role", "user"}, {"content", json::array()}}});
    for (int i = 0; i < 9; ++i)
        too_many[0]["content"].push_back(
            json{{"type", "input_image"}, {"image_url", data_url}});
    check(!vision::prepare_messages(too_many, &frames, &error) &&
              error.find("at most 8") != std::string::npos,
          "image-count-rejected", error);
    json video = json::array(
        {json{{"role", "user"},
              {"content", json::array(
                              {json{{"type", "input_video"}, {"video_url", data_url}}})}}});
    check(!vision::prepare_messages(video, &frames, &error) &&
              error.find("video input") != std::string::npos,
          "video-rejected", error);
    json file_id = json::array(
        {json{{"role", "user"},
              {"content", json::array(
                              {json{{"type", "input_image"}, {"file_id", "file_123"}}})}}});
    check(!vision::prepare_messages(file_id, &frames, &error) &&
              error.find("file_id") != std::string::npos,
          "file-id-rejected", error);

    qwenox::Tokenizer tokenizer;
    error.clear();
    check(tokenizer.load(argv[2], &error), "tokenizer-load", error);
    vision::Frame placeholder;
    placeholder.grid = {1, 4, 4};
    frames = {placeholder};
    std::vector<int> ids;
    const std::string prompt =
        "before<|vision_start|><|image_pad|><|vision_end|>after";
    error.clear();
    check(vision::encode_prompt(prompt, tokenizer, frames, &ids, &error),
          "placeholder-expand", error);
    check(std::count(ids.begin(), ids.end(), 248056) == placeholder.pad_tokens(),
          "placeholder-token-count");
    frames.push_back(placeholder);
    check(!vision::encode_prompt(prompt, tokenizer, frames, &ids, &error) &&
              error.find("placeholder count") != std::string::npos,
          "placeholder-mismatch-rejected", error);

    // A user-typed pad literal outside the template span is plain text, not
    // a placeholder: it must not survive as id 248056/248057.
    frames = {placeholder};
    const std::string stray =
        "the token <|image_pad|> discussed. "
        "before<|vision_start|><|image_pad|><|vision_end|>after";
    error.clear();
    check(vision::encode_prompt(stray, tokenizer, frames, &ids, &error),
          "stray-pad-accepted", error);
    check(std::count(ids.begin(), ids.end(), 248056) == placeholder.pad_tokens(),
          "stray-pad-token-count");
    // Same exclusion at the tokenizer level (text-only requests use it).
    const std::unordered_set<int> no_pads = {248056, 248057};
    const std::vector<int> text_ids =
        tokenizer.encode("the token <|image_pad|> is text", false, &no_pads);
    check(std::count(text_ids.begin(), text_ids.end(), 248056) == 0,
          "text-encode-excludes-pad");
    const std::vector<int> default_text_ids =
        tokenizer.encode("the token <|image_pad|> is text");
    check(!default_text_ids.empty() &&
              std::count(default_text_ids.begin(), default_text_ids.end(), 248056) == 1,
          "default-encode-keeps-pad");

    std::puts("RESULT PASS");
    return 0;
}
