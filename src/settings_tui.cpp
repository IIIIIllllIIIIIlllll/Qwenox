// qwenox-settings — Linux / Windows 共用的 service.conf 终端设置工具。
// 只编辑、检查和保存配置，不启动引擎或 API；现有 settings_win.exe GUI 保持独立。
//
// 终端层不依赖 ncurses：POSIX 使用 termios，Windows 使用 Console API + VT。
// 配置层复用 service_conf.h，保留未知键、注释、环境变量覆盖和 .bak 备份语义。
// 界面中英双语：TR(中文, English) 就地双语，第 6 页「语言」切换（立即生效并写
// lang 标记）；解析优先级 lang 标记 > UI_LANG > 系统界面语言，与 Windows 设置
// 面板/托盘（settings_panel.inc / tray_win.cpp）共享同一标记。
#include "service_conf.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace {

constexpr const char* kWeightsMarker = "# start_win: weights=";
constexpr const char* kLangMarker = "# start_win: lang=";

// 界面语言：中英双语，与 Windows 设置面板/托盘（settings_panel.inc / tray_win.cpp）
// 共享同一个 service.conf 注释标记 "# start_win: lang=zh|en"。文案处用 TR(中文, English)
// 就地双语。解析优先级：lang 标记 > UI_LANG 环境变量 > 系统界面语言
//（Windows 取 GetUserDefaultUILanguage；POSIX 没有 UI 语言 API，用 locale 变量近似）。
bool g_lang_en = false;
#define TR(zh, en) (g_lang_en ? (en) : (zh))

bool lang_name_is(const char* s, const char* want) {
    return s && std::strlen(s) == 2 &&
           std::tolower(static_cast<unsigned char>(s[0])) == want[0] &&
           std::tolower(static_cast<unsigned char>(s[1])) == want[1];
}

void lang_resolve(const std::string& conf_path) {
    const std::string marker = svcconf::conf_marker(conf_path, kLangMarker);
    if (lang_name_is(marker.c_str(), "en")) { g_lang_en = true; return; }
    if (lang_name_is(marker.c_str(), "zh")) { g_lang_en = false; return; }
    if (const char* env = std::getenv("UI_LANG")) {
        if (lang_name_is(env, "en")) { g_lang_en = true; return; }
        if (lang_name_is(env, "zh")) { g_lang_en = false; return; }
    }
#ifdef _WIN32
    g_lang_en = PRIMARYLANGID(GetUserDefaultUILanguage()) != LANG_CHINESE;
#else
    const char* locale = std::getenv("LANGUAGE");
    if (!locale || !*locale) locale = std::getenv("LC_ALL");
    if (!locale || !*locale) locale = std::getenv("LANG");
    g_lang_en = !(locale && std::strstr(locale, "zh"));
#endif
}

std::string path_text(const fs::path& p) { return p.u8string(); }

#ifdef _WIN32
std::string wide_text(const wchar_t* s, int n = -1) {
    if (!s || n == 0) return "";
    if (n < 0) n = static_cast<int>(wcslen(s));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, s, n, nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(bytes), '\0');
    if (bytes) WideCharToMultiByte(CP_UTF8, 0, s, n, &out[0], bytes, nullptr, nullptr);
    return out;
}
#endif

fs::path executable_path() {
#ifdef _WIN32
    std::wstring buf(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    return n ? fs::path(buf.substr(0, n)) : fs::current_path();
#else
    std::vector<char> buf(4096, 0);
    const ssize_t n = readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    return n > 0 ? fs::path(std::string(buf.data(), static_cast<size_t>(n)))
                 : fs::current_path();
#endif
}

fs::path absolute_path(const fs::path& p) {
    std::error_code ec;
    fs::path out = fs::absolute(p, ec);
    return ec ? p : out.lexically_normal();
}

std::string join_path(const std::string& base, const std::string& leaf) {
    return (fs::path(base) / leaf).generic_string();
}

// ---- UTF-8 与终端显示宽度 -----------------------------------------------------

uint32_t utf8_at(const std::string& s, size_t pos, size_t* len) {
    const unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c < 0x80) {
        *len = 1;
        return c;
    }
    int n = 0;
    uint32_t cp = 0, min_cp = 0;
    if ((c & 0xe0) == 0xc0) n = 2, cp = c & 0x1f, min_cp = 0x80;
    else if ((c & 0xf0) == 0xe0) n = 3, cp = c & 0x0f, min_cp = 0x800;
    else if ((c & 0xf8) == 0xf0) n = 4, cp = c & 0x07, min_cp = 0x10000;
    else {
        *len = 1;
        return 0xfffd;
    }
    if (pos + static_cast<size_t>(n) > s.size()) {
        *len = 1;
        return 0xfffd;
    }
    for (int i = 1; i < n; ++i) {
        const unsigned char d = static_cast<unsigned char>(s[pos + i]);
        if ((d & 0xc0) != 0x80) {
            *len = 1;
            return 0xfffd;
        }
        cp = (cp << 6) | (d & 0x3f);
    }
    if (cp < min_cp || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
        *len = 1;
        return 0xfffd;
    }
    *len = static_cast<size_t>(n);
    return cp;
}

std::vector<size_t> utf8_offsets(const std::string& s) {
    std::vector<size_t> out(1, 0);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t n = 1;
        utf8_at(s, pos, &n);
        pos += n;
        out.push_back(pos);
    }
    return out;
}

int codepoint_width(uint32_t cp) {
    if (cp == 0 || cp < 32 || (cp >= 0x7f && cp < 0xa0)) return 0;
    if ((cp >= 0x0300 && cp <= 0x036f) || (cp >= 0x0483 && cp <= 0x0489) ||
        (cp >= 0x0591 && cp <= 0x05bd) || (cp >= 0x0610 && cp <= 0x061a) ||
        (cp >= 0x064b && cp <= 0x065f) || (cp >= 0x06d6 && cp <= 0x06ed) ||
        (cp >= 0x1ab0 && cp <= 0x1aff) || (cp >= 0x1dc0 && cp <= 0x1dff) ||
        (cp >= 0x200b && cp <= 0x200f) || (cp >= 0x20d0 && cp <= 0x20ff) ||
        (cp >= 0xfe00 && cp <= 0xfe0f) || (cp >= 0xfe20 && cp <= 0xfe2f) ||
        (cp >= 0xe0100 && cp <= 0xe01ef))
        return 0;
    if (cp >= 0x1100 &&
        (cp <= 0x115f || cp == 0x2329 || cp == 0x232a ||
         (cp >= 0x2e80 && cp <= 0xa4cf && cp != 0x303f) ||
         (cp >= 0xac00 && cp <= 0xd7a3) || (cp >= 0xf900 && cp <= 0xfaff) ||
         (cp >= 0xfe10 && cp <= 0xfe19) || (cp >= 0xfe30 && cp <= 0xfe6f) ||
         (cp >= 0xff00 && cp <= 0xff60) || (cp >= 0xffe0 && cp <= 0xffe6) ||
         (cp >= 0x1f300 && cp <= 0x1faff) || (cp >= 0x20000 && cp <= 0x3fffd)))
        return 2;
    return 1;
}

int text_width(const std::string& s, size_t begin = 0, size_t end = std::string::npos) {
    if (end == std::string::npos || end > s.size()) end = s.size();
    int width = 0;
    size_t pos = begin;
    while (pos < end) {
        size_t n = 1;
        const uint32_t cp = utf8_at(s, pos, &n);
        if (pos + n > end) break;
        width += codepoint_width(cp);
        pos += n;
    }
    return width;
}

std::string clip_text(const std::string& s, int width, bool tail = false) {
    if (width <= 0) return "";
    if (text_width(s) <= width) return s;
    if (width == 1) return "…";
    const std::vector<size_t> off = utf8_offsets(s);
    if (!tail) {
        size_t pos = 0;
        int used = 0;
        while (pos < s.size()) {
            size_t n = 1;
            const uint32_t cp = utf8_at(s, pos, &n);
            const int w = codepoint_width(cp);
            if (used + w > width - 1) break;
            used += w;
            pos += n;
        }
        return s.substr(0, pos) + "…";
    }
    size_t first = off.size() - 1;
    int used = 0;
    while (first > 0) {
        const size_t pos = off[first - 1];
        size_t n = 1;
        const uint32_t cp = utf8_at(s, pos, &n);
        const int w = codepoint_width(cp);
        if (used + w > width - 1) break;
        used += w;
        --first;
    }
    return "…" + s.substr(off[first]);
}

std::string pad_text(const std::string& s, int width, bool tail = false) {
    std::string out = clip_text(s, width, tail);
    const int padding = width - text_width(out);
    if (padding > 0) out.append(static_cast<size_t>(padding), ' ');
    return out;
}

std::vector<std::string> wrap_text(const std::string& s, int width) {
    std::vector<std::string> lines;
    if (width <= 0) return lines;
    std::string line;
    int used = 0;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t n = 1;
        const uint32_t cp = utf8_at(s, pos, &n);
        const int w = codepoint_width(cp);
        if (used && used + w > width) {
            lines.push_back(line);
            line.clear();
            used = 0;
        }
        line.append(s, pos, n);
        used += w;
        pos += n;
    }
    lines.push_back(line);
    return lines;
}

// ---- 跨平台终端 ---------------------------------------------------------------

enum class KeyType {
    kNone, kResize, kUp, kDown, kLeft, kRight, kTab, kBackTab, kEnter,
    kEscape, kBackspace, kDelete, kHome, kEnd, kSave, kQuit, kCharacter,
};
struct Key {
    KeyType type = KeyType::kNone;
    std::string text;
};

class Terminal {
public:
    ~Terminal() { close(); }

    bool open(std::string* error) {
#ifdef _WIN32
        in_ = GetStdHandle(STD_INPUT_HANDLE);
        out_ = GetStdHandle(STD_OUTPUT_HANDLE);
        if (in_ == INVALID_HANDLE_VALUE || out_ == INVALID_HANDLE_VALUE ||
            !GetConsoleMode(in_, &old_in_) || !GetConsoleMode(out_, &old_out_)) {
            *error = TR("需要 Windows 控制台（Windows Terminal / PowerShell / CMD；Git Bash 可用 winpty）",
                        "A Windows console is required (Windows Terminal / PowerShell / CMD; use winpty from Git Bash)");
            return false;
        }
        old_out_cp_ = GetConsoleOutputCP();
        old_in_cp_ = GetConsoleCP();
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
        active_ = true;
        DWORD in_mode = (old_in_ & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT |
                                     ENABLE_PROCESSED_INPUT | ENABLE_QUICK_EDIT_MODE)) |
                        ENABLE_WINDOW_INPUT | ENABLE_EXTENDED_FLAGS;
        if (!SetConsoleMode(in_, in_mode) ||
            !SetConsoleMode(out_, old_out_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            *error = TR("无法启用控制台 VT 模式，请使用 Windows 10 或更新版本",
                        "Could not enable console VT mode; Windows 10 or newer is required");
            close();
            return false;
        }
#else
        if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
            *error = TR("交互模式需要终端；脚本中请使用 --check",
                        "Interactive mode needs a terminal; use --check from scripts");
            return false;
        }
        if (tcgetattr(STDIN_FILENO, &old_term_) != 0) {
            *error = TR("无法读取终端状态", "Could not read terminal state");
            return false;
        }
        termios raw = old_term_;
        raw.c_lflag &= static_cast<tcflag_t>(~(ECHO | ICANON | ISIG | IEXTEN));
        raw.c_iflag &= static_cast<tcflag_t>(~(IXON | ICRNL | BRKINT | INPCK | ISTRIP));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) {
            *error = TR("无法切换终端输入模式", "Could not switch terminal input mode");
            return false;
        }
        active_ = true;
#endif
        active_ = true;
        write("\x1b[?1049h\x1b[?25l\x1b[0m");
        size_ = query_size();
        return true;
    }

    void close() {
        if (!active_) return;
        write("\x1b[0m\x1b[?25h\x1b[?1049l");
#ifdef _WIN32
        SetConsoleMode(in_, old_in_);
        SetConsoleMode(out_, old_out_);
        SetConsoleOutputCP(old_out_cp_);
        SetConsoleCP(old_in_cp_);
#else
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &old_term_);
#endif
        active_ = false;
    }

    std::pair<int, int> size() const { return size_; }

    void render(const std::string& frame) { write(frame); }

    Key read_key() {
#ifdef _WIN32
        for (;;) {
            if (WaitForSingleObject(in_, 250) == WAIT_TIMEOUT) return resize_key();
            INPUT_RECORD rec{};
            DWORD count = 0;
            if (!ReadConsoleInputW(in_, &rec, 1, &count) || !count) continue;
            if (rec.EventType == WINDOW_BUFFER_SIZE_EVENT) {
                size_ = query_size();
                return {KeyType::kResize, ""};
            }
            if (rec.EventType != KEY_EVENT || !rec.Event.KeyEvent.bKeyDown) continue;
            const KEY_EVENT_RECORD& k = rec.Event.KeyEvent;
            const DWORD state = k.dwControlKeyState;
            const bool ctrl = (state & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) != 0;
            if (ctrl && (k.wVirtualKeyCode == 'S' || k.wVirtualKeyCode == 'C'))
                return {k.wVirtualKeyCode == 'S' ? KeyType::kSave : KeyType::kQuit, ""};
            if (ctrl && k.wVirtualKeyCode == 'U')
                return {KeyType::kCharacter, std::string(1, 21)};
            switch (k.wVirtualKeyCode) {
                case VK_UP: return {KeyType::kUp, ""};
                case VK_DOWN: return {KeyType::kDown, ""};
                case VK_LEFT: return {KeyType::kLeft, ""};
                case VK_RIGHT: return {KeyType::kRight, ""};
                case VK_TAB:
                    return {(state & SHIFT_PRESSED) ? KeyType::kBackTab : KeyType::kTab, ""};
                case VK_RETURN: return {KeyType::kEnter, ""};
                case VK_ESCAPE: return {KeyType::kEscape, ""};
                case VK_BACK: return {KeyType::kBackspace, ""};
                case VK_DELETE: return {KeyType::kDelete, ""};
                case VK_HOME: return {KeyType::kHome, ""};
                case VK_END: return {KeyType::kEnd, ""};
                case VK_F2: return {KeyType::kSave, ""};
                default: break;
            }
            wchar_t wc = k.uChar.UnicodeChar;
            if (!wc || (wc < 32 && wc != 9)) continue;
            if (pending_high_) {
                if (wc >= 0xdc00 && wc <= 0xdfff) {
                    const wchar_t pair[] = {pending_high_, wc};
                    pending_high_ = 0;
                    return {KeyType::kCharacter, wide_text(pair, 2)};
                }
                pending_high_ = 0;
            }
            if (wc >= 0xd800 && wc <= 0xdbff) {
                pending_high_ = wc;
                continue;
            }
            return {KeyType::kCharacter, wide_text(&wc, 1)};
        }
#else
        unsigned char c = 0;
        if (!read_byte(&c, 250)) return resize_key();
        if (c == 3) return {KeyType::kQuit, ""};
        if (c == 19) return {KeyType::kSave, ""};
        if (c == 9) return {KeyType::kTab, ""};
        if (c == 10 || c == 13) return {KeyType::kEnter, ""};
        if (c == 8 || c == 127) return {KeyType::kBackspace, ""};
        if (c == 27) return read_escape();
        if (c == 21) return {KeyType::kCharacter, std::string(1, 21)};
        if (c < 32) return {KeyType::kNone, ""};
        std::string text(1, static_cast<char>(c));
        int need = 0;
        if ((c & 0xe0) == 0xc0) need = 1;
        else if ((c & 0xf0) == 0xe0) need = 2;
        else if ((c & 0xf8) == 0xf0) need = 3;
        for (int i = 0; i < need; ++i) {
            unsigned char d = 0;
            if (!read_byte(&d, 30)) break;
            text.push_back(static_cast<char>(d));
        }
        return {KeyType::kCharacter, text};
#endif
    }

private:
    void write(const std::string& s) {
#ifdef _WIN32
        if (out_ != INVALID_HANDLE_VALUE) {
            DWORD written = 0;
            WriteFile(out_, s.data(), static_cast<DWORD>(s.size()), &written, nullptr);
        }
#else
        size_t pos = 0;
        while (pos < s.size()) {
            const ssize_t n = ::write(STDOUT_FILENO, s.data() + pos, s.size() - pos);
            if (n <= 0) {
                if (errno == EINTR) continue;
                break;
            }
            pos += static_cast<size_t>(n);
        }
#endif
    }

    std::pair<int, int> query_size() const {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO info{};
        if (GetConsoleScreenBufferInfo(out_, &info))
            return {info.srWindow.Right - info.srWindow.Left + 1,
                    info.srWindow.Bottom - info.srWindow.Top + 1};
#else
        winsize ws{};
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col && ws.ws_row)
            return {ws.ws_col, ws.ws_row};
#endif
        return {80, 24};
    }

    Key resize_key() {
        const auto now = query_size();
        if (now != size_) {
            size_ = now;
            return {KeyType::kResize, ""};
        }
        return {};
    }

#ifndef _WIN32
    bool read_byte(unsigned char* c, int timeout_ms) {
        for (;;) {
            pollfd p{STDIN_FILENO, POLLIN, 0};
            const int ready = poll(&p, 1, timeout_ms);
            if (ready == 0) return false;
            if (ready < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            const ssize_t n = ::read(STDIN_FILENO, c, 1);
            if (n == 1) return true;
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
    }

    Key read_escape() {
        unsigned char c = 0;
        if (!read_byte(&c, 25)) return {KeyType::kEscape, ""};
        if (c != '[' && c != 'O') return {KeyType::kEscape, ""};
        std::string seq;
        seq.push_back(static_cast<char>(c));
        for (int i = 0; i < 15; ++i) {
            if (!read_byte(&c, 25)) return {KeyType::kEscape, ""};
            seq.push_back(static_cast<char>(c));
            if (c >= '@' && c <= '~') break;
        }
        if (seq == "[A" || seq == "OA") return {KeyType::kUp, ""};
        if (seq == "[B" || seq == "OB") return {KeyType::kDown, ""};
        if (seq == "[C" || seq == "OC") return {KeyType::kRight, ""};
        if (seq == "[D" || seq == "OD") return {KeyType::kLeft, ""};
        if (seq == "[H" || seq == "OH" || seq == "[1~" || seq == "[7~")
            return {KeyType::kHome, ""};
        if (seq == "[F" || seq == "OF" || seq == "[4~" || seq == "[8~")
            return {KeyType::kEnd, ""};
        if (seq == "[3~") return {KeyType::kDelete, ""};
        if (seq == "[Z") return {KeyType::kBackTab, ""};
        if (seq == "[12~" || seq == "OQ") return {KeyType::kSave, ""};
        return {KeyType::kNone, ""};
    }
#endif

    bool active_ = false;
    std::pair<int, int> size_{80, 24};
#ifdef _WIN32
    HANDLE in_ = INVALID_HANDLE_VALUE;
    HANDLE out_ = INVALID_HANDLE_VALUE;
    DWORD old_in_ = 0, old_out_ = 0;
    UINT old_in_cp_ = 0, old_out_cp_ = 0;
    wchar_t pending_high_ = 0;
#else
    termios old_term_{};
#endif
};

// ---- 配置模型 -----------------------------------------------------------------

using Values = std::map<std::string, std::string>;

enum class FieldKind { kText, kBool, kChoice };
enum class FieldGroup { kAlways, kHgn, kV1, kV2, kGguf };
struct Option {
    const char* value;
    const char* label;
    const char* label_en = nullptr;
    // 当前语言的选项名；语言中立的选项（如 "hgn v1"）不写英文，回退到 label。
    const char* text() const { return g_lang_en && label_en ? label_en : label; }
};
struct Field {
    int page;
    const char* key;
    const char* label;
    const char* hint;
    const char* label_en;
    const char* hint_en;
    FieldKind kind;
    FieldGroup group;
    std::vector<Option> options;
    // 表是 static 的，语言切换后不能重建；访问器按 g_lang_en 现场取值。
    const char* label_text() const { return g_lang_en && label_en ? label_en : label; }
    const char* hint_text() const { return g_lang_en && hint_en ? hint_en : hint; }
};

const std::vector<Field>& field_specs() {
    static const std::vector<Field> fields = {
        {0, "#WEIGHTS", "权重格式", "V1：内嵌 n-gram；V2：独立 n-gram；GGUF：使用 GGUF 主分片",
         "Weight format", "V1: embedded n-gram; V2: separate n-gram; GGUF: use the GGUF main shard",
         FieldKind::kChoice, FieldGroup::kAlways,
         {{"v1", "hgn v1"}, {"v2", "hgn v2"}, {"gguf", "GGUF"}}},
        {0, "MODEL_FILE", "主模型", "hgn 主权重文件",
         "Main model", "Main hgn weights file", FieldKind::kText, FieldGroup::kHgn, {}},
        {0, "NGRAM_FILE", "n-gram 表", "V2 必须是与主模型不同的独立文件",
         "n-gram table", "V2 requires a separate file, different from the main model",
         FieldKind::kText, FieldGroup::kV2, {}},
        {0, "OVERLAY_FILE", "overlay", "V1 可选；留空表示不使用",
         "Overlay", "Optional for V1; leave empty to disable", FieldKind::kText, FieldGroup::kV1, {}},
        {0, "MTP_FILE", "MTP 草稿", "留空表示不使用独立 MTP 草稿文件",
         "MTP draft", "Leave empty to not use a separate MTP draft file",
         FieldKind::kText, FieldGroup::kHgn, {}},
        {0, "VISION_FILE", "视觉塔", "留空表示纯文本服务",
         "Vision tower", "Leave empty for text-only serving", FieldKind::kText, FieldGroup::kHgn, {}},
        {0, "GGUF_FILE", "GGUF 主分片", "多分片模型填写 -00001-of-XXXXX.gguf",
         "GGUF main shard", "For multi-shard models use the -00001-of-XXXXX.gguf file",
         FieldKind::kText, FieldGroup::kGguf, {}},
        {0, "GGUF_MTP_FILE", "GGUF MTP", "留空表示不使用 MTP sidecar",
         "GGUF MTP", "Leave empty to not use an MTP sidecar", FieldKind::kText, FieldGroup::kGguf, {}},
        {0, "GGUF_VISION_FILE", "GGUF 视觉塔", "留空表示纯文本服务",
         "GGUF vision tower", "Leave empty for text-only serving",
         FieldKind::kText, FieldGroup::kGguf, {}},
        {0, "GGUF_I4R_PATCH", "i4r 补丁 [Linux]", "可选 .hgn 专家补丁；留空表示不使用",
         "i4r patch [Linux]", "Optional .hgn expert patch; leave empty to disable",
         FieldKind::kText, FieldGroup::kGguf, {}},
        {0, "TOKENIZER_DIR", "tokenizer 目录", "目录中必须有 tokenizer.json",
         "Tokenizer dir", "The directory must contain tokenizer.json",
         FieldKind::kText, FieldGroup::kAlways, {}},

        {1, "MAX_CONTEXT", "单请求上下文", "输入与输出合计 token 上限",
         "Context/request", "Max tokens (input + output) per request",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {1, "PARALLEL", "并发路数", "1–8；大于 1 时必须启用分页 KV",
         "Parallel slots", "1-8; values above 1 require paged KV", FieldKind::kText, FieldGroup::kAlways, {}},
        {1, "KV_POOL_TOKENS", "并发 KV 总池", "0 表示跟随单请求上下文；小于它时 Linux 也按单请求上限计算",
         "KV pool tokens", "0 = follow the per-request context; smaller values still count as the per-request limit on Linux",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {1, "KV_PAGED", "分页 KV", "并发模式必须开启",
         "Paged KV", "Required for parallel mode", FieldKind::kBool, FieldGroup::kAlways, {}},
        {1, "PREFILL_CHUNK", "prefill 分段", "0：Linux 默认 16384，Windows 默认 6144",
         "Prefill chunk", "0: Linux default 16384, Windows default 6144",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {1, "CONC_PREFILL_CHUNK", "并发分段 [Linux]", "0 表示不另设；实际值不会超过 PREFILL_CHUNK",
         "Concurrent chunk [Linux]", "0 = not set separately; the effective value never exceeds PREFILL_CHUNK",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {1, "QSA_INDEXER_BUDGET", "QSA 检索预算", "0 使用默认 2048；4096 会增加 prefill 开销",
         "QSA indexer budget", "0 = default 2048; 4096 adds prefill cost",
         FieldKind::kChoice, FieldGroup::kAlways,
         {{"0", "0（默认 2048）", "0 (default 2048)"}, {"4096", "4096"}}},
        {1, "ROPE_FACTOR", "YaRN 倍数", "1 为关闭；超过原生上下文时按扩展倍数设置",
         "YaRN factor", "1 = off; set to the extension factor beyond the native context",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {1, "MOE_A4", "prefill 4-bit", "仅 hgn v2 / i4r 补丁路径有效；生成 decode 不经过此路径",
         "Prefill 4-bit", "Only affects hgn v2 / i4r patch paths; decode never uses it",
         FieldKind::kChoice, FieldGroup::kAlways,
         {{"0", "0（关闭）", "0 (off)"}, {"1", "1（gate/up）", "1 (gate/up)"},
          {"2", "2（down）", "2 (down)"}, {"3", "3（两者）", "3 (both)"}}},
        {1, "MTP_GAMMA", "MTP gamma", "0 自动；1–8 固定草稿长度",
         "MTP gamma", "0 = auto; 1-8 fixes the draft length", FieldKind::kText, FieldGroup::kAlways, {}},
        {1, "MAX_IMAGES", "单请求图片上限", "1–256，按整个 messages 历史累计",
         "Max images/request", "1-256, counted across the whole messages history",
         FieldKind::kText, FieldGroup::kAlways, {}},

        {2, "KVSNAP_MAX_GB", "SSD 快照上限 GiB", "0 表示关闭；超过上限按 LRU 逐出",
         "SSD snapshot cap GiB", "0 = off; entries are evicted LRU past the cap",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {2, "KVSNAP_DIR", "SSD 快照目录", "留空使用 data/kvsnap；否则使用其 qwenox-kvsnap 子目录",
         "SSD snapshot dir", "Empty = data/kvsnap; otherwise its qwenox-kvsnap subdirectory is used",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {2, "RCKPT_MAX", "RAM 检查点 [Linux]", "0 表示关闭；每个约 118 MiB",
         "RAM checkpoints [Linux]", "0 = off; about 118 MiB each", FieldKind::kText, FieldGroup::kAlways, {}},
        {2, "PLE_URING", "PLE io_uring [Linux]", "使用 io_uring 批量读取磁盘分页 n-gram 表",
         "PLE io_uring [Linux]", "Use io_uring to batch-read the disk-paged n-gram table",
         FieldKind::kBool, FieldGroup::kAlways, {}},
        {2, "MEMORY_CAP_GB", "进程内存 cap [Linux]", "启动器传给 run_capped.sh 的内存上限",
         "Memory cap GB [Linux]", "Memory limit the launcher passes to run_capped.sh",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {2, "MIN_AVAILABLE_GB", "最低可用内存 [Linux]", "可用内存低于此值时启动器拒绝启动",
         "Min free memory [Linux]", "The launcher refuses to start below this much free memory",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {2, "START_TIMEOUT", "启动超时秒 [Linux]", "等待引擎进入监听状态的总时间",
         "Start timeout s [Linux]", "Total time to wait for the engine to start listening",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {2, "STALL_TIMEOUT", "加载停滞秒 [Linux]", "加载进度停滞超过此时间则拒绝继续等待",
         "Load stall s [Linux]", "Give up waiting if load progress stalls for this long",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {2, "AFFINITY", "CPU 亲和性 [Windows]", "空、lower-half、upper-half 或非零整数 mask；Linux 请用 taskset",
         "CPU affinity [Windows]", "Empty, lower-half, upper-half, or a nonzero integer mask; on Linux use taskset",
         FieldKind::kText, FieldGroup::kAlways, {}},

        {3, "ENGINE_HOST", "引擎监听地址", "内部行协议，推荐 127.0.0.1",
         "Engine listen addr", "Internal line protocol; 127.0.0.1 recommended",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {3, "ENGINE_PORT", "引擎端口", "1–65535，且不能与 API 端口相同",
         "Engine port", "1-65535; must differ from the API port", FieldKind::kText, FieldGroup::kAlways, {}},
        {3, "API_HOST", "API 监听地址", "127.0.0.1 仅本机；0.0.0.0 对外开放",
         "API listen addr", "127.0.0.1 = local only; 0.0.0.0 = exposed to the LAN",
         FieldKind::kText, FieldGroup::kAlways, {}},
        {3, "API_PORT", "API 端口", "浏览器和客户端连接此端口",
         "API port", "Browsers and clients connect to this port", FieldKind::kText, FieldGroup::kAlways, {}},
        {3, "VIZ", "运行时可视化", "可访问 API 的人能看到生成内容；decode 略有开销",
         "Runtime viz", "Anyone with API access can see generated content; slight decode cost",
         FieldKind::kBool, FieldGroup::kAlways, {}},
        {3, "REQLOG", "完整请求日志", "完整保存提示词和回复，注意隐私与磁盘占用",
         "Full request log", "Stores full prompts and replies; mind privacy and disk usage",
         FieldKind::kBool, FieldGroup::kAlways, {}},
        {3, "REQLOG_DIR", "请求日志目录", "留空保存为 logs/reqlog",
         "Request log dir", "Empty = logs/reqlog", FieldKind::kText, FieldGroup::kAlways, {}},
        {3, "REQLOG_LEVEL", "请求日志级别", "1–10",
         "Request log level", "1-10", FieldKind::kText, FieldGroup::kAlways, {}},
    };
    return fields;
}

bool field_visible(const Field& f, const Values& values) {
    const auto it = values.find("#WEIGHTS");
    const std::string ver = it == values.end() ? "v2" : it->second;
    switch (f.group) {
        case FieldGroup::kAlways: return true;
        case FieldGroup::kHgn: return ver != "gguf";
        case FieldGroup::kV1: return ver == "v1";
        case FieldGroup::kV2: return ver == "v2";
        case FieldGroup::kGguf: return ver == "gguf";
    }
    return true;
}

struct Model {
    fs::path root;
    fs::path conf_path;
    svcconf::Conf conf;
    Values values;

    std::string get(const std::string& key, const std::string& fallback = "") const {
        const auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    }

    void load() {
        conf.clear();
        values.clear();
        svcconf::load_conf(path_text(conf_path), &conf);
        const std::string model_dir = svcconf::cfg(conf, "MODEL_DIR", "./models");
        const std::string model = svcconf::cfg(
            conf, "MODEL_FILE", join_path(model_dir, "qwen38-flash-next-v2.hgn"));
        const std::string ngram = svcconf::cfg(conf, "NGRAM_FILE", model);
        const std::string marker = svcconf::conf_marker(path_text(conf_path), kWeightsMarker);
        std::string ver = marker;
        if (ver != "v1" && ver != "v2" && ver != "gguf") ver = ngram == model ? "v1" : "v2";
        values["#WEIGHTS"] = ver;
        values["MODEL_FILE"] = model;
        values["NGRAM_FILE"] = ngram;
        values["OVERLAY_FILE"] = svcconf::cfg_optional(conf, "OVERLAY_FILE", "");
        values["MTP_FILE"] = svcconf::cfg_optional(
            conf, "MTP_FILE", join_path(model_dir, "qwen38-flash-next-mtp.hgn"));
        values["VISION_FILE"] = svcconf::cfg_optional(
            conf, "VISION_FILE", join_path(model_dir, "qwen38-flash-next-vision.hgn"));
        values["GGUF_FILE"] = svcconf::cfg(
            conf, "GGUF_FILE",
            join_path(model_dir, "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"));
        values["GGUF_MTP_FILE"] = svcconf::cfg_optional(
            conf, "GGUF_MTP_FILE", join_path(model_dir, "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"));
        values["GGUF_VISION_FILE"] = svcconf::cfg_optional(
            conf, "GGUF_VISION_FILE", join_path(model_dir, "mmproj-BF16.gguf"));
        values["GGUF_I4R_PATCH"] = svcconf::cfg_optional(conf, "GGUF_I4R_PATCH", "");
        values["TOKENIZER_DIR"] = svcconf::cfg(conf, "TOKENIZER_DIR", join_path(model_dir, "tokenizer"));

        values["MAX_CONTEXT"] = svcconf::cfg(conf, "MAX_CONTEXT", "262144");
        values["PARALLEL"] = svcconf::cfg(conf, "PARALLEL", "1");
        values["KV_POOL_TOKENS"] = svcconf::cfg(conf, "KV_POOL_TOKENS", "0");
        values["KV_PAGED"] = svcconf::cfg(conf, "KV_PAGED", "1");
        values["PREFILL_CHUNK"] = svcconf::cfg(conf, "PREFILL_CHUNK", "0");
        values["CONC_PREFILL_CHUNK"] = svcconf::cfg(conf, "CONC_PREFILL_CHUNK", "8192");
        values["QSA_INDEXER_BUDGET"] = svcconf::cfg(conf, "QSA_INDEXER_BUDGET", "0");
        values["ROPE_FACTOR"] = svcconf::cfg(conf, "ROPE_FACTOR", "1");
        values["MOE_A4"] = svcconf::cfg(conf, "MOE_A4", "0");
        values["MTP_GAMMA"] = svcconf::cfg(conf, "MTP_GAMMA", "0");
        values["MAX_IMAGES"] = svcconf::cfg(conf, "MAX_IMAGES", "8");

        values["KVSNAP_MAX_GB"] = svcconf::cfg(conf, "KVSNAP_MAX_GB", "20");
        values["KVSNAP_DIR"] = svcconf::cfg_optional(conf, "KVSNAP_DIR", "");
        values["RCKPT_MAX"] = svcconf::cfg(conf, "RCKPT_MAX", "8");
        values["PLE_URING"] = svcconf::cfg(conf, "PLE_URING", "1");
        values["MEMORY_CAP_GB"] = svcconf::cfg(conf, "MEMORY_CAP_GB", "86");
        values["MIN_AVAILABLE_GB"] = svcconf::cfg(conf, "MIN_AVAILABLE_GB", "100");
        values["START_TIMEOUT"] = svcconf::cfg(conf, "START_TIMEOUT", "360");
        values["STALL_TIMEOUT"] = svcconf::cfg(conf, "STALL_TIMEOUT", "60");
        values["AFFINITY"] = svcconf::cfg(conf, "AFFINITY", "");

        values["ENGINE_HOST"] = svcconf::cfg(conf, "ENGINE_HOST", "127.0.0.1");
        values["ENGINE_PORT"] = svcconf::cfg(conf, "ENGINE_PORT", "8730");
        values["API_HOST"] = svcconf::cfg(conf, "API_HOST", "0.0.0.0");
        values["API_PORT"] = svcconf::cfg(conf, "API_PORT", "8731");
        values["VIZ"] = svcconf::cfg(conf, "VIZ", "0");
        values["REQLOG"] = svcconf::cfg(conf, "REQLOG", "0");
        values["REQLOG_DIR"] = svcconf::cfg(conf, "REQLOG_DIR", "logs/reqlog");
        values["REQLOG_LEVEL"] = svcconf::cfg(conf, "REQLOG_LEVEL", "6");

        // 不在界面中暴露，但参与与启动器一致的联动校验；保存时保持原行不动。
        values["ROPE_ORIGINAL_CTX"] = svcconf::cfg(conf, "ROPE_ORIGINAL_CTX", "262144");
        values["ROPE_BETA_FAST"] = svcconf::cfg(conf, "ROPE_BETA_FAST", "32");
        values["ROPE_BETA_SLOW"] = svcconf::cfg(conf, "ROPE_BETA_SLOW", "1");
        values["ROPE_ATTN_SCALE"] = svcconf::cfg(conf, "ROPE_ATTN_SCALE", "0");
    }

    Values conf_values() const {
        Values out;
        const std::string ver = get("#WEIGHTS", "v2");
        out["TOKENIZER_DIR"] = get("TOKENIZER_DIR");
        if (ver == "gguf") {
            for (const char* key : {"GGUF_FILE", "GGUF_MTP_FILE", "GGUF_VISION_FILE", "GGUF_I4R_PATCH"})
                out[key] = get(key);
        } else {
            out["MODEL_FILE"] = get("MODEL_FILE");
            out["NGRAM_FILE"] = ver == "v1" ? "$MODEL_FILE" : get("NGRAM_FILE");
            out["OVERLAY_FILE"] = ver == "v2" ? "" : get("OVERLAY_FILE");
            out["MTP_FILE"] = get("MTP_FILE");
            out["VISION_FILE"] = get("VISION_FILE");
        }
        for (const Field& f : field_specs()) {
            const std::string key = f.key;
            if (key == "#WEIGHTS" || f.page == 0 || key == "GGUF_I4R_PATCH") continue;
            out[key] = get(key);
        }
        while (!out["TOKENIZER_DIR"].empty() &&
               (out["TOKENIZER_DIR"].back() == '/' || out["TOKENIZER_DIR"].back() == '\\'))
            out["TOKENIZER_DIR"].pop_back();
        while (!out["KVSNAP_DIR"].empty() &&
               (out["KVSNAP_DIR"].back() == '/' || out["KVSNAP_DIR"].back() == '\\'))
            out["KVSNAP_DIR"].pop_back();
        if (out["REQLOG_DIR"].empty()) out["REQLOG_DIR"] = "logs/reqlog";
        if (out["REQLOG_LEVEL"].empty()) out["REQLOG_LEVEL"] = "6";
        return out;
    }

    fs::path resolve(const std::string& value) const {
        fs::path p(value);
        return p.is_relative() ? root / p : p;
    }
};

// ---- 校验 ---------------------------------------------------------------------

enum class Level { kHint, kPass, kWarn, kFail };
struct Check {
    Level level;
    std::string text;
};

const char* level_text(Level level) {
    // 英文 token 与 Windows 面板保持一致（settings_panel.inc）。
    switch (level) {
        case Level::kFail: return TR("失败", "FAIL");
        case Level::kWarn: return TR("警告", "WARN");
        case Level::kPass: return TR("通过", "PASS");
        default: return TR("提示", "HINT");
    }
}

bool strict_int(const std::string& s, long* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const long n = std::strtol(s.c_str(), &end, 10);
    if (errno || !end || *end || end == s.c_str()) return false;
    *out = n;
    return true;
}

bool strict_double(const std::string& s, double* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const double n = std::strtod(s.c_str(), &end);
    if (errno || !end || *end || end == s.c_str() || !std::isfinite(n)) return false;
    *out = n;
    return true;
}

[[maybe_unused]] std::string lower_text(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

enum class PathKind { kNone, kFile, kDirectory };

PathKind path_kind(const Field& f) {
    const std::string key = f.key;
    if (key == "TOKENIZER_DIR" || key == "KVSNAP_DIR" || key == "REQLOG_DIR")
        return PathKind::kDirectory;
    if (f.page == 0 && key != "#WEIGHTS") return PathKind::kFile;
    return PathKind::kNone;
}

struct PathCandidate {
    std::string name;
    bool directory = false;
};
struct CompletionState {
    bool active = false;
    std::string head, tail, prefix;
    std::vector<PathCandidate> candidates;
    int selected = -1;
    int scroll = 0;
};

uint32_t folded_name_cp(uint32_t cp) {
#ifdef _WIN32
    if (cp >= 'A' && cp <= 'Z') return cp + ('a' - 'A');
#endif
    return cp;
}

bool candidate_prefix(const std::string& name, const std::string& prefix) {
#ifdef _WIN32
    const std::string folded_name = lower_text(name);
    const std::string folded_prefix = lower_text(prefix);
    return folded_prefix.size() <= folded_name.size() &&
           folded_name.compare(0, folded_prefix.size(), folded_prefix) == 0;
#else
    return prefix.size() <= name.size() && name.compare(0, prefix.size(), prefix) == 0;
#endif
}

size_t common_name_prefix(const std::string& a, const std::string& b) {
    size_t pa = 0, pb = 0;
    while (pa < a.size() && pb < b.size()) {
        size_t na = 1, nb = 1;
        const uint32_t ca = folded_name_cp(utf8_at(a, pa, &na));
        const uint32_t cb = folded_name_cp(utf8_at(b, pb, &nb));
        if (ca != cb || na != nb) break;
        pa += na;
        pb += nb;
    }
    return pa;
}

bool ipv4_address(const std::string& s) {
    size_t pos = 0;
    for (int part = 0; part < 4; ++part) {
        const size_t dot = s.find('.', pos);
        if ((part < 3 && dot == std::string::npos) || (part == 3 && dot != std::string::npos))
            return false;
        const std::string item = s.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos);
        if (item.empty() || item.size() > 3 || (item.size() > 1 && item[0] == '0')) return false;
        for (char c : item)
            if (c < '0' || c > '9') return false;
        long n = 0;
        if (!strict_int(item, &n) || n > 255) return false;
        pos = dot == std::string::npos ? s.size() : dot + 1;
    }
    return true;
}

std::vector<Check> validate(const Model& model) {
    std::vector<Check> out;
    auto add = [&](Level level, const std::string& text) { out.push_back({level, text}); };
    auto fail = [&](const std::string& text) { add(Level::kFail, text); };
    auto warn = [&](const std::string& text) { add(Level::kWarn, text); };
    auto hint = [&](const std::string& text) { add(Level::kHint, text); };
    const Values& v = model.values;

    auto check_file = [&](const char* label, const std::string& value, bool required,
                          bool warn_missing, const char* magic) {
        if (value.empty()) {
            if (required) fail(std::string(label) + TR("：不能为空", ": must not be empty"));
            else hint(std::string(label) + TR("：未配置，将禁用对应功能", ": not set; the feature stays disabled"));
            return;
        }
        std::error_code ec;
        const fs::path p = model.resolve(value);
        if (!fs::is_regular_file(p, ec)) {
            if (warn_missing) warn(std::string(label) + TR("：文件不存在，启动器将跳过：", ": file not found, the launcher will skip it: ") + value);
            else fail(std::string(label) + TR("：文件不存在：", ": file not found: ") + value);
            return;
        }
        if (magic) {
            FILE* f = std::fopen(path_text(p).c_str(), "rb");
            char got[4] = {};
            const size_t n = f ? std::fread(got, 1, 4, f) : 0;
            if (f) std::fclose(f);
            if (n != 4 || std::memcmp(got, magic, 4) != 0) {
                fail(std::string(label) + TR("：文件 magic 不是 ", ": file magic is not ") + magic + TR("：", ": ") + value);
                return;
            }
        }
        add(Level::kPass, std::string(label) + TR("：", ": ") + value);
    };

    const std::string ver = model.get("#WEIGHTS", "v2");
    if (ver == "gguf") {
        check_file(TR("GGUF 主分片", "GGUF main shard"), model.get("GGUF_FILE"), true, false, "GGUF");
        check_file(TR("GGUF MTP", "GGUF MTP"), model.get("GGUF_MTP_FILE"), false, false, "GGUF");
        check_file(TR("GGUF 视觉塔", "GGUF vision tower"), model.get("GGUF_VISION_FILE"), false, false, "GGUF");
        check_file(TR("i4r 补丁", "i4r patch"), model.get("GGUF_I4R_PATCH"), false, false, "HGN1");
        if (model.get("GGUF_FILE").find("-00001-of-") != std::string::npos)
            hint(TR("GGUF 多分片：这里检查第 1 分片，其余分片须与它位于同一目录",
                    "GGUF multi-shard: only shard 1 is checked; the other shards must sit in the same directory"));
    } else {
        check_file(TR("主模型", "Main model"), model.get("MODEL_FILE"), true, false, "HGN1");
        if (ver == "v2") {
            const std::string model_file = model.get("MODEL_FILE");
            const std::string ngram = model.get("NGRAM_FILE");
            if (ngram.empty()) {
                fail(TR("n-gram 表：V2 必须填写独立文件", "n-gram table: V2 requires a separate file"));
            } else {
                std::error_code ec1, ec2;
                const fs::path a = model.resolve(model_file), b = model.resolve(ngram);
                const bool same = fs::equivalent(a, b, ec1) ||
                                  (!ec1 && !ec2 && absolute_path(a) == absolute_path(b));
                if (same || a.lexically_normal() == b.lexically_normal())
                    fail(TR("n-gram 表：V2 必须与主模型使用不同文件",
                            "n-gram table: V2 must use a file different from the main model"));
                else
                    check_file(TR("n-gram 表", "n-gram table"), ngram, true, false, "HGN1");
            }
        } else {
            hint(TR("n-gram 表：V1 保存为 $MODEL_FILE，与主模型同文件",
                    "n-gram table: V1 saves it as $MODEL_FILE, the same file as the main model"));
            check_file("overlay", model.get("OVERLAY_FILE"), false, true, "HGN1");
        }
        check_file(TR("MTP 草稿", "MTP draft"), model.get("MTP_FILE"), false, false, "HGN1");
        check_file(TR("视觉塔", "Vision tower"), model.get("VISION_FILE"), false, false, "HGN1");
    }
    const std::string tok = model.get("TOKENIZER_DIR");
    std::error_code tok_ec;
    if (!fs::is_regular_file(model.resolve(join_path(tok, "tokenizer.json")), tok_ec))
        fail(TR("tokenizer：找不到 ", "tokenizer: cannot find ") + join_path(tok, "tokenizer.json"));
    else
        add(Level::kPass, TR("tokenizer：", "tokenizer: ") + join_path(tok, "tokenizer.json"));

    std::map<std::string, long> nums;
    auto need_int = [&](const char* key, long lo, long hi) {
        long n = 0;
        const std::string value = model.get(key);
        if (!strict_int(value, &n) || n < lo || n > hi) {
            fail(std::string(key) + TR(" 必须是 ", " must be an integer in [") + std::to_string(lo) +
                 TR("–", "-") + std::to_string(hi) + TR(" 的整数（当前为 \"", "] (currently \"") +
                 value + TR("\"）", "\")"));
            return false;
        }
        nums[key] = n;
        return true;
    };
    need_int("MAX_CONTEXT", 1024, 1 << 20);
    need_int("PARALLEL", 1, 8);
    need_int("KV_POOL_TOKENS", 0, 1 << 24);
    need_int("KV_PAGED", 0, 1);
    need_int("PREFILL_CHUNK", 0, 1 << 20);
    need_int("CONC_PREFILL_CHUNK", 0, 1 << 20);
    need_int("MAX_IMAGES", 1, 256);
    need_int("MTP_GAMMA", 0, 8);
    need_int("KVSNAP_MAX_GB", 0, 1 << 16);
    need_int("RCKPT_MAX", 0, 99999999);
    need_int("PLE_URING", 0, 1);
    need_int("MEMORY_CAP_GB", 1, 99999999);
    need_int("MIN_AVAILABLE_GB", 1, 99999999);
    need_int("START_TIMEOUT", 1, 99999999);
    need_int("STALL_TIMEOUT", 1, 99999999);
    need_int("ROPE_ORIGINAL_CTX", 1, 1 << 30);
    if (model.get("QSA_INDEXER_BUDGET") != "0" && model.get("QSA_INDEXER_BUDGET") != "4096")
        fail(TR("QSA_INDEXER_BUDGET 只允许 0（默认 2048）或 4096",
                "QSA_INDEXER_BUDGET only accepts 0 (default 2048) or 4096"));
    if (model.get("MOE_A4") < "0" || model.get("MOE_A4") > "3" || model.get("MOE_A4").size() != 1)
        fail(TR("MOE_A4 只允许 0、1、2、3", "MOE_A4 only accepts 0, 1, 2 or 3"));

    double rope = 1, rope_fast = 32, rope_slow = 1, rope_attn = 0;
    if (!strict_double(model.get("ROPE_FACTOR"), &rope) || rope < 1)
        fail(TR("ROPE_FACTOR 必须是大于等于 1 的有限小数", "ROPE_FACTOR must be a finite number >= 1"));
    if (!strict_double(model.get("ROPE_BETA_FAST"), &rope_fast) || rope_fast <= 0)
        fail(TR("ROPE_BETA_FAST 必须大于 0", "ROPE_BETA_FAST must be > 0"));
    if (!strict_double(model.get("ROPE_BETA_SLOW"), &rope_slow) || rope_slow <= 0)
        fail(TR("ROPE_BETA_SLOW 必须大于 0", "ROPE_BETA_SLOW must be > 0"));
    if (rope_fast < rope_slow) fail(TR("ROPE_BETA_FAST 必须大于等于 ROPE_BETA_SLOW",
                                       "ROPE_BETA_FAST must be >= ROPE_BETA_SLOW"));
    if (!strict_double(model.get("ROPE_ATTN_SCALE"), &rope_attn) || rope_attn < 0)
        fail(TR("ROPE_ATTN_SCALE 必须是非负有限小数", "ROPE_ATTN_SCALE must be a finite number >= 0"));

    const auto num = [&](const char* key, long fallback) {
        const auto it = nums.find(key);
        return it == nums.end() ? fallback : it->second;
    };
    const long max_context = num("MAX_CONTEXT", 262144);
    const long parallel = num("PARALLEL", 1);
    const long kv_paged = num("KV_PAGED", 1);
    const long kv_pool = num("KV_POOL_TOKENS", 0);
    const long rope_original = num("ROPE_ORIGINAL_CTX", 262144);
    if (parallel > 1 && !kv_paged) fail(TR("PARALLEL>1 需要 KV_PAGED=1", "PARALLEL>1 requires KV_PAGED=1"));
    if (rope > 1 && max_context > rope * rope_original)
        fail(TR("MAX_CONTEXT 超出 ROPE_FACTOR×ROPE_ORIGINAL_CTX 的 YaRN 上限",
                "MAX_CONTEXT exceeds the YaRN limit of ROPE_FACTOR*ROPE_ORIGINAL_CTX"));
    if (rope <= 1 && max_context > rope_original)
        warn(TR("未开启 YaRN 且 MAX_CONTEXT 超过原生上下文，超出部分的位置编码未验证",
                "YaRN is off and MAX_CONTEXT exceeds the native context; positions beyond it are unverified"));
    const long effective_pool = std::max(max_context, kv_pool);
    if (kv_pool > 0 && kv_pool < max_context) {
#ifdef _WIN32
        fail(TR("Windows 要求 KV_POOL_TOKENS=0 或不小于 MAX_CONTEXT",
                "Windows requires KV_POOL_TOKENS=0 or a value >= MAX_CONTEXT"));
#else
        hint(TR("KV_POOL_TOKENS 小于 MAX_CONTEXT：Linux 按 MAX_CONTEXT 计算有效池",
                "KV_POOL_TOKENS < MAX_CONTEXT: Linux sizes the effective pool by MAX_CONTEXT"));
#endif
    }
    if (effective_pool > 524288) {
#ifdef _WIN32
        fail(TR("有效 KV 池超过 Windows 512K 上限", "Effective KV pool exceeds the Windows 512K limit"));
#else
        warn(TR("有效 KV 池超过 512K：Linux 可尝试，但这份配置不能直接用于 Windows",
                "Effective KV pool exceeds 512K: fine to try on Linux, but this config will not work on Windows"));
#endif
    }

    const std::string affinity = model.get("AFFINITY");
    if (!affinity.empty() && !svcconf::parse_affinity(affinity))
        fail(TR("AFFINITY 必须为空、lower-half、upper-half 或非零整数 mask",
                "AFFINITY must be empty, lower-half, upper-half, or a nonzero integer mask"));

    if (!ipv4_address(model.get("ENGINE_HOST"))) fail(TR("ENGINE_HOST 必须是 IPv4 点分地址", "ENGINE_HOST must be a dotted IPv4 address"));
    if (!ipv4_address(model.get("API_HOST"))) fail(TR("API_HOST 必须是 IPv4 点分地址", "API_HOST must be a dotted IPv4 address"));
    if (model.get("ENGINE_HOST") != "127.0.0.1" && ipv4_address(model.get("ENGINE_HOST")))
        warn(TR("ENGINE_HOST 非 127.0.0.1：内部行协议会暴露到网络",
                "ENGINE_HOST is not 127.0.0.1: the internal line protocol is exposed to the network"));
    need_int("ENGINE_PORT", 1, 65535);
    need_int("API_PORT", 1, 65535);
    if (!model.get("REQLOG_LEVEL").empty()) need_int("REQLOG_LEVEL", 1, 10);
    need_int("VIZ", 0, 1);
    need_int("REQLOG", 0, 1);
    if (num("ENGINE_PORT", 8730) == num("API_PORT", 8731)) fail(TR("ENGINE_PORT 与 API_PORT 必须不同", "ENGINE_PORT and API_PORT must differ"));
    if (num("VIZ", 0)) warn(TR("运行时可视化已开启：能访问 API 的人可以看到正在生成的内容",
                               "Runtime visualization is on: anyone with API access can watch generated content"));
    if (num("REQLOG", 0)) warn(TR("完整请求日志已开启：提示词与回复会写入磁盘",
                                  "Full request logging is on: prompts and replies are written to disk"));

    for (const auto& item : v) {
        if (item.first.empty() || item.first[0] == '#') continue;
        const char* env = std::getenv(item.first.c_str());
        if (env && *env)
            warn(item.first + TR(" 正被环境变量覆盖，文件中的修改对当前进程环境不生效",
                                 " is overridden by an environment variable; file edits have no effect in the current process environment"));
    }
    bool has_fail = false;
    for (const Check& c : out) has_fail = has_fail || c.level == Level::kFail;
    if (!has_fail) add(Level::kPass, TR("静态配置校验完成；端口占用、运行库和实际内存请在启动前用启动器 --check 复核",
                                        "Static checks passed; re-check port availability, runtime libraries and actual memory with the launcher's --check before starting"));
    return out;
}

int failure_count(const std::vector<Check>& checks) {
    int n = 0;
    for (const Check& c : checks)
        if (c.level == Level::kFail) ++n;
    return n;
}

// ---- 单实例 -------------------------------------------------------------------

class SingleInstance {
public:
    bool acquire(const fs::path& conf_path, std::string* error) {
#ifdef _WIN32
        (void)conf_path;
        mutex_ = CreateMutexW(nullptr, TRUE, L"Local\\Qwenox-settings-tui");
        if (!mutex_ || GetLastError() == ERROR_ALREADY_EXISTS) {
            *error = TR("另一个 qwenox-settings 已经打开", "Another qwenox-settings instance is already open");
            return false;
        }
#else
        const std::string id = std::to_string(std::hash<std::string>{}(path_text(conf_path)));
        const fs::path lock = fs::temp_directory_path() / ("qwenox-settings-" + id + ".lock");
        fd_ = ::open(path_text(lock).c_str(), O_RDWR | O_CREAT, 0600);
        if (fd_ < 0 || flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            *error = TR("另一个 qwenox-settings 正在编辑这份配置",
                        "Another qwenox-settings instance is editing this config");
            return false;
        }
#endif
        return true;
    }

    ~SingleInstance() {
#ifdef _WIN32
        if (mutex_) CloseHandle(mutex_);
#else
        if (fd_ >= 0) ::close(fd_);
#endif
    }

private:
#ifdef _WIN32
    HANDLE mutex_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ---- TUI ----------------------------------------------------------------------

class App {
public:
    App(Model model, Terminal* term) : model_(std::move(model)), term_(term) {
        lang_sel_ = g_lang_en ? 1 : 0;
        refresh_checks();
    }

    int run() {
        running_ = true;
        while (running_) {
            draw();
            const Key key = term_->read_key();
            if (key.type == KeyType::kNone) continue;
            handle(key);
        }
        return 0;
    }

private:
    enum class Confirm { kNone, kSaveErrors, kQuitDirty };

    std::vector<const Field*> fields(int page) const {
        std::vector<const Field*> out;
        for (const Field& f : field_specs())
            if (f.page == page && field_visible(f, model_.values)) out.push_back(&f);
        return out;
    }

    const Field* selected_field() const {
        if (page_ >= 4) return nullptr;
        const auto list = fields(page_);
        if (list.empty()) return nullptr;
        const int index = std::max(0, std::min(selected_[page_], static_cast<int>(list.size()) - 1));
        return list[static_cast<size_t>(index)];
    }

    std::string display_value(const Field& f) const {
        const std::string value = model_.get(f.key);
        if (f.kind == FieldKind::kBool) return value == "1" ? TR("[x] 开启", "[x] On") : TR("[ ] 关闭", "[ ] Off");
        if (f.kind == FieldKind::kChoice) {
            for (const Option& option : f.options)
                if (value == option.value) return option.text();
        }
        return value.empty() ? TR("（空）", "(empty)") : value;
    }

    void refresh_checks() { checks_ = validate(model_); }

    void changed() {
        dirty_ = true;
        status_.clear();
        refresh_checks();
    }

    std::string expand_path_head(std::string head) const {
        const std::string model_dir = svcconf::cfg(model_.conf, "MODEL_DIR", "./models");
        auto replace_all = [&](const std::string& needle, const std::string& replacement) {
            size_t pos = 0;
            while ((pos = head.find(needle, pos)) != std::string::npos) {
                head.replace(pos, needle.size(), replacement);
                pos += replacement.size();
            }
        };
        replace_all("${MODEL_DIR}", model_dir);
        replace_all("$MODEL_DIR", model_dir);
        if (head == "~" || head.compare(0, 2, "~/") == 0 || head.compare(0, 2, "~\\") == 0) {
            const char* home = nullptr;
#ifdef _WIN32
            home = std::getenv("USERPROFILE");
            if (!home || !*home) home = std::getenv("HOME");
#else
            home = std::getenv("HOME");
#endif
            if (home && *home) head = std::string(home) + head.substr(1);
        }
        return head;
    }

    char completion_separator() const {
        return completion_.head.find('\\') != std::string::npos &&
                       completion_.head.find('/') == std::string::npos
                   ? '\\'
                   : '/';
    }

    void apply_completion(int index) {
        if (index < 0 || index >= static_cast<int>(completion_.candidates.size())) return;
        const PathCandidate& candidate = completion_.candidates[static_cast<size_t>(index)];
        const std::string inserted = candidate.name + (candidate.directory
                                                           ? std::string(1, completion_separator())
                                                           : std::string());
        std::string& value = model_.values[edit_key_];
        value = completion_.head + inserted + completion_.tail;
        edit_cursor_ = utf8_offsets(completion_.head + inserted).size() - 1;
        edit_scroll_ = 0;
        completion_.selected = index;
    }

    void cycle_completion() {
        if (completion_.candidates.empty()) return;
        const int next = (completion_.selected + 1) % static_cast<int>(completion_.candidates.size());
        apply_completion(next);
        const PathCandidate& candidate = completion_.candidates[static_cast<size_t>(next)];
        status_ = TR("候选 ", "Candidate ") + std::to_string(next + 1) + "/" +
                  std::to_string(completion_.candidates.size()) + TR("：", ": ") + candidate.name +
                  (candidate.directory ? TR("（目录）", " (dir)") : "");
    }

    void complete_path(PathKind kind) {
        if (completion_.active) {
            cycle_completion();
            return;
        }
        std::string& value = model_.values[edit_key_];
        const auto off = utf8_offsets(value);
        edit_cursor_ = std::min(edit_cursor_, off.size() - 1);
        const size_t cursor_byte = off[edit_cursor_];
        const std::string before = value.substr(0, cursor_byte);
        const size_t separator = before.find_last_of("/\\");
        const size_t name_begin = separator == std::string::npos ? 0 : separator + 1;

        CompletionState next;
        next.head = before.substr(0, name_begin);
        next.prefix = before.substr(name_begin);
        next.tail = value.substr(cursor_byte);
        const std::string expanded = expand_path_head(next.head);
        fs::path directory = expanded.empty() ? fs::path(".") : fs::path(expanded);
        if (directory.is_relative()) directory = model_.root / directory;

        std::error_code ec;
        fs::directory_iterator it(directory, ec), end;
        if (ec) {
            status_ = TR("无法读取目录：", "Cannot read directory: ") + path_text(directory);
            return;
        }
        for (; it != end; it.increment(ec)) {
            if (ec) break;
            std::error_code type_ec;
            const bool is_directory = it->is_directory(type_ec);
            if (type_ec || (kind == PathKind::kDirectory && !is_directory)) continue;
            const std::string name = path_text(it->path().filename());
            if (candidate_prefix(name, next.prefix)) next.candidates.push_back({name, is_directory});
        }
        std::sort(next.candidates.begin(), next.candidates.end(), [](const PathCandidate& a, const PathCandidate& b) {
            if (a.directory != b.directory) return a.directory > b.directory;
#ifdef _WIN32
            return lower_text(a.name) < lower_text(b.name);
#else
            return a.name < b.name;
#endif
        });
        if (next.candidates.empty()) {
            status_ = ec ? TR("读取目录时出错：", "Error reading directory: ") + path_text(directory)
                         : TR("没有匹配的路径候选", "No matching path candidates");
            return;
        }

        std::string common = next.candidates.front().name;
        for (size_t i = 1; i < next.candidates.size(); ++i)
            common.resize(common_name_prefix(common, next.candidates[i].name));
        next.active = true;
        completion_ = std::move(next);

        if (completion_.candidates.size() == 1) {
            const PathCandidate candidate = completion_.candidates.front();
            apply_completion(0);
            completion_ = CompletionState{};
            status_ = TR("已补全：", "Completed: ") + candidate.name +
                      (candidate.directory ? TR("（目录，可继续 Tab）", " (dir, Tab again to descend)") : "");
            return;
        }
        if (common.size() > completion_.prefix.size()) {
            model_.values[edit_key_] = completion_.head + common + completion_.tail;
            edit_cursor_ = utf8_offsets(completion_.head + common).size() - 1;
            edit_scroll_ = 0;
            status_ = TR("已补到公共前缀；Tab 循环候选，Enter 接受",
                         "Completed to common prefix; Tab cycles candidates, Enter accepts");
        } else {
            status_ = std::to_string(completion_.candidates.size()) +
                      TR(" 个候选；Tab 循环，Enter 接受", " candidates; Tab cycles, Enter accepts");
        }
    }

    void draw() {
        const auto size = term_->size();
        const int w = size.first, h = size.second;
        std::string frame = "\x1b[H\x1b[0m\x1b[2J";
        if (w < 50 || h < 12) {
            frame += TR("\x1b[H终端窗口太小，请至少调整为 50×12。\x1b[K",
                        "\x1b[HTerminal too small; please resize to at least 50x12.\x1b[K");
            term_->render(frame);
            return;
        }
        auto line = [&](int y, const std::string& text, const char* style = "") {
            frame += "\x1b[" + std::to_string(y + 1) + ";1H" + style + clip_text(text, w) +
                     "\x1b[0m\x1b[K";
        };
        line(0, TR(" Qwenox 跨平台设置  ", " Qwenox Settings  ") + path_text(model_.conf_path), "\x1b[1;36m");
        // 页签就地双语；英文用短名，保证 50 列最小宽度放得下 6 个页签。
        static const char* names[][2] = {{"1 权重", "1 Model"},   {"2 推理", "2 Infer"},
                                         {"3 缓存/启动", "3 Cache"}, {"4 端口/日志", "4 Ports"},
                                         {"5 检查", "5 Check"},   {"6 语言", "6 Lang"}};
        std::string tabs;
        for (int i = 0; i < 6; ++i) {
            if (i == page_) tabs += " [";
            else tabs += "  ";
            tabs += TR(names[i][0], names[i][1]);
            tabs += i == page_ ? "] " : "  ";
        }
        line(1, tabs, "\x1b[1m");
        std::string separator;
        for (int i = 0; i < w; ++i) separator += "─";
        line(2, separator);

        const int body_top = 3, body_height = h - 7;
        int cursor_x = 1, cursor_y = 1;
        bool show_cursor = false;
        if (page_ < 4) {
            const auto list = fields(page_);
            if (!list.empty()) {
                selected_[page_] = std::max(0, std::min(selected_[page_], static_cast<int>(list.size()) - 1));
                if (selected_[page_] < top_[page_]) top_[page_] = selected_[page_];
                if (selected_[page_] >= top_[page_] + body_height)
                    top_[page_] = selected_[page_] - body_height + 1;
                top_[page_] = std::max(0, std::min(top_[page_], std::max(0, static_cast<int>(list.size()) - body_height)));
                const int label_width = std::min(27, std::max(18, w / 3));
                for (int row = 0; row < body_height && top_[page_] + row < static_cast<int>(list.size()); ++row) {
                    const int index = top_[page_] + row;
                    const Field& f = *list[static_cast<size_t>(index)];
                    const bool selected = index == selected_[page_];
                    const int y = body_top + row;
                    const std::string prefix = selected ? "> " : "  ";
                    line(y, prefix + pad_text(f.label_text(), label_width), selected ? "\x1b[7m" : "");
                    const int value_x = 2 + label_width + 1;
                    const int value_width = std::max(4, w - value_x - 1);
                    std::string value = display_value(f);
                    if (selected && editing_) {
                        const std::string& raw = model_.values[edit_key_];
                        const auto off = utf8_offsets(raw);
                        edit_cursor_ = std::min(edit_cursor_, off.size() - 1);
                        if (edit_cursor_ < edit_scroll_) edit_scroll_ = edit_cursor_;
                        while (edit_scroll_ < edit_cursor_ &&
                               text_width(raw, off[edit_scroll_], off[edit_cursor_]) > value_width - 1)
                            ++edit_scroll_;
                        size_t last = edit_scroll_;
                        while (last < off.size() - 1) {
                            const int candidate = text_width(raw, off[edit_scroll_], off[last + 1]);
                            if (candidate > value_width) break;
                            ++last;
                        }
                        value = raw.substr(off[edit_scroll_], off[last] - off[edit_scroll_]);
                        cursor_x = value_x + text_width(raw, off[edit_scroll_], off[edit_cursor_]);
                        cursor_y = y + 1;
                        show_cursor = true;
                    } else {
                        value = clip_text(value, value_width, f.kind == FieldKind::kText);
                    }
                    frame += "\x1b[" + std::to_string(y + 1) + ";" + std::to_string(value_x) + "H" +
                             (selected ? "\x1b[7m" : "") + pad_text(value, value_width) + "\x1b[0m";
                }
            }
            if (editing_ && completion_.active) {
                const int shown = std::min(body_height, std::max(0, static_cast<int>(list.size()) - top_[page_]));
                const int candidate_top = body_top + shown;
                const int available = body_top + body_height - candidate_top;
                if (available > 0) {
                    line(candidate_top,
                         TR("  路径候选 ", "  Path candidates: ") + std::to_string(completion_.candidates.size()) +
                             TR(" 个（Tab 循环，Enter 接受）", " (Tab cycles, Enter accepts)"),
                         "\x1b[36m");
                    const int rows = available - 1;
                    if (rows > 0) {
                        if (completion_.selected < 0) completion_.scroll = 0;
                        if (completion_.selected >= 0 && completion_.selected < completion_.scroll)
                            completion_.scroll = completion_.selected;
                        if (completion_.selected >= completion_.scroll + rows)
                            completion_.scroll = completion_.selected - rows + 1;
                        completion_.scroll = std::max(0, std::min(
                            completion_.scroll,
                            std::max(0, static_cast<int>(completion_.candidates.size()) - rows)));
                        for (int row = 0; row < rows && completion_.scroll + row < static_cast<int>(completion_.candidates.size()); ++row) {
                            const int index = completion_.scroll + row;
                            const PathCandidate& candidate = completion_.candidates[static_cast<size_t>(index)];
                            const bool selected = index == completion_.selected;
                            line(candidate_top + 1 + row,
                                 std::string(selected ? "  > " : "    ") + candidate.name +
                                     (candidate.directory ? "/" : ""),
                                 selected ? "\x1b[7m" : (candidate.directory ? "\x1b[36m" : ""));
                        }
                    }
                }
            }
            const Field* f = selected_field();
            std::string help;
            if (editing_ && completion_.active) {
                if (completion_.selected >= 0) {
                    const PathCandidate& candidate = completion_.candidates[static_cast<size_t>(completion_.selected)];
                    help = TR("Tab 下一候选；Enter 接受 ", "Tab next candidate; Enter accepts ") + candidate.name +
                           (candidate.directory ? TR("（目录）", " (dir)") : "");
                } else {
                    help = TR("Tab 循环候选；Enter 接受当前公共前缀并继续编辑",
                              "Tab cycles candidates; Enter accepts the common prefix and keeps editing");
                }
            } else {
                help = f ? f->hint_text() : "";
            }
            line(h - 4, help, "\x1b[2m");
        } else if (page_ == 4) {
            std::vector<std::pair<Level, std::string>> rows;
            for (const Check& check : checks_) {
                const std::string prefix = std::string("[") + level_text(check.level) + "] ";
                const auto wrapped = wrap_text(prefix + check.text, w - 2);
                for (const std::string& text : wrapped) rows.push_back({check.level, text});
            }
            check_scroll_ = std::max(0, std::min(check_scroll_, std::max(0, static_cast<int>(rows.size()) - body_height)));
            for (int row = 0; row < body_height && check_scroll_ + row < static_cast<int>(rows.size()); ++row) {
                const auto& item = rows[static_cast<size_t>(check_scroll_ + row)];
                const char* color = item.first == Level::kFail   ? "\x1b[31m"
                                    : item.first == Level::kWarn ? "\x1b[33m"
                                    : item.first == Level::kPass ? "\x1b[32m"
                                                                 : "\x1b[2m";
                line(body_top + row, item.second, color);
            }
            line(h - 4, TR("环境覆盖、文件格式和平台差异在此汇总；保存前请处理失败项。",
                           "Environment overrides, file formats and platform differences are summarized here; resolve failures before saving."),
                 "\x1b[2m");
        } else {
            // 语言页：中英界面切换，Enter/空格 应用后立即生效并写入 lang 标记，
            // 不经过 F2 保存，也不触碰 dirty 与任何配置值。
            static const char* kLangNames[] = {"中文", "English"};
            for (int i = 0; i < 2; ++i) {
                const bool current = (i == 1) == g_lang_en;
                const bool selected = i == lang_sel_;
                line(body_top + i,
                     std::string(selected ? "> " : "  ") + (current ? "[x] " : "[ ] ") + kLangNames[i],
                     selected ? "\x1b[7m" : "");
            }
            line(body_top + 3, TR("↑↓ 移动，Enter/空格 应用；选择立即生效，无需保存。",
                                  "Up/Down to move, Enter/Space to apply; takes effect at once, no save needed."),
                 "\x1b[2m");
            line(h - 4, TR("写入 service.conf 的 \"# start_win: lang=\" 注释标记，Windows 设置面板/托盘读取同一标记。",
                           "Stored as the \"# start_win: lang=\" comment marker in service.conf, shared with the Windows panel/tray."),
                 "\x1b[2m");
        }

        if (confirm_ != Confirm::kNone) {
            std::string message = confirm_ == Confirm::kSaveErrors
                                      ? TR("当前有 ", "") + std::to_string(failure_count(checks_)) +
                                            TR(" 个失败项，仍保存？ y=保存 / n=取消",
                                               " failing checks; save anyway? y=save / n=cancel")
                                      : TR("有未保存修改，确定退出并放弃？ y=退出 / n=继续编辑",
                                           "Unsaved changes; quit and discard? y=quit / n=keep editing");
            line(h - 3, message, "\x1b[1;33m");
        } else {
            const std::string state = dirty_ ? TR("● 未保存  ", "● Unsaved  ") : TR("✓ 已保存  ", "✓ Saved  ");
            line(h - 3, state + status_, dirty_ ? "\x1b[33m" : "\x1b[32m");
        }
        line(h - 2, TR("↑↓选择  ←→/数字翻页  Enter编辑/切换  Tab下一项  F2/Ctrl+S保存  Esc退出",
                       "↑↓ select  ←→/1-6 pages  Enter edit/toggle  Tab next  F2/Ctrl+S save  Esc quit"), "\x1b[2m");
        line(h - 1, editing_ ? TR("编辑中：路径按 Tab 补全/循环，Enter 接受/确认，←→移动，Ctrl+U清空，Esc取消",
                                  "Editing: Tab completes/cycles paths, Enter accepts, ←→ move, Ctrl+U clears, Esc cancels")
                             : TR("本工具只修改 service.conf；保存后重启服务生效，不启动引擎或 API。",
                                  "Only service.conf is modified; restart the service after saving. The engine or API is never started."),
             "\x1b[2m");
        if (show_cursor)
            frame += "\x1b[" + std::to_string(cursor_y) + ";" + std::to_string(cursor_x) + "H\x1b[?25h";
        else
            frame += "\x1b[?25l";
        term_->render(frame);
    }

    void handle(const Key& key) {
        if (confirm_ != Confirm::kNone) {
            if (key.type == KeyType::kCharacter && (key.text == "y" || key.text == "Y")) {
                const Confirm action = confirm_;
                confirm_ = Confirm::kNone;
                if (action == Confirm::kSaveErrors) save_now();
                else running_ = false;
            } else if ((key.type == KeyType::kCharacter && (key.text == "n" || key.text == "N")) ||
                       key.type == KeyType::kEscape) {
                confirm_ = Confirm::kNone;
            }
            return;
        }
        if (editing_) {
            handle_edit(key);
            return;
        }
        switch (key.type) {
            case KeyType::kQuit:
            case KeyType::kEscape:
                if (dirty_) confirm_ = Confirm::kQuitDirty;
                else running_ = false;
                return;
            case KeyType::kSave:
                request_save();
                return;
            case KeyType::kLeft:
                page_ = (page_ + 5) % 6;
                return;
            case KeyType::kRight:
                page_ = (page_ + 1) % 6;
                return;
            case KeyType::kUp:
                if (page_ == 4) --check_scroll_;
                else if (page_ == 5) { if (lang_sel_ > 0) --lang_sel_; }
                else if (selected_[page_] > 0) --selected_[page_];
                return;
            case KeyType::kDown:
                if (page_ == 4) ++check_scroll_;
                else if (page_ == 5) { if (lang_sel_ < 1) ++lang_sel_; }
                else if (selected_[page_] + 1 < static_cast<int>(fields(page_).size())) ++selected_[page_];
                return;
            case KeyType::kTab:
                if (page_ < 4 && !fields(page_).empty())
                    selected_[page_] = (selected_[page_] + 1) % static_cast<int>(fields(page_).size());
                return;
            case KeyType::kBackTab:
                if (page_ < 4 && !fields(page_).empty())
                    selected_[page_] = (selected_[page_] + static_cast<int>(fields(page_).size()) - 1) %
                                       static_cast<int>(fields(page_).size());
                return;
            case KeyType::kEnter:
                if (page_ == 5) {
                    apply_language();
                    return;
                }
                activate();
                return;
            case KeyType::kCharacter:
                if (key.text == "q" || key.text == "Q") {
                    if (dirty_) confirm_ = Confirm::kQuitDirty;
                    else running_ = false;
                } else if (page_ == 5 && key.text == " ") {
                    apply_language();
                } else if (key.text.size() == 1 && key.text[0] >= '1' && key.text[0] <= '6') {
                    page_ = key.text[0] - '1';
                }
                return;
            default:
                return;
        }
    }

    void activate() {
        const Field* f = selected_field();
        if (!f) return;
        if (f->kind == FieldKind::kText) {
            editing_ = true;
            completion_ = CompletionState{};
            status_.clear();
            edit_key_ = f->key;
            edit_original_ = model_.get(edit_key_);
            edit_cursor_ = utf8_offsets(edit_original_).size() - 1;
            edit_scroll_ = 0;
            return;
        }
        std::string& value = model_.values[f->key];
        if (f->kind == FieldKind::kBool) {
            value = value == "1" ? "0" : "1";
        } else if (!f->options.empty()) {
            size_t index = 0;
            for (; index < f->options.size(); ++index)
                if (value == f->options[index].value) break;
            value = f->options[(index + 1) % f->options.size()].value;
        }
        changed();
        if (std::string(f->key) == "#WEIGHTS") selected_[0] = top_[0] = 0;
    }

    void commit_edit(bool move_next = false) {
        if (!editing_) return;
        editing_ = false;
        completion_ = CompletionState{};
        if (model_.get(edit_key_) != edit_original_) changed();
        if (move_next && page_ < 4 && !fields(page_).empty())
            selected_[page_] = (selected_[page_] + 1) % static_cast<int>(fields(page_).size());
    }

    void handle_edit(const Key& key) {
        if (key.type == KeyType::kSave) {
            commit_edit();
            request_save();
            return;
        }
        if (key.type == KeyType::kQuit || key.type == KeyType::kEscape) {
            model_.values[edit_key_] = edit_original_;
            editing_ = false;
            completion_ = CompletionState{};
            return;
        }
        if (key.type == KeyType::kEnter) {
            if (completion_.active) {
                completion_ = CompletionState{};
                status_ = TR("已接受补全；继续 Tab 可补下一级，Enter 确认字段",
                             "Completion accepted; Tab again for the next level, Enter commits the field");
                return;
            }
            commit_edit();
            return;
        }
        if (key.type == KeyType::kTab) {
            const Field* f = selected_field();
            const PathKind kind = f ? path_kind(*f) : PathKind::kNone;
            if (kind == PathKind::kNone) commit_edit(true);
            else complete_path(kind);
            return;
        }
        if (key.type == KeyType::kBackTab) {
            completion_ = CompletionState{};
            commit_edit();
            if (page_ < 4 && !fields(page_).empty())
                selected_[page_] = (selected_[page_] + static_cast<int>(fields(page_).size()) - 1) %
                                   static_cast<int>(fields(page_).size());
            return;
        }
        if (key.type == KeyType::kNone || key.type == KeyType::kResize) return;
        completion_ = CompletionState{};
        status_.clear();
        std::string& value = model_.values[edit_key_];
        auto off = utf8_offsets(value);
        edit_cursor_ = std::min(edit_cursor_, off.size() - 1);
        switch (key.type) {
            case KeyType::kLeft:
                if (edit_cursor_ > 0) --edit_cursor_;
                return;
            case KeyType::kRight:
                if (edit_cursor_ + 1 < off.size()) ++edit_cursor_;
                return;
            case KeyType::kHome:
                edit_cursor_ = 0;
                return;
            case KeyType::kEnd:
                edit_cursor_ = off.size() - 1;
                return;
            case KeyType::kBackspace:
                if (edit_cursor_ > 0) {
                    value.erase(off[edit_cursor_ - 1], off[edit_cursor_] - off[edit_cursor_ - 1]);
                    --edit_cursor_;
                }
                return;
            case KeyType::kDelete:
                if (edit_cursor_ + 1 < off.size())
                    value.erase(off[edit_cursor_], off[edit_cursor_ + 1] - off[edit_cursor_]);
                return;
            case KeyType::kCharacter:
                if (key.text.size() == 1 && static_cast<unsigned char>(key.text[0]) == 21) {
                    value.clear();
                    edit_cursor_ = edit_scroll_ = 0;
                    return;
                }
                value.insert(off[edit_cursor_], key.text);
                edit_cursor_ += utf8_offsets(key.text).size() - 1;
                return;
            default:
                return;
        }
    }

    void request_save() {
        refresh_checks();
        if (failure_count(checks_) > 0) confirm_ = Confirm::kSaveErrors;
        else save_now();
    }

    // 语言页应用所选语言：立即切换界面语言并写入 lang 注释标记（与 Windows
    // 面板同款即时落盘），不走 F2 保存流程，也不影响配置值与 dirty 标记。
    void apply_language() {
        const bool en = lang_sel_ == 1;
        if (en == g_lang_en) {
            status_ = TR("界面语言未变化", "UI language unchanged");
            return;
        }
        g_lang_en = en;
        svcconf::conf_write_marker(path_text(model_.conf_path), kLangMarker, en ? "en" : "zh");
        status_ = TR("界面语言已切换并写入 lang 标记", "UI language switched and saved to the lang marker");
    }

    void save_now() {
        std::string error;
        if (!svcconf::save_conf(path_text(model_.conf_path), model_.conf_values(), &error,
                                kWeightsMarker, model_.get("#WEIGHTS", "v2"))) {
            status_ = TR("保存失败：", "Save failed: ") + error;
            return;
        }
        dirty_ = false;
        status_ = TR("已写回配置并生成 .bak；重启服务后生效",
                     "Config written back with a .bak backup; restart the service to apply");
        refresh_checks();
    }

    Model model_;
    Terminal* term_;
    int page_ = 0;
    int selected_[4] = {};
    int top_[4] = {};
    int check_scroll_ = 0;
    int lang_sel_ = 0;
    bool running_ = false;
    bool dirty_ = false;
    bool editing_ = false;
    std::string edit_key_, edit_original_;
    size_t edit_cursor_ = 0, edit_scroll_ = 0;
    CompletionState completion_;
    Confirm confirm_ = Confirm::kNone;
    std::string status_;
    std::vector<Check> checks_;
};

void usage(const char* argv0) {
    std::cout << TR("用法：", "Usage: ") << argv0 << " [--conf PATH] [--check]\n"
              << TR("  --conf PATH  编辑指定 service.conf；相对路径按当前目录解析\n",
                    "  --conf PATH  Edit the given service.conf; relative paths resolve against the current directory\n")
              << TR("  --check      只做静态校验并打印结果，不进入 TUI、不写文件\n",
                    "  --check      Run static checks and print the results without entering the TUI or writing files\n")
              << TR("  -h, --help   显示帮助\n",
                    "  -h, --help   Show this help\n")
              << TR("按键：↑↓选择，←→或数字 1–6 翻页，Enter 编辑/切换，F2/Ctrl+S 保存，Esc 退出。\n",
                    "Keys: ↑↓ select, ←→ or digits 1-6 switch pages, Enter edit/toggle, F2/Ctrl+S save, Esc quit.\n")
              << TR("路径编辑：Tab 补全/循环候选，Enter 接受补全；再按 Enter 确认字段。\n",
                    "Path editing: Tab completes/cycles candidates, Enter accepts a completion; Enter again commits the field.\n")
              << TR("界面语言：第 6 页「语言」切换中英界面；也可设 UI_LANG=zh|en 或 service.conf 的 \"# start_win: lang=\" 标记。\n",
                    "UI language: page 6 (Language) switches between Chinese and English; UI_LANG=zh|en or the \"# start_win: lang=\" marker in service.conf also work.\n");
}

}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
    std::setlocale(LC_ALL, "");
#endif
    // 先粗扫一遍 --conf：界面语言的解析依赖 conf 路径，而帮助/报错文案也要随语言。
    // 粗扫不报错（--conf 缺值等情况留给下面的正式参数循环处理）。
    fs::path conf;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--conf") { conf = argv[i + 1]; break; }

    Model model;
    if (!conf.empty()) {
        model.conf_path = absolute_path(conf);
        model.root = model.conf_path.parent_path();
    } else {
        const fs::path exe = executable_path();
        fs::path root = exe.parent_path();
        if (root.filename() == "build") root = root.parent_path();
        std::error_code ec;
        if (!fs::exists(root / "service.conf", ec) && fs::exists(exe.parent_path() / "service.conf"))
            root = exe.parent_path();
        model.root = root;
        model.conf_path = root / "service.conf";
    }
    lang_resolve(path_text(model.conf_path));

    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--check") {
            check_only = true;
        } else if (arg == "--conf") {
            if (i + 1 >= argc) {
                std::cerr << TR("错误：--conf 缺少路径\n", "error: --conf requires a path\n");
                return 2;
            }
            ++i;  // 路径已在上方粗扫时解析，这里只做校验
        } else {
            std::cerr << TR("错误：未知参数 ", "error: unknown argument ") << arg << "\n";
            usage(argv[0]);
            return 2;
        }
    }
    model.load();

    if (check_only) {
        const auto checks = validate(model);
        for (const Check& c : checks)
            std::cout << '[' << level_text(c.level) << "] " << c.text << '\n';
        return failure_count(checks) ? 1 : 0;
    }

    std::string error;
    SingleInstance instance;
    if (!instance.acquire(model.conf_path, &error)) {
        std::cerr << TR("错误：", "error: ") << error << '\n';
        return 1;
    }
    Terminal terminal;
    if (!terminal.open(&error)) {
        std::cerr << TR("错误：", "error: ") << error << '\n';
        return 1;
    }
    App app(std::move(model), &terminal);
    const int result = app.run();
    terminal.close();
    return result;
}
