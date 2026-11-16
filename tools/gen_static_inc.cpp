// gen_static_inc.cpp —— 构建时工具:把 src/api/static/ 下的文件以字节数组形式
// 嵌入成 C++ 源文件(static_gen.inc)。目录里放自由格式的原始 HTML/CSS/JS 等,
// 无任何语法约束;build.sh / build_win.sh 编译 API 前自动重跑本工具。
//
// 用法: gen_static_inc <static_dir> <output.inc> [--map=<url>=<磁盘路径> ...]
// 规则: HTML 文件路由去掉 .html/.htm 后缀(index.html/index.htm -> "/");
//       其余文件 -> "/<文件名>";隐藏文件(以 . 或 _ 开头)与子目录跳过;
//       --map 把目录外的单个文件嵌入为指定 URL(图标等资产多程序共用一份源文件),
//       url 不带前导斜杠(参数里出现 /x 会被 Git Bash 改写成安装目录路径);
//       输出按 URL 排序,内容未变化时不重写(避免 mtime 抖动与无效重编)。
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Asset {
    std::string url;    // 对外路由,如 "/dashboard"
    std::string file;   // 磁盘上的文件名
    std::string symbol; // 生成的 C 标识符
    std::string mime;
    std::string data;
};

std::string mime_for(const std::string& name) {
    static const std::map<std::string, const char*> kTypes = {
        {".html", "text/html; charset=utf-8"},
        {".htm", "text/html; charset=utf-8"},
        {".css", "text/css; charset=utf-8"},
        {".js", "application/javascript; charset=utf-8"},
        {".json", "application/json"},
        {".txt", "text/plain; charset=utf-8"},
        {".svg", "image/svg+xml"},
        {".png", "image/png"},
        {".jpg", "image/jpeg"},
        {".jpeg", "image/jpeg"},
        {".gif", "image/gif"},
        {".webp", "image/webp"},
        {".ico", "image/x-icon"},
        {".woff", "font/woff"},
        {".woff2", "font/woff2"},
    };
    std::string ext = fs::path(name).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    auto it = kTypes.find(ext);
    return it != kTypes.end() ? it->second : "application/octet-stream";
}

// 文件名 -> 合法 C 标识符:非字母数字一律换成 '_';前缀保证首字符不是数字。
std::string symbol_for(const std::string& name) {
    std::string s = "kStatic_";
    for (char c : name)
        s.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
    return s;
}

// 文件名 -> URL:HTML 文件去后缀(index -> "/"),其余保持原名。
std::string url_for(const std::string& name) {
    fs::path p(name);
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".html" || ext == ".htm") {
        std::string stem = p.stem().string();
        return stem == "index" ? "/" : "/" + stem;
    }
    return "/" + name;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <static_dir> <output.inc> [--map=url=file ...]\n", argv[0]);
        return 2;
    }
    std::error_code ec;
    if (!fs::is_directory(argv[1], ec)) {
        fprintf(stderr, "%s: not a directory\n", argv[1]);
        return 1;
    }

    std::vector<Asset> assets;
    std::set<std::string> urls, symbols;
    auto add_asset = [&](const std::string& url, const fs::path& path,
                         const std::string& name) -> bool {
        Asset a;
        a.file = name;
        a.url = url;
        a.symbol = symbol_for(name);
        // mime 优先按对外 URL 的后缀(映射文件按映射名);URL 无后缀时按源文件名
        a.mime = mime_for(fs::path(url).has_extension() ? url : name);
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            fprintf(stderr, "%s: cannot open %s\n", argv[0], path.string().c_str());
            return false;
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        a.data = buf.str();
        if (!urls.insert(a.url).second) {
            fprintf(stderr, "%s: URL collision: %s\n", argv[0], a.url.c_str());
            return false;
        }
        if (!symbols.insert(a.symbol).second) {
            fprintf(stderr, "%s: symbol collision: %s\n", argv[0], a.symbol.c_str());
            return false;
        }
        assets.push_back(std::move(a));
        return true;
    };
    for (const auto& e : fs::directory_iterator(argv[1], ec)) {
        if (!e.is_regular_file()) continue;
        std::string name = e.path().filename().string();
        if (name.empty() || name[0] == '.' || name[0] == '_') continue;
        if (!add_asset(url_for(name), e.path(), name)) return 1;
    }
    // --map=<url>=<路径>:目录之外的单文件按指定 URL 嵌入(多程序共用一份图标源)。
    // url 不带前导斜杠——Git Bash 会把参数中的 /x 段改写成安装目录路径,斜杠由这里补。
    for (int i = 3; i < argc; ++i) {
        std::string spec = argv[i];
        if (spec.rfind("--map=", 0) != 0) {
            fprintf(stderr, "%s: unknown arg: %s\n", argv[0], spec.c_str());
            return 2;
        }
        std::string m = spec.substr(6);
        size_t eq = m.find('=');
        if (eq == std::string::npos || eq == 0 || m[0] == '/') {
            fprintf(stderr, "%s: bad --map (want --map=url=file, url 不带斜杠): %s\n", argv[0], spec.c_str());
            return 2;
        }
        std::string url = "/" + m.substr(0, eq);
        if (!add_asset(url, fs::path(m.substr(eq + 1)),
                       fs::path(url).filename().string()))
            return 1;
    }
    if (assets.empty()) {
        fprintf(stderr, "%s: no embeddable files under %s\n", argv[0], argv[1]);
        return 1;
    }
    std::sort(assets.begin(), assets.end(),
              [](const Asset& x, const Asset& y) { return x.url < y.url; });

    std::ostringstream out;
    out << "// static_gen.inc —— 由 tools/gen_static_inc.cpp 生成,请勿手工编辑!\n"
           "// 来源目录与 --map 映射见构建脚本调用,HTML 文件路由去 .html 后缀(index 对应根路径)。\n"
           "// main.cpp include 本文件后按 kStaticAssets 逐条注册 GET 路由。\n"
           "// file 是原始文件名：WEB_DIR（默认项目根 web/）下有同名文件时优先返回磁盘版。\n"
           "#ifndef QWENOX_STATIC_GEN_INC_\n"
           "#define QWENOX_STATIC_GEN_INC_\n"
           "#include <cstddef>\n"
           "\n"
           "struct StaticAsset {\n"
           "    const char* url;\n"
           "    const char* file;\n"
           "    const char* mime;\n"
           "    const unsigned char* data;\n"
           "    std::size_t len;\n"
           "};\n";
    for (const Asset& a : assets) {
        out << "\n// " << a.file << " (" << a.data.size() << " bytes)\n"
            << "static const unsigned char " << a.symbol << "[] = {";
        if (a.data.empty()) {
            out << " 0x00";  // 占位,避免零长数组(-Wpedantic)
        } else {
            for (size_t i = 0; i < a.data.size(); ++i) {
                if (i % 16 == 0) out << "\n   ";
                char hex[8];
                snprintf(hex, sizeof(hex), " 0x%02x,", static_cast<unsigned char>(a.data[i]));
                out << hex;
            }
            out << "\n";
        }
        out << "};";
    }
    out << "\n\nstatic const StaticAsset kStaticAssets[] = {\n";
    for (const Asset& a : assets)
        out << "    {\"" << a.url << "\", \"" << a.file << "\", \"" << a.mime << "\", " << a.symbol << ", "
            << a.data.size() << "},\n";
    out << "};\n"
           "#endif  // QWENOX_STATIC_GEN_INC_\n";

    // 与现有内容逐字节一致则不重写,避免无意义的 git diff / 重编。
    std::string text = out.str();
    {
        std::ifstream old(argv[2], std::ios::binary);
        std::ostringstream ob;
        ob << old.rdbuf();
        if (old && ob.str() == text) return 0;
    }
    std::ofstream f(argv[2], std::ios::binary | std::ios::trunc);
    if (!f) {
        fprintf(stderr, "%s: cannot write %s\n", argv[0], argv[2]);
        return 1;
    }
    f << text;
    fprintf(stderr, "generated %s (%zu assets)\n", argv[2], assets.size());
    return 0;
}
