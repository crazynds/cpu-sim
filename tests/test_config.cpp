#include <string>

#include "check.h"
#include "common/config.h"
#include "common/units.h"

using namespace cpusim;

int main() {
    EXPECT(parse_size("16GiB") == 16 * GiB);
    EXPECT(parse_size(" 512 MiB ") == 512 * MiB);
    EXPECT(parse_size("4096") == 4096u);
    EXPECT(!parse_size("12XB"));
    EXPECT(format_size(2 * GiB) == "2GiB");
    EXPECT(format_cpu_list(*parse_cpu_list("4-7,20,21,22,1")) == "1,4-7,20-22");

    std::string err;
    auto cfg = parse_config(
        "# comment\n[device]\nname = \"My #1 GPU\"  # trailing\ncores = 12\nvram_private = \"8GiB\"\n"
        "hugepages = \"none\"\nprefault = true\n[driver]\nsocket = \"/tmp/x.sock\"\n",
        &err);
    EXPECT(cfg.has_value());
    if (cfg) {
        EXPECT(cfg->name == "My #1 GPU");
        EXPECT(cfg->cores == 12);
        EXPECT(cfg->vram_private == 8 * GiB);
        EXPECT(cfg->hugepages == HugePages::None);
        EXPECT(cfg->prefault);
        EXPECT(cfg->socket == "/tmp/x.sock");
    }
    EXPECT(!parse_config("[device]\nvram = \"1GiB\"\n", &err) && err.find("line 2") == 0);
    EXPECT(!parse_config("[gpu]\n", &err));
    EXPECT(!parse_config("[device]\ncores = \"many\"\n", &err));
    TEST_MAIN_END();
}
