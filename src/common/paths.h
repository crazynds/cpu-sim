#pragma once

#include <string>

namespace cpusim {

struct Config;

// $CPUSIM_CONFIG, or $XDG_CONFIG_HOME/cpusim/device.toml, or ~/.config/cpusim/device.toml.
std::string config_path();

// The driver's control socket: cfg.socket if set, else $XDG_RUNTIME_DIR/cpusim/driver.sock,
// else /tmp/cpusim-<uid>/driver.sock.
std::string socket_path(const Config& cfg);

}  // namespace cpusim
