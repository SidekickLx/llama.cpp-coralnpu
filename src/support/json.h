// SPDX-License-Identifier: MIT
#pragma once
#include <iomanip>
#include <sstream>
#include <string>

namespace coral {
inline std::string JsonString(const std::string &value) {
    std::ostringstream output;
    output << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            output << '\\' << c;
        } else if (c < 0x20) {
            output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        } else {
            output << c;
        }
    }
    output << '"';
    return output.str();
}
} // namespace coral
