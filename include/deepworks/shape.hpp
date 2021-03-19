#pragma once

#include <vector>
#include <iostream>

namespace deepworks {

using Shape = std::vector<int>;

} // namespace deepworks;

inline std::ostream& operator<<(std::ostream& stream, const deepworks::Shape& shape) {
    stream << "Shape(";
    size_t index = 0;
    for (; index + 1 < shape.size(); ++index) {
        stream << shape[index] << ", ";
    }
    if (!shape.empty()) {
        stream << shape[index];
    }
    stream << ")";
    return stream;
}