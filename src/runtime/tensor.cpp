#include <deepworks/tensor.hpp>

#include <numeric>
#include <algorithm>
#include <iostream>

namespace deepworks {

Tensor::Descriptor::Descriptor(const Shape &shape) : m_shape(shape) {
    allocate(shape);
}

void Tensor::Descriptor::copyTo(Tensor::Descriptor &descriptor) {
    if (this == &descriptor) {
        throw std::runtime_error("Tensor cannot copy itself.");
    }
    if (m_shape != descriptor.m_shape || m_strides != descriptor.m_strides) {
        throw std::runtime_error("Copy to another layout isn't supported.");
    }

    if (descriptor.m_data == nullptr) {
        throw std::runtime_error("copyTo: Output tensor should be allocated.");
    }

    m_shape = descriptor.m_shape;
    m_strides = descriptor.m_strides;

    std::copy_n(m_data, m_total, descriptor.m_data);
}

void Tensor::Descriptor::allocate(const Shape &shape) {
    bool have_negative_dim = std::any_of(shape.begin(), shape.end(), [](int dim) {
        return dim < 0;
    });
    if (have_negative_dim) {
        throw std::runtime_error("Cannot allocate tensor dynamic shape.");
    }
    if (m_data != nullptr) {
        throw std::runtime_error("Tensor already allocated, cannot allocate twice.");
    }
    m_shape = shape;
    m_total = std::accumulate(shape.begin(),
                              shape.end(),
                              1,
                              std::multiplies<>());
    m_data = new Type[m_total];
    calculateStrides(shape);
}

void Tensor::Descriptor::calculateStrides(const Shape &shape) {
    m_strides.resize(shape.size());

    size_t initial_stride = 1;
    auto dim_it = shape.rbegin();
    for (auto stride_it = m_strides.rbegin(); stride_it != m_strides.rend(); ++stride_it, ++dim_it) {
        *stride_it = initial_stride;
        initial_stride *= *dim_it;
    }
}

Tensor::Descriptor::~Descriptor() {
    delete[] m_data;
}

/* Tensor */
Tensor::Tensor() : m_descriptor(new Descriptor()) {
}

Tensor::Tensor(const Shape &shape) : m_descriptor(new Descriptor(shape)) {
}

void Tensor::copyTo(Tensor tensor) {
    m_descriptor->copyTo(*(tensor.m_descriptor));
}

Tensor::Type *Tensor::data() {
    return m_descriptor->m_data;
}

const Tensor::Type *Tensor::data() const {
    return m_descriptor->m_data;
}

size_t Tensor::total() const {
    return m_descriptor->m_total;
}

void Tensor::allocate(const Shape &shape) {
    m_descriptor->allocate(shape);
}

bool Tensor::empty() const {
    return m_descriptor->m_total == 0;
}

const Strides &Tensor::strides() const {
    return m_descriptor->m_strides;
}

const Shape &Tensor::shape() const {
    return m_descriptor->m_shape;
}

namespace {
template<class T>
struct View {
    const T *data;
    size_t length;
};

void PrintArray(const View<float> &tensor_data, std::ostream &stream) {
    stream << "[";
    size_t index = 0;
    for (; index + 1 < tensor_data.length; ++index) {
        stream << tensor_data.data[index] << ", ";
    }
    stream << tensor_data.data[index] << "]";
}

void PrintTensor(const View<float> &tensor_data, const View<int> &shape, std::ostream &stream) {
    if (tensor_data.length == 0ul || shape.length == 0ul) {
        stream << "Tensor()";
        return;
    }
    if (shape.length == 1ul) {
        stream << "Tensor(";
        PrintArray(tensor_data, stream);
        stream << ")";
        return;
    }
    stream << "Tensor([";
    const size_t current_dim = shape.data[0];
    const size_t elements_in_dim = tensor_data.length / current_dim;
    for (size_t index = 0; index < current_dim; ++index) {
        PrintTensor(View<float>{tensor_data.data + index * elements_in_dim, elements_in_dim},
                    View<int>{shape.data + 1, shape.length - 1},
                    stream);
        if (index + 1 != current_dim) {
            stream << ", " << std::endl;
        }
    }
    stream << "])";
}
}


std::ostream &operator<<(std::ostream &stream, const Tensor &tensor) {
    if (tensor.total() == 0) {
        return stream;
    }
    const auto shape = tensor.shape();
    const auto shape_view = View<int>{shape.data(), shape.size()};
    const auto data_view = View<float>{tensor.data(), tensor.total()};

    PrintTensor(data_view, shape_view, stream);

    return stream;
}

} // namespace deepworks
