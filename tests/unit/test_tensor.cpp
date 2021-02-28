#include <deepworks.hpp>
#include <gtest/gtest.h>
#include <tensor.hpp>

TEST(TensorTest, Strides) {
    struct CaseData {
        deepworks::Shape actual_shape;
        deepworks::Strides expected_strides;
    };
    auto test_cases = std::vector<CaseData>{
            {deepworks::Shape{1, 3, 224, 224}, deepworks::Strides{150528, 50176, 224, 1}},
            {deepworks::Shape{1, 200, 200, 300}, deepworks::Strides{12000000, 60000, 300, 1}},
            {deepworks::Shape{32, 200, 200, 300}, deepworks::Strides{12000000, 60000, 300, 1}},
            {deepworks::Shape{32, 3, 64, 64}, deepworks::Strides{12288, 4096, 64, 1}},
            {deepworks::Shape{4, 6, 12, 18}, deepworks::Strides{1296, 216, 18, 1}},
            {deepworks::Shape{4, 6, 12, 18}, deepworks::Strides{1296, 216, 18, 1}},
            {deepworks::Shape{2, 5}, deepworks::Strides{5, 1}},
            {deepworks::Shape{1}, deepworks::Strides{1}},
            {deepworks::Shape{}, deepworks::Strides{}},
    };
    for (const auto& tcase : test_cases) {
        auto tensor = deepworks::Tensor(tcase.actual_shape);
        EXPECT_EQ(tensor.strides(), tcase.expected_strides);
    }
}

TEST(TensorTest, Shape) {
    auto shapes = std::vector<deepworks::Shape>{
            deepworks::Shape{1, 3, 224, 224},
            deepworks::Shape{1, 200, 200, 300},
            deepworks::Shape{32, 200, 200, 300},
            deepworks::Shape{32, 3, 64, 64},
            deepworks::Shape{4, 6, 12, 18},
    };
    for (const auto& shape : shapes) {
        auto tensor = deepworks::Tensor(shape);
        EXPECT_EQ(tensor.shape(), shape);
    }
}

TEST(TensorTest, DefaultCtor) {
    deepworks::Tensor tensor;
    EXPECT_EQ(tensor.shape(), deepworks::Shape{});
    EXPECT_EQ(tensor.strides(), deepworks::Strides{});
    EXPECT_EQ(tensor.data(), nullptr);
    ASSERT_NO_THROW(tensor.copyTo(tensor));
}

TEST(TensorTest, Reassignment) {
    deepworks::Tensor src_tensor({1, 3, 224, 224});
    deepworks::Tensor tensor;

    tensor = src_tensor;

    EXPECT_EQ(tensor.shape(), src_tensor.shape());
    EXPECT_EQ(tensor.strides(), src_tensor.strides());
    EXPECT_EQ(tensor.data(), src_tensor.data());
}

TEST(TensorTest, CopyTo) {
    {
        deepworks::Tensor src_tensor({1, 3, 224, 224});
        for (size_t index = 0; index < 1 * 3 * 224 * 224; ++index) {
            src_tensor.data()[index] = index;
        }

        deepworks::Tensor dst_tensor;
        src_tensor.copyTo(dst_tensor);

        ASSERT_EQ(dst_tensor.shape(), src_tensor.shape());
        ASSERT_EQ(dst_tensor.strides(), src_tensor.strides());
        ASSERT_NE(dst_tensor.data(), src_tensor.data());
        for (size_t index = 0; index < 1 * 3 * 224 * 224; ++index) {
            ASSERT_EQ(dst_tensor.data()[index], index);
        }

        deepworks::Tensor non_empty_tensor({1, 3, 16, 16});
        ASSERT_THROW(src_tensor.copyTo(non_empty_tensor), std::runtime_error);
    }
    {
        deepworks::Tensor src_tensor({1, 3, 224, 224});
        deepworks::Tensor dst_tensor;
        ASSERT_THROW(dst_tensor.copyTo(src_tensor), std::runtime_error);

    }
}