/**
 * \file Range.h
 * \brief View and attack range in GridWorld
 */

#ifndef MAGNET_GRIDWORLD_RANGE_H
#define MAGNET_GRIDWORLD_RANGE_H

#include <cstdio>
#include <tgmath.h>
#include <cstring>

namespace magent {
namespace gridworld {

static const double PI = 3.1415926536;

class Range {
public:
    // 构造函数，width等默认-1，-1, 0
    Range() : width(-1), height(-1), count(0) {
        is_in_range = nullptr;
        dx = dy = nullptr;
    }

    // 拷贝构造函数
    Range(const Range &other) :  width(other.width), height(other.height), count(other.count) {
        // 根据原实例的 width 和 height 计算得出（width * height），用于存储布尔类型的数据，相当于初始化矩阵
        is_in_range = new bool[width * height];
        dx = new int[width * height];
        dy = new int[width * height];

        // 将other.is_in_range的内容按照第三个输入的空间（为了确保三个输入的维度都相同）完整地拷贝到is_in_range
        memcpy(is_in_range, other.is_in_range, sizeof(bool) * width * height);
        memcpy(dx, other.dx, sizeof(bool) * width * height);
        memcpy(dy, other.dy, sizeof(bool) * width * height);
    }

    ~Range() {
        if (is_in_range != nullptr)
            delete [] is_in_range;
        if (dx != nullptr)
            delete [] dx;
        if (dy != nullptr)
            delete [] dy;
    }

    // is_in_range是一个一维数组（把矩阵拍扁），现在输入row和col判断是否在range内
    bool is_in(int row, int col) const {
        return is_in_range[row * width + col];
    }

    int get_width()  const { return width; }
    int get_height() const { return height; }

    // 将对象的属性x1，x2，y1，y2赋值给传入的参数
    void get_range_rela_offset(int &x1, int &y1, int &x2, int &y2) const {
        x1 = this->x1; y1 = this->y1;
        x2 = this->x2; y2 = this->y2;
    }

    // 为x1，x2等加入偏移量x_off，y_off
    int add_rela_offset(int x_off, int y_off) {
        x1 += x_off; x2 += x_off;
        y1 += y_off; y2 += y_off;
        return -1;
    }

    int get_count() const { return count; }

    // 这个函数的作用是根据传入的索引值 n，从 dx 和 dy 数组中获取相应的偏移量信息，并将其分别赋给传入的引用参数 dx 和 dy。
    void num2delta(int n, int &dx, int &dy) const {
        // do not check boundary
        dx = this->dx[n];
        dy = this->dy[n];
    }

    // 遍历每个格子，打印is_in_range信息
    void print_self() {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (is_in_range[i * width + j]) {
                    printf("1");
                } else
                    printf("0");
            }
            printf("\n");
        }
        int ct = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (is_in_range[i * width + j]) {
                    printf("%2d,%2d ", dx[ct], dy[ct]);
                    ct++;
                } else {
                    printf("%2d,%2d ", 0, 0);
                }
            }
            printf("\n");
        }
        printf("\n");
    }

protected:
    int width, height;
    int count;
    int x1, y1, x2, y2;
    bool *is_in_range;
    int *dx;
    int *dy;
};

// sector range
// stored as a rectangle with sector mask
class SectorRange : public Range {
public:
    // 构造函数
    // angle表示扇形角度；radius表示扇形半径；parity与奇偶性相关
    SectorRange(float angle, float radius, int parity) {
        // 四舍五入取整
        // height为扇形的半径
        height = (int)(radius + 0.5);
        // width为扇形的边长
        width  = (int)(2 * radius * sin(angle / 2 * (PI / 180)) + 0.5);
        // 使宽度符合奇偶性要求
        if (width % 2 != parity) {  // fit to parity, pick ceil
            width--;
        }

        is_in_range = new bool[width * height];
        dx = new int[width * height];
        dy = new int[width * height];

        const double eps = 0.00001;

        count = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double dis_x, dis_y;

                // 与扇形中心线距离的绝对值
                dis_x = std::fabs(j - (width-1)/2.0);
                // 与扇形高度距离的绝对值
                dis_y = std::fabs(height - i);

                double dis = sqrt(dis_x * dis_x + dis_y * dis_y);

                // 判断is_in_range的每个元素是否在扇形范围内，填充is_in_range
                // count表示在扇形范围内的格子的个数
                if (dis < radius + 0.2 + eps && dis_x / dis_y
                                                < tan(angle / 2 * PI / 180) + eps) {
                    is_in_range[i * width + j] = true;
                    dx[count] = j - width/2;
                    dy[count] = i - height;
                    count++;
                } else {
                    is_in_range[i * width + j] = false;
                }
            }
        }

        x1 = -width / 2; y1 = -height;
        x2 = (width-1) / 2; y2 = -1;
    }
};


// circle range
// stored as a rectangle with circular mask
// 圆环类
class CircleRange : public Range {
public:
    // radius外半径；inner_radius内半径；parity奇偶性
    CircleRange (float radius, float inner_radius, int parity) {
        const double eps = 1e-8;

        width  = (2 * int(radius + eps) + parity);
        int center = (int)(radius);

        if (width % 2 != parity) {  // fit to parity, pick ceil
            width++;
        }
        height = width;

        is_in_range = new bool[width * width];
        dx = new int[width * width];
        dy = new int[width * width];

        // count为width*width区域中在圆环range内的格子数
        // dx,dy为相对圆心的位置偏移
        count = 0;
        double delta = (parity == 0 ? 0.5 : 0);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < width; j++) {
                double dis_x = fabs(j - center + delta);
                double dis_y = fabs(i - center + delta);
                double dis = sqrt(dis_x * dis_x + dis_y * dis_y);

                if (dis < radius + eps) {
                    if (dis > inner_radius - eps) { // if inc_center is false, exclude the center
                        is_in_range[i * width + j] = true;
                        dx[count] = j - center;
                        dy[count] = i - center;
                        count++;
                    }
                } else {
                    is_in_range[i * width + j] = false;
                }
            }
        }

        x1 = y1 = -center;
        x2 = y2 = width - center - 1;
    }
};

} // namespace gridworld
} // namespace magent

#endif //MAGNET_GRIDWORLD_RANGE_H
